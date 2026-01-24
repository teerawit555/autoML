#scripts/extract_features.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

# =============================================================================
# Config (ลด parameter กระจัดกระจาย)
# =============================================================================
TAIL_FRAC = 0.25          # ใช้ช่วงท้าย 25%
STABLE_K = 60             # ต้องนิ่ง/อยู่ใน band ต่อเนื่องใน "หน้าต่าง" K จุด
ALLOW_M = 1               # วิธี 1: อนุญาตหลุด band ได้ M จุดในหน้าต่าง K
MIN_THR = 1e-4            # กัน threshold เล็กเกินใน t_est_settle_ms
EPS = 1e-12               # กันหารศูนย์


# =============================================================================
# Utility: normalize column names
# =============================================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "time" in out.columns and "time_ms" not in out.columns:
        out = out.rename(columns={"time": "time_ms"})
    if "current" in out.columns and "value" not in out.columns:
        out = out.rename(columns={"current": "value"})

    if "type" not in out.columns:
        out["type"] = "unknown"
    else:
        out["type"] = out["type"].fillna("unknown")

    return out


# =============================================================================
# Spectral features
# =============================================================================
def dominant_freq(x: np.ndarray, fs: float) -> float:
    if len(x) < 8:
        return 0.0
    f, pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
    return float(f[int(np.argmax(pxx))])


def spectral_centroid(x: np.ndarray, fs: float) -> float:
    if len(x) < 8:
        return 0.0
    f, pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
    total = float(np.sum(pxx))
    if total <= 0:
        return 0.0
    return float(np.sum(f * pxx) / total)


# =============================================================================
# Tail features
# =============================================================================
def tail_features(t_ms: np.ndarray, x: np.ndarray) -> dict:
    n_total = len(x)
    if n_total < 8:
        return {"tail_std": 0.0, "tail_p2p": 0.0, "tail_mean_abs_slope": 0.0}

    start_idx = int((1.0 - TAIL_FRAC) * n_total)
    xt = x[start_idx:]
    tt = t_ms[start_idx:]

    dt_s = (tt[1] - tt[0]) * 1e-3 if len(tt) > 1 else 1e-6
    dt_s = max(dt_s, EPS)

    dxdt = np.diff(xt) / dt_s

    return {
        "tail_std": float(np.std(xt)),
        "tail_p2p": float(np.ptp(xt)),
        "tail_mean_abs_slope": float(np.mean(np.abs(dxdt))) if len(dxdt) else 0.0,
    }


# =============================================================================
# Estimated settle time (no low/high)
# =============================================================================
def estimate_settle_time_ms(t_ms: np.ndarray, x: np.ndarray) -> float:
    """
    t_est_settle_ms:
    - threshold = max(3*tail_std, MIN_THR)
    - หาเวลาแรกที่ |x - tail_mean| <= thr แบบต่อเนื่องใน window STABLE_K
    """
    n_total = len(x)
    if n_total < (STABLE_K + 5):
        return float(t_ms[-1])

    tail_start = int((1.0 - TAIL_FRAC) * n_total)
    tail = x[tail_start:]

    mu = float(np.mean(tail))
    sigma = float(np.std(tail))
    thr = max(3.0 * sigma, MIN_THR)

    ok = (np.abs(x - mu) <= thr).astype(np.int32)
    bad = 1 - ok  # 1 = หลุดเงื่อนไข

    bad_cnt = np.convolve(bad, np.ones(STABLE_K, dtype=np.int32), mode="valid")
    idx = np.where(bad_cnt == 0)[0]
    if len(idx) == 0:
        return float(t_ms[-1])

    return float(t_ms[int(idx[0])])


# =============================================================================
# In-band enter time (allow spike) - วิธี 1
# =============================================================================
def t_enter_band_ms_allow_spike(
    t_ms: np.ndarray,
    x: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> float:
    """
    หาเวลาแรกที่สัญญาณ "อยู่ใน band เกือบตลอด" ภายในหน้าต่าง STABLE_K จุด
    โดยอนุญาตให้หลุด band ได้ <= ALLOW_M จุดใน window เดียวกัน
    """
    n_total = len(x)
    if n_total < STABLE_K:
        return float(t_ms[-1])

    ok = ((x >= lo) & (x <= hi)).astype(np.int32)
    bad = 1 - ok  # 1 = หลุด band

    bad_cnt = np.convolve(bad, np.ones(STABLE_K, dtype=np.int32), mode="valid")
    idx = np.where(bad_cnt <= ALLOW_M)[0]
    if len(idx) == 0:
        return float(t_ms[-1])

    return float(t_ms[int(idx[0])])


# =============================================================================
# Main feature extraction
# =============================================================================
def extract_features_from_raw(in_csv: str, out_csv: str) -> None:
    raw = pd.read_csv(in_csv)
    raw = normalize_columns(raw)

    required = ["wave_id", "time_ms", "value"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"{in_csv}: Missing required columns: {missing}. Found: {list(raw.columns)}")

    has_label = "wait_time_ms" in raw.columns

    rows: list[dict] = []

    for wid, group in raw.groupby("wave_id"):
        g = group.sort_values("time_ms")
        t = g["time_ms"].to_numpy(dtype=float)
        x = g["value"].to_numpy(dtype=float)

        # sampling
        dt_s = (t[1] - t[0]) * 1e-3 if len(t) > 1 else 1e-6
        dt_s = max(dt_s, EPS)
        fs = 1.0 / dt_s

        # base stats
        p2p_value = float(np.ptp(x))
        std_value = float(np.std(x))
        t_end_ms = float(t[-1])

        max_slope = 0.0
        mean_abs_slope = 0.0
        if len(x) > 1:
            dxdt = np.diff(x) / dt_s
            max_slope = float(np.max(np.abs(dxdt)))
            mean_abs_slope = float(np.mean(np.abs(dxdt)))

        feat: dict = {
            "wave_id": int(wid),
            "type": str(g["type"].iloc[0]),
            "n_samples": int(len(x)),
            "t_end_ms": t_end_ms,

            "mean_value": float(np.mean(x)),
            "std_value": std_value,
            "min_value": float(np.min(x)),
            "max_value": float(np.max(x)),
            "p2p_value": p2p_value,

            "max_slope": max_slope,
            "mean_abs_slope": mean_abs_slope,

            "energy_total": float(np.sum(x ** 2)),

            "dominant_freq_hz": dominant_freq(x, fs),
            "spectral_centroid_hz": spectral_centroid(x, fs),
        }

        # tail features
        tail_feat = tail_features(t, x)
        feat.update(tail_feat)

        # estimated settle time (no low/high)
        t_est = estimate_settle_time_ms(t, x)
        feat["t_est_settle_ms"] = t_est

        # in-band settle time + band_width
        if ("low_limit" in g.columns) and ("high_limit" in g.columns):
            lo = g["low_limit"].to_numpy(dtype=float)
            hi = g["high_limit"].to_numpy(dtype=float)

            t_enter = t_enter_band_ms_allow_spike(t, x, lo, hi)
            band_width = float(np.median(hi - lo))

            feat["t_enter_band_ms"] = t_enter
            feat["band_width"] = band_width
        else:
            feat["t_enter_band_ms"] = t_end_ms
            feat["band_width"] = 0.0

        # ==============================
        # Ratio / normalized features (NEW) - เหมือนเดิม แต่ไม่ซ้ำ eps
        # ==============================
        feat["band_to_p2p"] = feat["band_width"] / (p2p_value + EPS)
        feat["tail_noise_ratio"] = feat["tail_std"] / (std_value + EPS)
        feat["enter_vs_settle"] = feat["t_enter_band_ms"] / (t_est + 1e-6)
        feat["slope_norm"] = max_slope / (p2p_value + EPS)

        # ✅ ตัวที่คุณใช้แก้ error เดิม
        feat["t_enter_norm"] = feat["t_enter_band_ms"] / (t_end_ms + EPS)
        feat["enter_minus_est"] = feat["t_enter_band_ms"] - t_est
        feat["enter_margin_ms"] = t_end_ms - feat["t_enter_band_ms"]

        # train-only label
        if has_label:
            feat["wait_time_ms"] = float(g["wait_time_ms"].iloc[0])

        rows.append(feat)

    out = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"✅ Saved: {out_path} (label_present={has_label})")


def main():
    ap = argparse.ArgumentParser("Extract features for TRAIN and/or PRED in one run")

    ap.add_argument("--in_train", required=False, help="Raw train CSV (has wait_time_ms)")
    ap.add_argument("--out_train", required=False, help="Output train features CSV")

    ap.add_argument("--in_pred", required=False, help="Raw pred CSV (no wait_time_ms ok)")
    ap.add_argument("--out_pred", required=False, help="Output pred features CSV")

    args = ap.parse_args()

    if not args.in_train and not args.in_pred:
        raise ValueError("You must provide at least --in_train or --in_pred")

    if args.in_train and not args.out_train:
        raise ValueError("When using --in_train, you must provide --out_train")

    if args.in_pred and not args.out_pred:
        raise ValueError("When using --in_pred, you must provide --out_pred")

    if args.in_train:
        extract_features_from_raw(args.in_train, args.out_train)

    if args.in_pred:
        extract_features_from_raw(args.in_pred, args.out_pred)

    print("✨ Done.")


if __name__ == "__main__":
    main()
