# scripts/eval_ag_v23.py
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

# =========================
# Config (keep consistent)
# =========================
FAST_MS = 0.1

COLS_TO_DROP = ["force_mA", "range_V", "temp_C"]
DROP_ALWAYS = ["type"]  # don't feed into model
LABEL_LEAK_COLS = ["wait_time_ms", "wait_time_log", "is_fast", "is_zero"]


# =========================
# Helpers
# =========================
def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def load_json(path: str, default: Any = None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path: str, default: str = "") -> str:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def align_columns(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in required_cols:
        if c not in out.columns:
            out[c] = 0.0
    return out[required_cols]


def extract_proba_class1(p) -> np.ndarray:
    if isinstance(p, pd.DataFrame):
        if 1 in p.columns:
            return p[1].to_numpy(float)
        return p.iloc[:, -1].to_numpy(float)
    arr = np.asarray(p, float)
    return arr[:, 1] if arr.ndim == 2 else arr


def load_fast_thresholds(model_path: str, single_fallback: float) -> tuple[float, float, str]:
    """
    Priority:
      1) fast_thresholds.json -> {"thr_high":..., "thr_low":...}
      2) fast_thresholds.txt  -> "high,low" or "high low"
      3) zero_threshold.txt   -> single
    """
    p_json = os.path.join(model_path, "fast_thresholds.json")
    if os.path.exists(p_json):
        j = load_json(p_json, {})
        th = float(j.get("thr_high", single_fallback))
        tl = float(j.get("thr_low", th))
        return th, tl, "fast_thresholds.json"

    p_txt = os.path.join(model_path, "fast_thresholds.txt")
    if os.path.exists(p_txt):
        s = load_text(p_txt, "")
        parts = [x for x in s.replace(",", " ").split() if x.strip()]
        if len(parts) >= 2:
            th, tl = float(parts[0]), float(parts[1])
            return th, tl, "fast_thresholds.txt"
        if len(parts) == 1:
            th = float(parts[0])
            return th, th, "fast_thresholds.txt(single)"

    # legacy single
    return single_fallback, single_fallback, "zero_threshold.txt(single)"


def compute_fast_metrics(is_fast_pred: np.ndarray, y_true_ms: np.ndarray) -> dict[str, float]:
    fast_true = y_true_ms <= FAST_MS + 1e-12
    tp = int(np.sum(is_fast_pred & fast_true))
    fn = int(np.sum((~is_fast_pred) & fast_true))
    fp = int(np.sum(is_fast_pred & (~fast_true)))
    tn = int(np.sum((~is_fast_pred) & (~fast_true)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fp_rate = fp / max(fp + tn, 1)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "fast_precision": float(precision),
        "fast_recall": float(recall),
        "fast_fp_rate": float(fp_rate),
        "pred_fast": int(is_fast_pred.sum()),
        "true_fast": int(fast_true.sum()),
    }


def threshold_scan(proba_fast: np.ndarray, wait_pred_all: np.ndarray, y_true_ms: np.ndarray,
                   thr_min: float = 0.10, thr_max: float = 0.90, steps: int = 161) -> pd.DataFrame:
    thrs = np.linspace(thr_min, thr_max, steps)
    rows = []
    for thr in thrs:
        is_fast = proba_fast >= thr
        y_hat = np.where(is_fast, FAST_MS, wait_pred_all)
        y_hat = np.clip(y_hat, 0.0, None)

        mae_all = float(np.mean(np.abs(y_hat - y_true_ms)))
        slow_mask = y_true_ms > FAST_MS + 1e-12
        mae_slow = float(np.mean(np.abs(y_hat[slow_mask] - y_true_ms[slow_mask]))) if np.any(slow_mask) else float("nan")

        m = compute_fast_metrics(is_fast, y_true_ms)
        rows.append({
            "thr": float(thr),
            "mae_all": mae_all,
            "mae_slow": mae_slow,
            "fast_precision": m["fast_precision"],
            "fast_fp_rate": m["fast_fp_rate"],
            "fast_recall": m["fast_recall"],
            "pred_fast": m["pred_fast"],
        })
    return pd.DataFrame(rows)


# =========================
# Eval
# =========================
def eval_model(
    model_path: str,
    test_csv: str,
    out_dir: str,
    *,
    label_col: str = "wait_time_ms",
    scan_thr_min: float = 0.10,
    scan_thr_max: float = 0.90,
    scan_steps: int = 161,
) -> None:
    fast_model_path = os.path.join(model_path, "zero_clf")
    reg_model_path  = os.path.join(model_path, "wait_reg")

    if not os.path.isdir(fast_model_path):
        raise FileNotFoundError(f"missing model dir: {fast_model_path}")
    if not os.path.isdir(reg_model_path):
        raise FileNotFoundError(f"missing model dir: {reg_model_path}")

    fast_feature_cols = load_json(os.path.join(model_path, "zero_feature_cols.json"))
    reg_feature_cols  = load_json(os.path.join(model_path, "reg_feature_cols.json"))
    if not fast_feature_cols:
        raise FileNotFoundError(f"missing zero_feature_cols.json in: {model_path}")
    if not reg_feature_cols:
        raise FileNotFoundError(f"missing reg_feature_cols.json in: {model_path}")

    single_thr = float(load_text(os.path.join(model_path, "zero_threshold.txt"), "0.5"))
    thr_high, thr_low, thr_src = load_fast_thresholds(model_path, single_thr)

    print(f"🧪 EVAL: {test_csv}")
    print(f"model_path={model_path}")
    if thr_high == thr_low:
        print(f"FAST threshold(single) = {thr_high:.3f} | src={thr_src} | FAST_MS={FAST_MS}")
    else:
        print(f"FAST thresholds(dual) = high={thr_high:.3f} low={thr_low:.3f} | src={thr_src} | FAST_MS={FAST_MS}")

    fast_clf = TabularPredictor.load(fast_model_path)
    wait_reg = TabularPredictor.load(reg_model_path)

    df_raw = pd.read_csv(test_csv)
    if label_col not in df_raw.columns:
        raise ValueError(f"test_csv must contain label column '{label_col}'")

    # keep wave_id/type for reporting (type might not exist)
    wave_id = df_raw["wave_id"] if "wave_id" in df_raw.columns else pd.Series(np.arange(len(df_raw)), name="wave_id")
    type_series = df_raw["type"].astype(str) if "type" in df_raw.columns else pd.Series([""] * len(df_raw), name="type")

    y_true = pd.to_numeric(df_raw[label_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y_true)
    if not np.all(ok):
        bad = int(np.sum(~ok))
        print(f"[WARN] dropping {bad} rows with NaN/inf label")
        df_raw = df_raw.loc[ok].reset_index(drop=True)
        wave_id = wave_id.loc[ok].reset_index(drop=True)
        type_series = type_series.loc[ok].reset_index(drop=True)
        y_true = y_true[ok]

    # drop leak/meta from features
    df_feat = df_raw.drop(columns=DROP_ALWAYS + COLS_TO_DROP + LABEL_LEAK_COLS, errors="ignore").copy()

    # ----- Stage A proba -----
    X_fast = align_columns(df_feat.drop(columns=["wave_id"], errors="ignore"), fast_feature_cols)
    proba_fast = extract_proba_class1(fast_clf.predict_proba(X_fast))

    # ----- Stage B reg (predict for ALL rows for eval/scan stability) -----
    X_reg_all = align_columns(df_feat.drop(columns=["wave_id"], errors="ignore"), reg_feature_cols)
    wait_log_all = wait_reg.predict(X_reg_all)
    wait_pred_all = np.expm1(np.asarray(wait_log_all, dtype=float))
    wait_pred_all = np.clip(wait_pred_all, 0.0, None)

    # ----- Deployed decision (single or dual) -----
    if thr_high == thr_low:
        is_fast = proba_fast >= thr_high
        fast_zone = np.where(is_fast, "FAST", "SLOW")
    else:
        # Dual = strict OR soft-zone (no extra forcing rule; pure probability band)
        is_fast_strict = proba_fast >= thr_high
        is_fast_soft   = (proba_fast >= thr_low) & (proba_fast < thr_high)
        is_fast = is_fast_strict | is_fast_soft
        fast_zone = np.where(is_fast_strict, "FAST_STRICT", np.where(is_fast_soft, "FAST_SOFT", "SLOW"))

    y_hat = np.where(is_fast, FAST_MS, wait_pred_all)
    y_hat = np.clip(y_hat, 0.0, None)

    # ----- Metrics -----
    mae_all = float(np.mean(np.abs(y_hat - y_true)))
    slow_mask = y_true > FAST_MS + 1e-12
    mae_slow = float(np.mean(np.abs(y_hat[slow_mask] - y_true[slow_mask]))) if np.any(slow_mask) else float("nan")

    m = compute_fast_metrics(is_fast, y_true)

    print("\n=== EVAL METRICS ===")
    print(f"rows={len(df_feat)}")
    print(f"MAE(all)={mae_all:.6f} | MAE(slow-only)={mae_slow:.6f}")
    print(
        f"FAST precision={m['fast_precision']:.6f} recall={m['fast_recall']:.6f} fp_rate={m['fast_fp_rate']:.6f} "
        f"| pred_fast={m['pred_fast']} true_fast={m['true_fast']}"
    )
    print(f"proba_is_fast min/mean/max = {proba_fast.min():.6f} {proba_fast.mean():.6f} {proba_fast.max():.6f}")

    # ----- Threshold scan report -----
    scan = threshold_scan(
        proba_fast=proba_fast,
        wait_pred_all=wait_pred_all,
        y_true_ms=y_true,
        thr_min=scan_thr_min,
        thr_max=scan_thr_max,
        steps=scan_steps,
    )

    top_mae = scan.sort_values("mae_all", ascending=True).head(10)
    top_prec = scan.sort_values(["fast_precision", "fast_fp_rate", "mae_all"], ascending=[False, True, True]).head(10)

    print("\nTop 10 thresholds by MAE:")
    print(top_mae[["thr", "mae_all", "fast_precision", "fast_fp_rate", "fast_recall", "pred_fast"]].to_string(index=False))

    print("\nTop 10 thresholds by PRECISION (then FP-rate, then MAE):")
    print(top_prec[["thr", "fast_precision", "fast_fp_rate", "fast_recall", "mae_all", "pred_fast"]].to_string(index=False))

    # ----- Per-type breakdown (if type exists / not empty) -----
    diag = pd.DataFrame({
        "wave_id": wave_id,
        "type": type_series,
        "y_true_ms": y_true,
        "proba_is_fast": proba_fast,
        "fast_zone": fast_zone,
        "pred_is_fast": is_fast.astype(int),
        "reg_wait_pred_ms": wait_pred_all,
        "pred_wait_ms": y_hat,
    })
    diag["abs_error"] = np.abs(diag["pred_wait_ms"].to_numpy(float) - diag["y_true_ms"].to_numpy(float))
    diag["is_fast_true"] = (diag["y_true_ms"] <= FAST_MS + 1e-12).astype(int)

    if (diag["type"].astype(str).str.len() > 0).any():
        by_type = []
        for t, g in diag.groupby("type"):
            y_t = g["y_true_ms"].to_numpy(float)
            is_fast_t = g["pred_is_fast"].to_numpy(int).astype(bool)
            mae_t = float(np.mean(np.abs(g["pred_wait_ms"].to_numpy(float) - y_t)))
            slow_mask_t = y_t > FAST_MS + 1e-12
            mae_slow_t = float(np.mean(np.abs(g["pred_wait_ms"].to_numpy(float)[slow_mask_t] - y_t[slow_mask_t]))) if np.any(slow_mask_t) else float("nan")
            mm = compute_fast_metrics(is_fast_t, y_t)
            by_type.append({
                "type": t,
                "rows": int(len(g)),
                "MAE(all)": mae_t,
                "MAE(slow-only)": mae_slow_t,
                "fast_precision": mm["fast_precision"],
                "fast_recall": mm["fast_recall"],
                "fast_fp_rate": mm["fast_fp_rate"],
                "pred_fast": mm["pred_fast"],
                "true_fast": mm["true_fast"],
            })
        by_type_df = pd.DataFrame(by_type).sort_values(["rows", "type"], ascending=[False, True])
        print("\n=== Per-type summary ===")
        print(by_type_df.to_string(index=False))
    else:
        by_type_df = None
        print("\n[INFO] 'type' not found (or empty) -> skipping per-type summary")

    # ----- Save outputs -----
    _ensure_dir(out_dir)
    diag_path = os.path.join(out_dir, "eval_diagnostics.csv")
    scan_path = os.path.join(out_dir, "eval_threshold_scan.csv")
    metrics_path = os.path.join(out_dir, "eval_metrics.json")
    type_path = os.path.join(out_dir, "eval_by_type.csv")

    diag.sort_values("abs_error", ascending=False).to_csv(diag_path, index=False)
    scan.to_csv(scan_path, index=False)

    metrics = {
        "rows": int(len(df_feat)),
        "FAST_MS": FAST_MS,
        "thr_high": float(thr_high),
        "thr_low": float(thr_low),
        "thr_source": thr_src,
        "mae_all": mae_all,
        "mae_slow_only": mae_slow,
        **{k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in m.items()},
        "proba_min": float(proba_fast.min()),
        "proba_mean": float(proba_fast.mean()),
        "proba_max": float(proba_fast.max()),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if by_type_df is not None:
        by_type_df.to_csv(type_path, index=False)

    print("\n✅ Saved:")
    print(" -", diag_path)
    print(" -", scan_path)
    print(" -", metrics_path)
    if by_type_df is not None:
        print(" -", type_path)


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--test", dest="test_csv", required=True, help="CSV that contains ground-truth label (wait_time_ms)")
    ap.add_argument("--out-dir", default="data/processed/eval/ag_v23_eval")
    ap.add_argument("--label-col", default="wait_time_ms")
    ap.add_argument("--scan-min", type=float, default=0.10)
    ap.add_argument("--scan-max", type=float, default=0.90)
    ap.add_argument("--scan-steps", type=int, default=161)
    args = ap.parse_args()

    eval_model(
        model_path=args.model_path,
        test_csv=args.test_csv,
        out_dir=args.out_dir,
        label_col=args.label_col,
        scan_thr_min=args.scan_min,
        scan_thr_max=args.scan_max,
        scan_steps=args.scan_steps,
    )


if __name__ == "__main__":
    main()
