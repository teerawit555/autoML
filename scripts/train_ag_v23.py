from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt

'''
fast FP = “ลดเวลารอให้สั้นเกินจริง” → อันตรายสุด

fast FN = “รอนานเกิน” → แค่ช้า แต่ปลอดภัยกว่า
'''

# =========================
# Config
# =========================
COLS_TO_DROP = ["force_mA", "range_V", "temp_C"]
DROP_ALWAYS = ["type"]

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_SAVE_PATH = f"AutogluonModels/ag-v22-{ts}"

# policy: “fast” class = 0.1ms (sine/glitch should map here)
FAST_MS = 0.1

# threshold picking policy (สำคัญมาก)
# เราจะ “กัน FP-fast” ก่อน เพราะ FP ทำให้ 8ms -> 0.1ms เละ
MIN_FAST_PRECISION = 0.999   # อยากให้ fast ที่ทายออกมาถูกจริง
MAX_FAST_FP_RATE   = 0.0002  # FP allowed over slow (0.1% ของ slow)


# =========================
# IO helpers
# =========================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def save_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(text))

def append_log(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")

def log_print(log_path: str, msg: str) -> None:
    print(msg)
    append_log(log_path, msg)


# =========================
# Data helpers
# =========================
def align_columns(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in required_cols:
        if c not in out.columns:
            out[c] = 0.0
    return out[required_cols]

def proba_class1(p) -> np.ndarray:
    if hasattr(p, "columns") and (1 in list(p.columns)):
        return p[1].to_numpy()
    arr = np.asarray(p)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    return arr.astype(float)

def fast_metrics_at_thr(
    proba_is_fast: np.ndarray,
    y_true_ms: np.ndarray,
    thr: float,
    fast_ms: float = 0.1,
) -> dict:
    y_true_ms = np.asarray(y_true_ms, float)
    proba_is_fast = np.asarray(proba_is_fast, float)

    zt = (y_true_ms <= fast_ms + 1e-12)         # true fast
    zp = (proba_is_fast >= thr)                 # predicted fast

    tp = int(np.sum(zp & zt))
    fp = int(np.sum(zp & (~zt)))
    fn = int(np.sum((~zp) & zt))
    tn = int(np.sum((~zp) & (~zt)))

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)

    slow_total = int(np.sum(~zt))
    fp_rate = fp / max(slow_total, 1)           # FP over slow samples

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "fp_rate": float(fp_rate),
        "pred_fast": int(np.sum(zp)),
    }

def plot_confusion_heatmap(tp, fp, fn, tn, title: str, save_path: str | None = None, normalize: str = "row"):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=float)

    if normalize == "row":
        cmn = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
        subtitle = " (row-normalized %)"
    elif normalize == "all":
        cmn = cm / (cm.sum() + 1e-12)
        subtitle = " (overall %)"
    else:
        cmn = None
        subtitle = " (counts)"

    fig, ax = plt.subplots(figsize=(6, 4.5))
    show = cmn if cmn is not None else cm
    im = ax.imshow(show)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred SLOW", "Pred FAST"])
    ax.set_yticklabels(["True SLOW", "True FAST"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title + subtitle)

    for i in range(2):
        for j in range(2):
            cnt = int(cm[i, j])
            if cmn is None:
                txt = f"{cnt}"
            else:
                pct = 100.0 * float(cmn[i, j])
                txt = f"{cnt}\n({pct:.2f}%)"
            ax.text(j, i, txt, ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if save_path:
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[saved] {save_path}")

    plt.close(fig)





def perm_importance_fp_safe(
    predictor,
    X: pd.DataFrame,
    y_true_ms: np.ndarray,
    *,
    thr: float,
    fast_ms: float = 0.1,
    features: list[str] | None = None,
    n_shuffle: int = 5,
    topk: int = 25,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Permutation importance ที่วัด "ความแย่ลง" ของ:
    - fp_rate (สำคัญสุด กัน 8ms -> 0.1ms)
    - precision (กัน FP)
    - recall (ดูด้วยว่า drop เยอะไหม)
    """
    rng = np.random.default_rng(random_state)

    # baseline
    base_proba = proba_class1(predictor.predict_proba(X))
    base_m = fast_metrics_at_thr(base_proba, y_true_ms, thr, fast_ms)

    feats = features or list(X.columns)

    rows = []
    for f in feats:
        fp_deltas = []
        prec_deltas = []
        rec_deltas = []

        for _ in range(n_shuffle):
            Xp = X.copy()

            col = Xp[f].to_numpy()
            col_shuf = col.copy()
            rng.shuffle(col_shuf)
            Xp[f] = col_shuf

            p = proba_class1(predictor.predict_proba(Xp))
            m = fast_metrics_at_thr(p, y_true_ms, thr, fast_ms)

            # + = worse (FP เพิ่ม, precision/recall ลด)
            fp_deltas.append(m["fp_rate"] - base_m["fp_rate"])
            prec_deltas.append(base_m["precision"] - m["precision"])
            rec_deltas.append(base_m["recall"] - m["recall"])

        rows.append({
            "feature": f,
            "fp_rate_increase_mean": float(np.mean(fp_deltas)),
            "fp_rate_increase_std":  float(np.std(fp_deltas)),
            "precision_drop_mean":   float(np.mean(prec_deltas)),
            "precision_drop_std":    float(np.std(prec_deltas)),
            "recall_drop_mean":      float(np.mean(rec_deltas)),
            "recall_drop_std":       float(np.std(rec_deltas)),
        })

    out = pd.DataFrame(rows)

    # เรียงตาม "FP-rate แย่ลง" ก่อน แล้ว precision drop
    out = out.sort_values(
        ["fp_rate_increase_mean", "precision_drop_mean", "recall_drop_mean"],
        ascending=[False, False, False],
    )

    print(
        f"\nPermutation Importance (FP-safe @thr={thr:.3f}) "
        f"baseline: fp_rate={base_m['fp_rate']:.6f} precision={base_m['precision']:.6f} recall={base_m['recall']:.6f} pred_fast={base_m['pred_fast']}"
    )
    print(out.head(topk).to_string(index=False))
    return out


# =========================
# Threshold scan / pick (FP-safe)
# =========================
def threshold_scan_report(
    proba_is_fast: np.ndarray,
    wait_pred: np.ndarray,
    y_true: np.ndarray,
    thr_min: float = 0.10,
    thr_max: float = 0.90,
    steps: int = 161,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    is_fast_true = (y_true <= FAST_MS + 1e-12)

    rows = []
    thr_grid = np.linspace(thr_min, thr_max, steps)

    for thr in thr_grid:
        is_fast_pred = proba_is_fast >= thr
        y_hat = np.where(is_fast_pred, FAST_MS, wait_pred)
        mae_all = float(np.mean(np.abs(y_hat - y_true)))

        zp = is_fast_pred
        zt = is_fast_true

        tp = int(np.sum(zp & zt))          # true fast predicted fast
        fn = int(np.sum((~zp) & zt))       # true fast predicted slow
        fp = int(np.sum(zp & (~zt)))       # true slow predicted fast  (อันนี้แหละที่ทำให้เละ)
        tn = int(np.sum((~zp) & (~zt)))

        fast_recall = tp / max(int(np.sum(zt)), 1)
        fast_fn_rate = fn / max(int(np.sum(zt)), 1)

        # precision & fp-rate (สำคัญ)
        fast_precision = tp / max(tp + fp, 1)
        slow_total = int(np.sum(~zt))
        fast_fp_rate = fp / max(slow_total, 1)  # FP over slow samples

        rows.append({
            "thr": float(thr),
            "mae_all": mae_all,
            "fast_recall": float(fast_recall),
            "fast_fn_rate": float(fast_fn_rate),
            "fast_precision": float(fast_precision),
            "fast_fp_rate": float(fast_fp_rate),
            "TP_fast": tp,
            "FN_fast": fn,
            "FP_fast": fp,
            "TN_slow": tn,
            "pred_fast": int(np.sum(zp)),
        })

    return pd.DataFrame(rows)



MIN_FAST_RECALL = 0.98  # ปรับได้ตามที่รับได้

def pick_threshold_fp_safe(
    scan: pd.DataFrame,
    *,
    min_precision: float = MIN_FAST_PRECISION,
    max_fp_rate: float = MAX_FAST_FP_RATE,
    min_recall: float = MIN_FAST_RECALL,
) -> tuple[float, float, str]:
    s = scan.copy()

    required = ["thr", "mae_all", "fast_precision", "fast_fp_rate", "fast_recall"]
    missing = [c for c in required if c not in s.columns]
    if missing:
        raise ValueError(f"scan missing columns: {missing}")

    for c in required:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.dropna(subset=required)

    cand = s[
        (s["fast_precision"] >= min_precision) &
        (s["fast_fp_rate"] <= max_fp_rate) &
        (s["fast_recall"] >= min_recall)
    ].copy()

    if len(cand) > 0:
        row = cand.sort_values(["mae_all", "fast_recall"], ascending=[True, False]).iloc[0]
        return float(row["thr"]), float(row["mae_all"]), "fp_safe(precision&fp_rate&recall constrained)"

    # fallback: เลือก precision สูงสุดก่อน แล้ว fp_rate ต่ำ แล้ว mae ต่ำ
    row = s.sort_values(["fast_precision", "fast_fp_rate", "mae_all"], ascending=[False, True, True]).iloc[0]
    return float(row["thr"]), float(row["mae_all"]), "fallback(precision_best_then_fp_then_mae)"

def pick_dual_thresholds(
    scan: pd.DataFrame,
    *,
    # strict (T_high): FP-safe
    min_precision_high: float = 0.999,
    max_fp_rate_high: float = 0.0002,
    # soft (T_low): recall-oriented แต่ยังคุม FP (สำหรับ MAYBE zone)
    min_recall_low: float = 0.995,     # <-- แนะนำให้ผ่อนจาก 0.998 เพื่อให้มีพื้นที่
    max_fp_rate_low: float = 0.0020,   # <-- แนะนำให้ผ่อนจาก 0.0010 เพื่อให้มี MAYBE zone
) -> tuple[float, float, str]:
    s = scan.copy()
    required = ["thr", "mae_all", "fast_precision", "fast_fp_rate", "fast_recall"]
    missing = [c for c in required if c not in s.columns]
    if missing:
        raise ValueError(f"scan missing columns: {missing}")

    for c in required:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.dropna(subset=required)

    # -------------------------
    # T_high: strict FP-safe
    # -------------------------
    cand_high = s[
        (s["fast_precision"] >= min_precision_high) &
        (s["fast_fp_rate"] <= max_fp_rate_high)
    ].copy()

    if len(cand_high) > 0:
        row_high = cand_high.sort_values(
            ["fast_fp_rate", "mae_all", "fast_recall", "thr"],
            ascending=[True, True, False, True],
        ).iloc[0]
        thr_high = float(row_high["thr"])
        reason_high = "high=fp_safe"
    else:
        row_high = s.sort_values(
            ["fast_precision", "fast_fp_rate", "mae_all", "thr"],
            ascending=[False, True, True, True],
        ).iloc[0]
        thr_high = float(row_high["thr"])
        reason_high = "high=fallback(precision_best_then_fp_then_mae)"

    # -------------------------
    # T_low: MAYBE zone threshold
    # choose LOWEST thr under thr_high that still satisfies (recall>=, fp_rate<=)
    # -------------------------
    s_low = s[s["thr"] <= thr_high].copy()
    cand_low = s_low[
        (s_low["fast_recall"] >= min_recall_low) &
        (s_low["fast_fp_rate"] <= max_fp_rate_low)
    ].copy()

    if len(cand_low) > 0:
        row_low = cand_low.sort_values(
            ["thr", "fast_fp_rate", "fast_precision", "mae_all"],
            ascending=[True, True, False, True],
        ).iloc[0]
        thr_low = float(row_low["thr"])
        reason_low = "low=lowest_thr_with_recall_and_fp_caps"
    else:
        # ไม่มีพื้นที่ MAYBE จริง ๆ -> collapse
        thr_low = thr_high
        reason_low = "low=none_under_high"

    # enforce
    thr_low = float(min(thr_low, thr_high))
    reason = f"{reason_high} | {reason_low}"
    return thr_high, thr_low, reason


# def pick_dual_thresholds(
#     scan: pd.DataFrame,
#     *,
#     # strict (T_high): FP-safe
#     min_precision_high: float = 0.999,
#     max_fp_rate_high: float = 0.0002,
#     # soft (T_low): recall-oriented แต่ยังคุม FP
#     min_recall_low: float = 0.998,
#     max_fp_rate_low: float = 0.0010,
# ) -> tuple[float, float, str]:
#     s = scan.copy()
#     required = ["thr", "mae_all", "fast_precision", "fast_fp_rate", "fast_recall"]
#     missing = [c for c in required if c not in s.columns]
#     if missing:
#         raise ValueError(f"scan missing columns: {missing}")

#     for c in required:
#         s[c] = pd.to_numeric(s[c], errors="coerce")
#     s = s.dropna(subset=required)

#     # -------------------------
#     # T_high: strict FP-safe
#     # -------------------------
#     cand_high = s[
#         (s["fast_precision"] >= min_precision_high)
#         & (s["fast_fp_rate"] <= max_fp_rate_high)
#     ].copy()

#     if len(cand_high) > 0:
#         row_high = cand_high.sort_values(
#             ["fast_fp_rate", "mae_all", "fast_recall"],
#             ascending=[True, True, False],
#         ).iloc[0]
#         thr_high = float(row_high["thr"])
#         reason_high = "high=fp_safe"
#     else:
#         row_high = s.sort_values(
#             ["fast_precision", "fast_fp_rate", "mae_all"],
#             ascending=[False, True, True],
#         ).iloc[0]
#         thr_high = float(row_high["thr"])
#         reason_high = "high=fallback(precision_best_then_fp_then_mae)"

#     # -------------------------
#     # T_low: soft zone (prefer high recall)
#     # must be <= thr_high
#     # -------------------------
#     # T_low: soft zone (maximize recall by lowering threshold)
#     s_low = s[s["thr"] <= thr_high].copy()
#     cand_low = s_low[
#         (s_low["fast_fp_rate"] <= max_fp_rate_low)
#         & (s_low["fast_recall"] >= min_recall_low)
#     ].copy()

#     if len(cand_low) > 0:
#         # choose LOWEST threshold that satisfies constraints
#         row_low = cand_low.sort_values(
#             ["thr", "fast_fp_rate", "mae_all"],
#             ascending=[True, True, True],
#         ).iloc[0]
#         thr_low = float(row_low["thr"])
#         reason_low = "low=lowest_thr_with_recall_and_fp_caps"
#     else:
#         thr_low = thr_high
#         reason_low = "low=none_under_high"


#     # s_low = s[s["thr"] <= thr_high].copy()
#     # cand_low = s_low[
#     #     (s_low["fast_recall"] >= min_recall_low)
#     #     & (s_low["fast_fp_rate"] <= max_fp_rate_low)
#     # ].copy()

#     # if len(cand_low) > 0:
#     #     row_low = cand_low.sort_values(
#     #         ["fast_fp_rate", "mae_all", "fast_precision"],
#     #         ascending=[True, True, False],
#     #     ).iloc[0]
#     #     thr_low = float(row_low["thr"])
#     #     reason_low = "low=high_recall_fp_capped"
#     # else:
#     #     # fallback: take best MAE under thr_high
#     #     row_low = s_low.sort_values(["mae_all"], ascending=True).iloc[0]
#     #     thr_low = float(row_low["thr"])
#     #     reason_low = "low=fallback(best_mae_under_high)"

#     # enforce ordering
#     thr_low = float(min(thr_low, thr_high))

#     reason = f"{reason_high} | {reason_low}"
#     return thr_high, thr_low, reason


# def pick_threshold_fp_safe(
#     scan: pd.DataFrame,
#     *,
#     min_precision: float = MIN_FAST_PRECISION,
#     max_fp_rate: float = MAX_FAST_FP_RATE,
# ) -> tuple[float, float, str]:
#     """
#     เลือก threshold โดย “กัน FP-fast” ก่อน (กันเคส 8ms -> 0.1ms)
#     1) กรอง candidate ที่ precision >= min_precision และ fp_rate <= max_fp_rate
#        แล้วเลือก mae_all ต่ำสุด
#     2) ถ้าไม่มี candidate: เลือก precision สูงสุดก่อน แล้ว mae_all ต่ำสุด
#     """
#     s = scan.copy()

#     cand = s[(s["fast_precision"] >= min_precision) & (s["fast_fp_rate"] <= max_fp_rate)].copy()
#     if len(cand) > 0:
#         row = cand.sort_values("mae_all", ascending=True).iloc[0]
#         return float(row["thr"]), float(row["mae_all"]), "fp_safe(precision&fp_rate constrained)"

#     row = s.sort_values(["fast_precision", "mae_all"], ascending=[False, True]).iloc[0]
    
#     return float(row["thr"]), float(row["mae_all"]), "fallback(precision_best_then_mae)"


# =========================
# Training
# =========================
def train(
    data_path: str,
    label: str,
    model_dir: str | None,
    presets: str,
    time_limit: int,
) -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Input file not found: "{data_path}"')

    df = pd.read_csv(data_path)

    if "wave_id" not in df.columns:
        df["wave_id"] = np.arange(len(df), dtype=int)

    df_train = df.dropna(subset=[label]).reset_index(drop=True)
    if len(df_train) == 0:
        raise ValueError(f"No rows with label '{label}' found in training file.")

    # stage-A label: is_fast (<=0.1ms)
    df_train["is_fast"] = (df_train[label].astype(float) <= FAST_MS + 1e-12).astype(int)

    # drop unstable meta
    df_train = df_train.drop(columns=DROP_ALWAYS, errors="ignore")

    # check classes
    u = df_train["is_fast"].unique()
    if len(u) < 2:
        raise ValueError(
            f"is_fast has only 1 class in training data: {u}. "
            f"ต้องมีทั้งเคส <=0.1ms และ >0.1ms"
        )

    save_path = model_dir or DEFAULT_SAVE_PATH
    zero_model_path = os.path.join(save_path, "zero_clf")
    reg_model_path = os.path.join(save_path, "wait_reg")
    _ensure_dir(save_path)

    log_path = os.path.join(save_path, f"train_log_{ts}.txt")
    append_log(log_path, f"=== TRAIN LOG START {ts} ===")

    gpu_count = 1 if torch.cuda.is_available() else 0
    log_print(log_path, f"🚀 Training device: {'GPU (CUDA)' if gpu_count > 0 else 'CPU'}")
    log_print(log_path, f"data_path={data_path}")
    log_print(log_path, f"rows={len(df_train)} cols={len(df_train.columns)}")
    log_print(log_path, f"presets={presets} time_limit={time_limit}s")
    log_print(log_path, f"FAST_MS={FAST_MS}")

    # distribution
    fast_cnt = int((df_train["is_fast"] == 1).sum())
    slow_cnt = int((df_train["is_fast"] == 0).sum())
    log_print(log_path, f"label distribution: fast={fast_cnt} slow={slow_cnt} fast_ratio={fast_cnt/max(len(df_train),1):.4f}")

    # split time
    t_zero = max(30, int(time_limit * 0.25))
    t_reg = max(60, int(time_limit * 0.75))
    log_print(log_path, f"stage time: clf={t_zero}s reg={t_reg}s")

    cols_to_drop_found = [c for c in COLS_TO_DROP if c in df_train.columns]
    if cols_to_drop_found:
        log_print(log_path, f"Dropping meta columns: {cols_to_drop_found}")

    # =========================
    # Stage A: Fast classifier (binary)
    # =========================
    log_print(log_path, "\n=== Stage A: is_fast classifier ===")

    df_zero = df_train.drop(columns=cols_to_drop_found, errors="ignore").copy()
    df_zero = df_zero.drop(columns=[label, "wave_id"], errors="ignore")
    df_zero = df_zero.drop(columns=["sd", "low_limit", "high_limit"], errors="ignore")

    df_zero["sample_weight"] = np.where(df_zero["is_fast"] == 0, 1.5, 1.0)

    # (optional) drop features you don't want classifier to use
    ZERO_DROP_EXTRA = [
        # identity flags (หลอกโมเดล)
        "logic_is_periodic",
        "logic_is_ringing",
        "dbg_label_reason",

        # FFT identity-heavy (ทำให้ periodic ถูกมองว่าไม่นิ่ง)
        # "fft_peak_freq_hz",
        # "fft_peak_power_ratio",
        # "fft_lowfreq_power_ratio",
        # "fft_peak_to_2nd_ratio",
        # "fft_spectral_entropy",

        # duplicate-ish raw stats ถ้ามี (แต่ระวัง อย่าไปตัด stability หมด)
        # "std_all", "std_mid",   # จะ drop ก็ได้ แต่ไม่จำเป็น ถ้ามันช่วยจริงก็เก็บ
        # "ringing_energy",       # จะ drop ได้ถ้าซ้ำกับ peak/spacing แล้ว
    ]

    # ZERO_DROP_EXTRA = [
    #     "logic_is_periodic",
    #     "logic_is_ringing",
    #     "dbg_label_reason",
    #     "std_all",
    #     "std_mid",
    #     "last_edge_time",          # keep last_edge_pos_ratio
    #     "ringing_energy",          # keep edge_count + late_activity
    #     # เลือก 1 ชุด:
    #     "ring_overshoot_z",        # ถ้าเก็บ overshoot_norm
    #     "ring_undershoot_z",       # ถ้าเก็บ undershoot_norm
    #     "fft_lowfreq_power_ratio",
    #     "edge_count", "late_activity", "mid_to_tail_ratio",
    #     "x_end",
    #     "std_tail_50",
    #     "crossing_rate",
    #     "end_med",
    #     "tail200_slope",
    #     "tail100_slope",
    #     "tail100_slope_norm",
    #     "meta_step_raw",
    #     "meta_abs_step",
    #     "meta_abs_step_to_noise",
    # ]

    df_zero = df_zero.drop(columns=ZERO_DROP_EXTRA, errors="ignore")

    # save feature cols after all drops
    zero_feature_cols = [c for c in df_zero.columns if c not in ["is_fast", "sample_weight"]]
    save_json(os.path.join(save_path, "zero_feature_cols.json"), zero_feature_cols)
    log_print(log_path, f"zero_clf feature cols = {len(zero_feature_cols)} (saved zero_feature_cols.json)")
    def suggest_drop_by_corr(df_zero, label_col="is_fast", weight_col="sample_weight",
                            corr_thr=0.995, topn=60):
        # เอาเฉพาะ feature numeric
        X = df_zero.drop(columns=[label_col, weight_col], errors="ignore")
        X = X.select_dtypes(include=[np.number]).copy()

        # corr matrix
        C = X.corr(numeric_only=True).abs()

        # ดึงคู่ที่ corr สูง (upper triangle)
        pairs = []
        cols = list(C.columns)
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                v = C.iat[i, j]
                if np.isfinite(v) and v >= corr_thr:
                    pairs.append((cols[i], cols[j], float(v)))

        pairs.sort(key=lambda x: x[2], reverse=True)

        print(f"\n[CORR] pairs with |corr| >= {corr_thr} : {len(pairs)}")
        for a, b, v in pairs[:topn]:
            print(f"  {a:28s}  <->  {b:28s}  corr={v:.6f}")

        # heuristic: drop ตัวที่ชื่อดู “raw/duplicate” มากกว่า
        drop = set()
        prefer_keep_keywords = [
            "fft_", "tail_ac_", "tail_zc_rate", "ring_peak_count"
        ]

        def score_keep(name: str) -> int:
            s = 0
            for k in prefer_keep_keywords:
                if k in name:
                    s += 3
            # อยากเก็บ ratio/norm มากกว่า time (scale invariant)
            if "pos_ratio" in name or "_norm" in name:
                s += 1
            if "time" in name:
                s -= 1
            if name in ["std_all", "std_mid", "ringing_energy"]:
                s -= 2
            return s

        for a, b, v in pairs:
            if a in drop or b in drop:
                continue
            # drop ตัวที่ score_keep ต่ำกว่า
            if score_keep(a) >= score_keep(b):
                drop.add(b)
            else:
                drop.add(a)

        drop = sorted(drop)
        print("\n[SUGGEST] drop candidates (corr-based heuristic):")
        for c in drop:
            print("  -", c)
        return drop

    # ---- call ----
    drop_suggest = suggest_drop_by_corr(df_zero, corr_thr=0.995)


    # IMPORTANT: ใช้ precision เพื่อลด FP-fast
    zero_clf = TabularPredictor(
        label="is_fast",
        path=zero_model_path,
        problem_type="binary",
        eval_metric="precision",
        verbosity=2,
        sample_weight="sample_weight", 
    ).fit(
        train_data=df_zero,
        presets=presets,
        time_limit=t_zero,
        num_gpus=gpu_count,
        dynamic_stacking=False,
    )

    # leaderboard / summary
    try:
        lb0 = zero_clf.leaderboard(df_zero, silent=True)
        lb0_path = os.path.join(save_path, f"leaderboard_zero_{ts}.csv")
        lb0.to_csv(lb0_path, index=False)
        log_print(log_path, f"saved: {lb0_path}")
        log_print(log_path, "zero_clf top models:\n" + lb0[["model", "score_val"]].head(5).to_string(index=False))
    except Exception as e:
        log_print(log_path, f"leaderboard zero_clf failed: {e}")

    try:
        fs0 = zero_clf.fit_summary(verbosity=0)
        fs0_path = os.path.join(save_path, f"fit_summary_zero_{ts}.json")
        save_json(fs0_path, fs0)
        log_print(log_path, f"saved: {fs0_path}")
    except Exception as e:
        log_print(log_path, f"fit_summary zero_clf failed: {e}")

    # =========================
    # Stage B: wait regressor (slow only)
    # =========================
    log_print(log_path, "\n=== Stage B: wait_reg (regression on slow only) ===")

    df_slow = df_train[df_train[label].astype(float) > FAST_MS].copy()
    log_print(log_path, f"slow rows for regressor = {len(df_slow)}")
    if len(df_slow) < 5:
        raise ValueError("slow samples too few to train regressor.")

    df_slow["wait_time_log"] = np.log1p(df_slow[label].astype(float))

    df_reg = df_slow.drop(columns=cols_to_drop_found, errors="ignore").copy()
    df_reg_for_fit = df_reg.drop(columns=["wave_id", label, "is_fast"], errors="ignore")

    reg_feature_cols = [c for c in df_reg_for_fit.columns if c != "wait_time_log"]
    save_json(os.path.join(save_path, "reg_feature_cols.json"), reg_feature_cols)
    log_print(log_path, f"wait_reg feature cols = {len(reg_feature_cols)} (saved reg_feature_cols.json)")

    wait_reg = TabularPredictor(
        label="wait_time_log",
        path=reg_model_path,
        problem_type="regression",
        eval_metric="mean_absolute_error",
        verbosity=2,
    ).fit(
        train_data=df_reg_for_fit,
        presets=presets,
        time_limit=t_reg,
        num_gpus=gpu_count,
        dynamic_stacking=False,
    )

    try:
        lbr = wait_reg.leaderboard(df_reg_for_fit, silent=True)
        lbr_path = os.path.join(save_path, f"leaderboard_reg_{ts}.csv")
        lbr.to_csv(lbr_path, index=False)
        log_print(log_path, f"saved: {lbr_path}")
        keep = [c for c in ["model", "score_val", "pred_time_val", "fit_time"] if c in lbr.columns]
        log_print(log_path, "wait_reg top models:\n" + lbr[keep].head(5).to_string(index=False))
    except Exception as e:
        log_print(log_path, f"leaderboard wait_reg failed: {e}")

    try:
        fsr = wait_reg.fit_summary(verbosity=0)
        fsr_path = os.path.join(save_path, f"fit_summary_reg_{ts}.json")
        save_json(fsr_path, fsr)
        log_print(log_path, f"saved: {fsr_path}")
    except Exception as e:
        log_print(log_path, f"fit_summary wait_reg failed: {e}")

    # =========================
    # Threshold selection (FP-safe)
    # =========================
    log_print(log_path, "\n=== Threshold selection & diagnostics (FP-safe) ===")

    X_all = df_train.drop(columns=[label], errors="ignore").copy()
    y_true = df_train[label].to_numpy(dtype=float)

    X_feat = X_all.drop(columns=cols_to_drop_found, errors="ignore").copy()

    # proba fast
    X_zero_in = X_feat.drop(columns=["wave_id"], errors="ignore").copy()
    X_zero_in = align_columns(X_zero_in, zero_feature_cols)
    proba_is_fast = proba_class1(zero_clf.predict_proba(X_zero_in))

    # reg pred for all (for threshold scan)
    X_reg_in = X_feat.drop(columns=["wave_id", "is_fast", label, "wait_time_log"], errors="ignore").copy()

    X_reg_in = align_columns(X_reg_in, reg_feature_cols)
    wait_log_pred = wait_reg.predict(X_reg_in)
    wait_pred = np.expm1(np.asarray(wait_log_pred, dtype=float))
    wait_pred = np.clip(wait_pred, 0, None)
    

    # 1) old strict (ถ้ายังอยากเก็บไว้)
    # best_thr, best_mae, thr_reason = pick_threshold_fp_safe(
    #     scan,
    #     min_precision=MIN_FAST_PRECISION,
    #     max_fp_rate=MAX_FAST_FP_RATE,
    # )

    scan = threshold_scan_report(proba_is_fast, wait_pred, y_true, thr_min=0.10, thr_max=0.90, steps=161)
    scan_path = os.path.join(save_path, f"threshold_scan_{ts}.csv")
    scan.to_csv(scan_path, index=False)
    log_print(log_path, f"saved: {scan_path}")

    # --- pick dual (รอบแรก) เพื่อให้มี thr_high/thr_low ---
    thr_high, thr_low, dual_reason = pick_dual_thresholds(
        scan,
        min_precision_high=MIN_FAST_PRECISION,
        max_fp_rate_high=MAX_FAST_FP_RATE,
        min_recall_low=0.995,
        max_fp_rate_low=0.0020,
    )

    # --- refine scan ใกล้ thr_high (ตอนนี้ thr_high มีค่าแล้ว) ---
    ref_min = max(0.0, thr_high - 0.03)
    ref_max = min(1.0, thr_high + 0.03)

    scan_ref = threshold_scan_report(
        proba_is_fast, wait_pred, y_true,
        thr_min=ref_min, thr_max=ref_max,
        steps=601,  # step ~0.0001
    )
    scan_ref_path = os.path.join(save_path, f"threshold_scan_ref_{ts}.csv")
    scan_ref.to_csv(scan_ref_path, index=False)
    log_print(log_path, f"saved: {scan_ref_path}")

    # 2) NEW: dual thresholds
    thr_high, thr_low, dual_reason = pick_dual_thresholds(
        scan_ref,
        min_precision_high=MIN_FAST_PRECISION,   # หรือ 1.0
        max_fp_rate_high=MAX_FAST_FP_RATE,       # เช่น 0.0002
        # min_recall_low=0.998,                    # ปรับได้
        # max_fp_rate_low=0.0010,                  # ปรับได้
    )
    if thr_low >= thr_high:
        thr_low = thr_high


    # 3) save thresholds for predict
    # thr_json = {
    #     "thr_high": float(thr_high),
    #     "thr_low": float(thr_low),
    #     "fast_ms": float(FAST_MS),
    #     "policy": "dual_threshold",
    #     "reason": dual_reason,
    # }
    save_json(
        os.path.join(save_path, "fast_thresholds.json"),
        {
            "thr_high": float(thr_high),
            "thr_low": float(thr_low),
            "reason": str(dual_reason),
            "ts": ts,
        },
    )

    log_print(log_path, f"dual_thresholds: high={thr_high:.3f} low={thr_low:.3f} | {dual_reason}")


    print("\n=== Feature Importance: zero_clf ===")

        # X สำหรับ zero_clf ต้องเป็น input ที่ align แล้ว (เหมือนตอน predict_proba)
    try:
        # 1) สร้าง X ให้ "เหมือนตอน fit zero_clf" ที่สุด
        # ตอน fit: df_zero = df_train(drop meta, drop label+wave_id) แล้วมี sample_weight แล้ว drop ZERO_DROP_EXTRA
        X_zero_in = df_train.drop(columns=cols_to_drop_found, errors="ignore").copy()
        X_zero_in = X_zero_in.drop(columns=[label, "wave_id"], errors="ignore")

        # NOTE: ตอน predict_proba ห้ามมี label is_fast / sample_weight ปน
        X_zero_in = X_zero_in.drop(columns=["is_fast", "sample_weight"], errors="ignore")

        # drop same extra columns as training
        X_zero_in = X_zero_in.drop(columns=ZERO_DROP_EXTRA, errors="ignore")

        # 2) align ให้ตรงกับ feature cols ที่ save ไว้
        X_zero_in = align_columns(X_zero_in, zero_feature_cols)

        # 3) y_true ที่ใช้วัด FP-safe ต้องเป็น "wait_time_ms จริง" (ไม่ใช่ y_fast)
        y_true_ms_all = df_train[label].to_numpy(dtype=float)

        # 4) ตัดคอลัมน์ที่ constant / all-NaN ออก (กัน warning + ทำให้ perm มีความหมาย)
        feat_cols = list(X_zero_in.columns)
        X_tmp = X_zero_in[feat_cols].copy()
        X_tmp = X_tmp.dropna(axis=1, how="all")
        nunique = X_tmp.nunique(dropna=True)
        X_tmp = X_tmp.loc[:, nunique > 1]
        X_zero_in = X_tmp
        feat_cols = list(X_zero_in.columns)

        # 5) FP-safe permutation importance (วัด fp_rate/precision/recall @ best_thr)
        imp0 = perm_importance_fp_safe(
            zero_clf,
            X_zero_in,
            y_true_ms_all,
            thr=thr_high,
            fast_ms=FAST_MS,
            features=feat_cols,
            n_shuffle=5,
            topk=25,
            random_state=42,
        )

        # (optional) save
        imp0_path = os.path.join(save_path, f"perm_importance_fp_safe_zero_{ts}.csv")
        imp0.to_csv(imp0_path, index=False)
        print(f"saved: {imp0_path}")

    except Exception as e:
        print(f"[WARN] zero_clf perm importance failed: {e}")

    print("\n=== Correlation with proba_is_fast (top +/-) ===")

    # try:
    #     df_imp0 = X_zero_in.copy()

    try:
        df_imp0 = X_zero_in.copy()
        proba = zero_clf.predict_proba(df_imp0)

        if hasattr(proba, "columns"):
            proba_fast = proba[1].to_numpy()
        else:
            proba_fast = proba[:, 1]

        df_corr = df_imp0.copy()
        df_corr["proba_fast"] = proba_fast

        drop_cols = [
            "proba_fast",
            "is_fast",
            "wait_time_ms",
            "wait_time_log",
            "wave_id",
            "sample_weight",              # ✅ เพิ่ม
        ]
        drop_cols = [c for c in drop_cols if c in df_corr.columns]
        feat_only = df_corr.drop(columns=drop_cols, errors="ignore")

        # ✅ drop คอลัมน์ที่เป็น NaN ทั้งคอลัมน์
        feat_only = feat_only.dropna(axis=1, how="all")

        # ✅ drop คอลัมน์ที่เป็น constant (std=0 ทำให้ corr warn)
        nunique = feat_only.nunique(dropna=True)
        feat_only = feat_only.loc[:, nunique > 1]

        corr = feat_only.corrwith(df_corr["proba_fast"]).sort_values(ascending=False)

        print("\nTop positive (push FAST):")
        print(corr.head(15).to_string())

        print("\nTop negative (push SLOW):")
        print(corr.tail(15).to_string())

    except Exception as e:
        print(f"[WARN] correlation debug failed: {e}")
    
    # =========================
    # Feature importance (FP-safe) for zero_clf
    # =========================
    log_print(log_path, "\n=== Permutation Importance (FP-safe): zero_clf ===")

    # X สำหรับ zero_clf (เหมือนตอน predict_proba)
    X_zero_imp = X_feat.drop(columns=["wave_id"], errors="ignore").copy()
    X_zero_imp = align_columns(X_zero_imp, zero_feature_cols)

    y_true_ms_all = y_true  # wait_time_ms true label ทั้งชุด

    # อย่าเอา sample_weight / label เข้า importance
    drop_imp = ["sample_weight", "is_fast"]
    imp_features = [c for c in X_zero_imp.columns if c not in drop_imp]

    # (กันคอลัมน์ NaN/constant ให้ corr/perm ไม่เพี้ยน)
    tmp = X_zero_imp[imp_features].copy()
    tmp = tmp.dropna(axis=1, how="all")
    nuniq = tmp.nunique(dropna=True)
    tmp = tmp.loc[:, nuniq > 1]
    imp_features = list(tmp.columns)
    X_zero_imp = X_zero_imp[imp_features]

    imp0 = perm_importance_fp_safe(
        zero_clf,
        X_zero_imp,
        y_true_ms_all,
        thr=thr_high,
        fast_ms=FAST_MS,
        features=imp_features,
        n_shuffle=5,
        topk=25,
        random_state=42,
    )

    # save
    imp0_path = os.path.join(save_path, f"perm_importance_fp_safe_zero_{ts}.csv")
    imp0.to_csv(imp0_path, index=False)
    log_print(log_path, f"saved: {imp0_path}")

    # save_text(os.path.join(save_path, "zero_threshold.txt"), f"{best_thr:.6f}")
    # log_print(log_path, f"threshold_pick_reason = {thr_reason}")
    # log_print(log_path, f"picked_thr = {best_thr:.3f} (scan mae={best_mae:.6f})")

    # print some rows
    top_mae = scan.sort_values("mae_all", ascending=True).head(10)
    log_print(log_path, "\nTop 10 thresholds by MAE:\n" + top_mae[["thr","mae_all","fast_precision","fast_fp_rate","fast_recall","pred_fast"]].to_string(index=False))

    top_prec = scan.sort_values(["fast_precision","fast_fp_rate","mae_all"], ascending=[False, True, True]).head(10)
    log_print(log_path, "\nTop 10 thresholds by PRECISION (then FP-rate, then MAE):\n" + top_prec[["thr","fast_precision","fast_fp_rate","fast_recall","mae_all","pred_fast"]].to_string(index=False))

    # final train metrics with selected threshold
    SANITY_MAX_REG_MS = 0.30  # ปรับได้

    is_fast_strict = (proba_is_fast >= thr_high)
    is_fast_soft = (proba_is_fast >= thr_low)
    sanity_ok = (wait_pred <= SANITY_MAX_REG_MS)

    is_fast_pred = is_fast_strict | (is_fast_soft & sanity_ok)
    y_hat = np.where(is_fast_pred, FAST_MS, wait_pred)

    # is_fast_pred = proba_is_fast >= best_thr
    # y_hat = np.where(is_fast_pred, FAST_MS, wait_pred)

    mae_all = float(np.mean(np.abs(y_hat - y_true)))

    slow_mask = y_true > FAST_MS + 1e-12
    mae_slow = float(np.mean(np.abs(y_hat[slow_mask] - y_true[slow_mask]))) if np.any(slow_mask) else float("nan")

    fast_true = (y_true <= FAST_MS + 1e-12)
    tp = int(np.sum(is_fast_pred & fast_true))
    fn = int(np.sum((~is_fast_pred) & fast_true))
    fp = int(np.sum(is_fast_pred & (~fast_true)))
    tn = int(np.sum((~is_fast_pred) & (~fast_true)))

    fast_precision = tp / max(tp + fp, 1)
    fast_recall = tp / max(int(np.sum(fast_true)), 1)
    fast_fp_rate = fp / max(int(np.sum(~fast_true)), 1)

    # ===== call (after you compute tp, fp, fn, tn) =====
    cm_path = os.path.join(save_path, f"confusion_heatmap_{ts}.png")
    plot_confusion_heatmap(tp, fp, fn, tn, title=f"FAST Confusion @ thr_high={thr_high:.3f}, thr_low={thr_low:.3f}", save_path=cm_path)
    log_print(log_path, f"[saved] {cm_path}")

    log_print(log_path, f"\nTrain MAE(all)={mae_all:.6f} | MAE(slow-only)={mae_slow:.6f}")
    log_print(log_path, f"FAST precision={fast_precision:.6f} recall={fast_recall:.6f} fp_rate={fast_fp_rate:.6f}")
    log_print(log_path, f"proba_is_fast min/mean/max = {proba_is_fast.min():.6f} {proba_is_fast.mean():.6f} {proba_is_fast.max():.6f}")
    log_print(log_path, f"predicted fast = {int(is_fast_pred.sum())} / {len(is_fast_pred)}")

    # diagnostics per row
    diag = df_train[["wave_id", label, "is_fast"]].copy()
    diag["proba_is_fast"] = proba_is_fast
    diag["reg_wait_pred_ms"] = wait_pred
    diag["pred_is_fast"] = is_fast_pred.astype(int)
    diag["pred_wait_ms"] = y_hat
    diag["abs_error"] = np.abs(diag["pred_wait_ms"].to_numpy(dtype=float) - diag[label].to_numpy(dtype=float))

    diag_path = os.path.join(save_path, f"train_diagnostics_{ts}.csv")
    diag.sort_values("abs_error", ascending=False).to_csv(diag_path, index=False)
    log_print(log_path, f"saved: {diag_path}")
    log_print(log_path, "\nWorst 15 rows (by abs_error):\n" + diag.sort_values("abs_error", ascending=False).head(15).to_string(index=False))

    log_print(log_path, "\n✅ Saved models at: " + save_path)
    log_print(log_path, f" - zero_clf: {zero_model_path}")
    log_print(log_path, f" - wait_reg: {reg_model_path}")
    #log_print(log_path, f" - fast_threshold: {best_thr:.3f}")
    log_print(log_path, f" - log file: {log_path}")
    log_print(log_path, "=== TRAIN LOG END ===")


def main():
    ap = argparse.ArgumentParser("Train AutoGluon v22 (2-stage: fast-clf + slow-reg) + FP-safe threshold")
    ap.add_argument("--data", required=True, help="train features csv (must include wait_time_ms)")
    ap.add_argument("--label", default="wait_time_ms")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--time-limit", type=int, default=300)
    ap.add_argument("--presets", default="medium_quality")  # medium_quality / high_quality / best_quality

    args = ap.parse_args()
    train(
        data_path=args.data,
        label=args.label,
        model_dir=args.model_dir,
        presets=args.presets,
        time_limit=args.time_limit,
    )

if __name__ == "__main__":
    main()
