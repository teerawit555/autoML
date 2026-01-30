# scripts/train_ag_v27_split.py
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt

"""
v27 (with holdout calibration split) goals
- ลด manual tuning: threshold + sanity gate derive จาก "CALIB" holdout (ไม่ใช้ train ทั้งก้อน)
- แยก "dual thresholds" + "sanity gate" เป็น artifact (json) ให้ predict ใช้ต่อ
- เน้น FP-safe: fast FP อันตรายสุด (8ms -> 0.1ms เละ)
- NOTE: การ clip/normalize feature (เช่น overshoot_norm ฯลฯ) ต้องทำใน feature extractor (make_features.py) ไม่ใช่ที่ train script
"""

# =========================
# Config (minimal & stable)
# =========================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_SAVE_PATH = f"AutogluonModels/ag-v27-{ts}"

FAST_MS = 0.1

# FP-safe threshold policy
MIN_FAST_PRECISION = 0.999
MAX_FAST_FP_RATE   = 0.0002
MIN_FAST_RECALL    = 0.98

# Holdout calibration split
CALIB_FRAC = 0.20
SPLIT_SEED = 42

# columns to drop if present
DROP_ALWAYS = ["type"]  # identity leak
COLS_TO_DROP = ["force_mA", "range_V", "temp_C"]  # optional meta

# sanity-gate feature names (optional; if missing -> auto-skip)
SANITY_FEATURES = {
    "meta_step_to_span": "meta_step_to_span",
    "per_tail_ac_best": "per_tail_ac_best",
}

# =========================
# IO helpers
# =========================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

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

# def stratified_holdout_split(
#     df: pd.DataFrame,
#     label_col: str,
#     frac: float,
#     seed: int
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Stratified split by label_col (binary 0/1).
#     Returns: (df_train, df_calib)
#     """
#     rng = np.random.default_rng(seed)

#     df0 = df[df[label_col] == 0].copy()
#     df1 = df[df[label_col] == 1].copy()

#     idx0 = df0.index.to_numpy()
#     idx1 = df1.index.to_numpy()

#     rng.shuffle(idx0)
#     rng.shuffle(idx1)

#     n0_cal = int(round(len(idx0) * frac))
#     n1_cal = int(round(len(idx1) * frac))

#     cal_idx = np.concatenate([idx0[:n0_cal], idx1[:n1_cal]])
#     trn_idx = np.concatenate([idx0[n0_cal:], idx1[n1_cal:]])

#     df_cal = df.loc[cal_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)
#     df_tr  = df.loc[trn_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)
#     return df_tr, df_cal

import numpy as np
import pandas as pd

def stratified_holdout_split(
    df: pd.DataFrame,
    label_col: str,
    frac: float,
    seed: int,
    *,
    type_col: str = "type",
    min_per_stratum: int = 1,
    max_frac_per_stratum: float = 0.50,  # กัน strata เล็กโดนดูดไป calib เยอะเกิน
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split by (label_col, type_col).
    - label_col: binary 0/1 (e.g. is_fast)
    - type_col : waveform type column (e.g. type/type_debug)
    Returns: (df_train, df_calib)

    Notes:
    - Ensures each (type, label) stratum contributes to CALIB when possible.
    - Small strata won't be emptied into CALIB (cap by max_frac_per_stratum).
    """
    if label_col not in df.columns:
        raise KeyError(f"Missing label_col: {label_col}")
    if type_col not in df.columns:
        # fallback to original behavior (label-only)
        rng = np.random.default_rng(seed)
        df0 = df[df[label_col] == 0].copy()
        df1 = df[df[label_col] == 1].copy()

        idx0 = df0.index.to_numpy()
        idx1 = df1.index.to_numpy()
        rng.shuffle(idx0); rng.shuffle(idx1)

        n0_cal = int(round(len(idx0) * frac))
        n1_cal = int(round(len(idx1) * frac))

        cal_idx = np.concatenate([idx0[:n0_cal], idx1[:n1_cal]])
        trn_idx = np.concatenate([idx0[n0_cal:], idx1[n1_cal:]])

        df_cal = df.loc[cal_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)
        df_tr  = df.loc[trn_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return df_tr, df_cal

    rng = np.random.default_rng(seed)

    # ทำ strata key = (type, label)
    d = df.copy()
    d["_stratum"] = d[type_col].astype(str) + "__" + d[label_col].astype(int).astype(str)

    cal_indices = []
    trn_indices = []

    for _, g in d.groupby("_stratum"):
        idx = g.index.to_numpy()
        rng.shuffle(idx)

        n = len(idx)
        if n <= 1:
            # มีแค่ 1 ตัว -> ส่งไป train เพื่อไม่ให้ calib บิด
            trn_indices.append(idx)
            continue

        # base target
        n_cal = int(round(n * frac))

        # min per stratum (แต่ต้องเหลือ train อย่างน้อย 1)
        n_cal = max(n_cal, min_per_stratum)
        n_cal = min(n_cal, n - 1)

        # cap ไม่ให้ดูดเกินสัดส่วน
        cap = int(np.floor(n * max_frac_per_stratum))
        cap = max(cap, 1)
        n_cal = min(n_cal, cap)

        cal_indices.append(idx[:n_cal])
        trn_indices.append(idx[n_cal:])

    cal_idx = np.concatenate(cal_indices) if len(cal_indices) else np.array([], dtype=int)
    trn_idx = np.concatenate(trn_indices) if len(trn_indices) else np.array([], dtype=int)

    df_cal = df.loc[cal_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_tr  = df.loc[trn_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df_tr, df_cal


def fast_metrics_at_thr(
    proba_is_fast: np.ndarray,
    y_true_ms: np.ndarray,
    thr: float,
    fast_ms: float = FAST_MS,
) -> dict:
    y_true_ms = np.asarray(y_true_ms, float)
    proba_is_fast = np.asarray(proba_is_fast, float)

    zt = (y_true_ms <= fast_ms + 1e-12)  # true fast
    zp = (proba_is_fast >= thr)          # predicted fast

    tp = int(np.sum(zp & zt))
    fp = int(np.sum(zp & (~zt)))
    fn = int(np.sum((~zp) & zt))
    tn = int(np.sum((~zp) & (~zt)))

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)

    slow_total = int(np.sum(~zt))
    fp_rate = fp / max(slow_total, 1)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "fp_rate": float(fp_rate),
        "pred_fast": int(np.sum(zp)),
    }

def plot_confusion_heatmap(tp, fp, fn, tn, title: str, save_path: str | None = None, normalize: str = "row"):
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
    proba_is_fast = np.asarray(proba_is_fast, float)
    wait_pred = np.asarray(wait_pred, float)

    is_fast_true = (y_true <= FAST_MS + 1e-12)
    rows = []
    thr_grid = np.linspace(thr_min, thr_max, steps)

    for thr in thr_grid:
        is_fast_pred = proba_is_fast >= thr
        y_hat = np.where(is_fast_pred, FAST_MS, wait_pred) # ค่าทำนายของระบบ
        mae_all = float(np.mean(np.abs(y_hat - y_true)))

        zp = is_fast_pred
        zt = is_fast_true

        tp = int(np.sum(zp & zt))
        fn = int(np.sum((~zp) & zt))
        fp = int(np.sum(zp & (~zt)))
        tn = int(np.sum((~zp) & (~zt)))

        fast_recall = tp / max(int(np.sum(zt)), 1)
        fast_precision = tp / max(tp + fp, 1)
        slow_total = int(np.sum(~zt))
        fast_fp_rate = fp / max(slow_total, 1)

        rows.append({
            "thr": float(thr),
            "mae_all": mae_all,
            "fast_recall": float(fast_recall),
            "fast_precision": float(fast_precision),
            "fast_fp_rate": float(fast_fp_rate),
            "TP_fast": tp, "FN_fast": fn, "FP_fast": fp, "TN_slow": tn,
            "pred_fast": int(np.sum(zp)),
        })

    return pd.DataFrame(rows)

def pick_dual_thresholds(
    scan: pd.DataFrame,
    *,
    # strict (T_high): FP-safe
    min_precision_high: float = MIN_FAST_PRECISION,
    max_fp_rate_high: float = MAX_FAST_FP_RATE,
    # soft (T_low): recall-oriented แต่ยังคุม FP
    min_recall_low: float = 0.995,
    max_fp_rate_low: float = 0.0020,
) -> tuple[float, float, str]:
    s = scan.copy()
    required = ["thr", "mae_all", "fast_precision", "fast_fp_rate", "fast_recall"]
    missing = [c for c in required if c not in s.columns]
    if missing:
        raise ValueError(f"scan missing columns: {missing}")

    for c in required:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.dropna(subset=required)

    # --- high (strict) ---
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

    # --- low (soft zone) ---
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
        thr_low = thr_high
        reason_low = "low=none_under_high"

    thr_low = float(min(thr_low, thr_high))
    reason = f"{reason_high} | {reason_low}"
    return thr_high, thr_low, reason

# =========================
# Sanity gate (AUTO from train data)
# =========================
EPS = 1e-12

def _fallback_from_global(
    df: pd.DataFrame,
    col: str,
    q: float,
    hard_min: float,
    hard_max: float,
    default: float,
) -> float:
    if col not in df.columns:
        return default
    v = df[col].to_numpy(float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return default
    x = np.quantile(v, q)
    return float(np.clip(x, hard_min, hard_max))

def _safe_quantile(a: np.ndarray, q: float, default: float) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float(default)
    return float(np.quantile(a, q))

def derive_sanity_gate_from_train(
    df_train: pd.DataFrame,
    y_true_ms: np.ndarray,
    *,
    proba_is_fast: np.ndarray,
    wait_pred: np.ndarray,
    thr_high: float,
    thr_low: float,
    q_feat: float = 0.98,
    q_reg_fp: float = 0.95,   # เข้มขึ้น (กันเคส reg หลุดใหญ่)
) -> dict:
    # global true-fast
    global_fast = (np.asarray(y_true_ms, float) <= FAST_MS + 1e-12)

    DEFAULTS = {
        "step_max": _fallback_from_global(
            df_train, SANITY_FEATURES["meta_step_to_span"],
            q=0.995, hard_min=0.05, hard_max=1.5, default=0.35
        ),
        "ac_max": _fallback_from_global(
            df_train, SANITY_FEATURES["per_tail_ac_best"],
            q=0.995, hard_min=0.05, hard_max=0.99, default=0.55
        ),
        "tail_std_max": _fallback_from_global(
            df_train, "tail_std",
            q=0.995, hard_min=0.01, hard_max=1.0, default=0.20
        ),
        "late_activity_max": _fallback_from_global(
            df_train, "late_activity",
            q=0.995, hard_min=0.01, hard_max=1.0, default=0.15
        ),
        "reg_fp_ms": _fallback_from_global(
            pd.DataFrame({"reg": np.asarray(wait_pred, float)[global_fast]}),
            "reg",
            q=0.99, hard_min=0.12, hard_max=0.8, default=0.5
        ),
    }

    y_true_ms = np.asarray(y_true_ms, float)
    p = np.asarray(proba_is_fast, float)
    wait_pred = np.asarray(wait_pred, float)

    is_fast_true = (y_true_ms <= FAST_MS + 1e-12)
    near_soft = (p >= float(thr_low)) & (p < float(thr_high))
    mask = is_fast_true & near_soft
    if not np.any(mask):
        mask = is_fast_true

    out: dict[str, Any] = {
        "enabled": True,
        "features_present": {},
        "policy": "soft_zone_fp_firewall",
        "thr_high": float(thr_high),
        "thr_low": float(thr_low),
        "notes": f"Derived from CALIB true-fast & near(thr_low={thr_low:.6f}); q_feat={q_feat}, q_reg_fp={q_reg_fp}. Used only for SOFT zone.",
    }

    # --- step_to_span ---
    f_step = SANITY_FEATURES["meta_step_to_span"]
    if f_step in df_train.columns:
        out["features_present"][f_step] = True
        v = df_train.loc[mask, f_step].to_numpy(float)
        step_max = _safe_quantile(v, q_feat, DEFAULTS["step_max"])
        out["step_max"] = float(np.clip(step_max, 0.05, 1.50))
    else:
        out["features_present"][f_step] = False
        out["step_max"] = DEFAULTS["step_max"]

    # --- per_tail_ac_best ---
    f_ac = SANITY_FEATURES["per_tail_ac_best"]
    if f_ac in df_train.columns:
        out["features_present"][f_ac] = True
        v = df_train.loc[mask, f_ac].to_numpy(float)
        ac_max = _safe_quantile(v, q_feat, DEFAULTS["ac_max"])
        out["ac_max"] = float(np.clip(ac_max, 0.05, 0.99))
    else:
        out["features_present"][f_ac] = False
        out["ac_max"] = DEFAULTS["ac_max"]

    # tail_std
    f_tail_std = "tail_std"
    if f_tail_std in df_train.columns:
        out["features_present"][f_tail_std] = True
        v = df_train.loc[mask, f_tail_std].to_numpy(float)
        out["tail_std_max"] = float(_safe_quantile(v, q_feat, DEFAULTS["tail_std_max"]))
    else:
        out["features_present"][f_tail_std] = False
        out["tail_std_max"] = DEFAULTS["tail_std_max"]

    # late_activity
    f_late = "late_activity"
    if f_late in df_train.columns:
        out["features_present"][f_late] = True
        v = df_train.loc[mask, f_late].to_numpy(float)
        out["late_activity_max"] = float(_safe_quantile(v, q_feat, DEFAULTS["late_activity_max"]))
    else:
        out["features_present"][f_late] = False
        out["late_activity_max"] = DEFAULTS["late_activity_max"]

    # reg veto (SOFT zone only) — derive จาก reg_pred ของ true-fast ใกล้ขอบ
    reg_vals = wait_pred[mask]
    reg_fp = _safe_quantile(reg_vals, q_reg_fp, DEFAULTS["reg_fp_ms"])
    out["reg_fp_ms"] = float(np.clip(reg_fp, 0.12, 0.60))
    out["strict_margin"] = 0.02

    return out

def apply_dual_threshold_with_sanity(
    *,
    proba_is_fast: np.ndarray,
    wait_pred: np.ndarray,
    df_feats: pd.DataFrame,
    thr_high: float,
    thr_low: float,
    sanity: dict,
) -> np.ndarray:
    p = np.asarray(proba_is_fast, float)
    wait_pred = np.asarray(wait_pred, float)

    strict_margin = float(sanity.get("strict_margin", 0.02))  # ปรับได้
    is_fast_strict_core = (p >= thr_high + strict_margin)
    is_fast_strict_edge = (p >= thr_high) & (p < thr_high + strict_margin)
    is_fast_soft = (p >= thr_low) & (p < thr_high)

    # sanity apply เฉพาะ edge+soft (core ผ่านเลย)
    need_sanity = is_fast_strict_edge | is_fast_soft
    sanity_ok = np.ones_like(p, dtype=bool)

    # feature gates
    step_max = float(sanity.get("step_max", 0.35))
    f_step = SANITY_FEATURES.get("meta_step_to_span", "meta_step_to_span")
    if f_step in df_feats.columns:
        sanity_ok &= (df_feats[f_step].to_numpy(float) <= step_max)

    f_ac = SANITY_FEATURES.get("per_tail_ac_best", "per_tail_ac_best")
    ac_max = float(sanity.get("ac_max", 0.55))
    if f_ac in df_feats.columns:
        sanity_ok &= (df_feats[f_ac].to_numpy(float) <= ac_max)

    tail_std_max = float(sanity.get("tail_std_max", np.inf))
    if "tail_std" in df_feats.columns and np.isfinite(tail_std_max):
        sanity_ok &= (df_feats["tail_std"].to_numpy(float) <= tail_std_max)

    late_activity_max = float(sanity.get("late_activity_max", np.inf))
    if "late_activity" in df_feats.columns and np.isfinite(late_activity_max):
        sanity_ok &= (df_feats["late_activity"].to_numpy(float) <= late_activity_max)

    # reg firewall (สำคัญ)
    reg_fp_ms = float(sanity.get("reg_fp_ms", np.inf))
    if np.isfinite(reg_fp_ms):
        sanity_ok &= (wait_pred <= reg_fp_ms)

    return is_fast_strict_core | (need_sanity & sanity_ok)


def fp_safe_feature_importance(
    model,
    X: pd.DataFrame,
    y_true_ms: np.ndarray,
    *,
    thr_high: float,
    fast_ms: float = 0.1,
    feat_cols: list[str] | None = None,
    n_repeat: int = 5,
    subsample: int | None = 3000,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    FP-safe permutation importance for zero_clf:
    - shuffle ทีละ feature
    - วัดผลกระทบต่อ fp_rate / precision / recall @ thr_high
    """
    X0 = X.copy()

    if feat_cols is None:
        feat_cols = list(X0.columns)

    X0 = X0[feat_cols].copy()
    X0 = X0.dropna(axis=1, how="all")
    nunique = X0.nunique(dropna=True)
    X0 = X0.loc[:, nunique > 1]
    feat_cols = list(X0.columns)

    if subsample is not None and len(X0) > subsample:
        idx = X0.sample(n=subsample, random_state=random_state).index
        X0 = X0.loc[idx].copy()
        y_true_ms = np.asarray(y_true_ms, float)[idx.to_numpy()]
        X0 = X0.reset_index(drop=True)
        y_true_ms = np.asarray(y_true_ms, float)
        assert len(X0) == len(y_true_ms), "X0 and y_true_ms length mismatch"

    y_true_ms = np.asarray(y_true_ms, float)
    fast_true = (y_true_ms <= fast_ms + 1e-12)
    slow_total = int((~fast_true).sum())

    base_proba = proba_class1(model.predict_proba(X0))
    base_pred_fast = (base_proba >= thr_high)

    tp0 = int((base_pred_fast & fast_true).sum())
    fp0 = int((base_pred_fast & ~fast_true).sum())
    fn0 = int((~base_pred_fast & fast_true).sum())

    base_fp_rate = fp0 / max(slow_total, 1)
    base_prec = tp0 / max(tp0 + fp0, 1)
    base_rec  = tp0 / max(tp0 + fn0, 1)

    rng = np.random.default_rng(random_state)

    rows = []
    for f in feat_cols:
        fp_rates, precs, recs = [], [], []

        col = X0[f].to_numpy()
        for _ in range(n_repeat):
            Xp = X0.copy()
            Xp[f] = rng.permutation(col)

            p = proba_class1(model.predict_proba(Xp))
            pred_fast = (p >= thr_high)

            tp = int((pred_fast & fast_true).sum())
            fp = int((pred_fast & ~fast_true).sum())
            fn = int((~pred_fast & fast_true).sum())

            fp_rates.append(fp / max(slow_total, 1))
            precs.append(tp / max(tp + fp, 1))
            recs.append(tp / max(tp + fn, 1))

        rows.append({
            "feature": f,
            "fp_rate_increase": float(np.mean(fp_rates) - base_fp_rate),
            "precision_drop": float(base_prec - np.mean(precs)),
            "recall_drop": float(base_rec - np.mean(recs)),
            "fp_rate_base": float(base_fp_rate),
            "precision_base": float(base_prec),
            "recall_base": float(base_rec),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["fp_rate_increase", "precision_drop"], ascending=[False, False]).reset_index(drop=True)
    return out

# def derive_glitch_phase_thresholds(df: pd.DataFrame) -> dict:
#     # ใช้เฉพาะแถวที่มี edge จริง (กัน 0.0 จาก no-edge)
#     d = df.copy()
#     d = d[np.isfinite(d["edge_count"])]

#     EDGES_MIN = 3
#     d_edge = d[d["edge_count"] >= 1]          # สำหรับ first/late
#     d_persist = d[d["edge_count"] >= EDGES_MIN]  # สำหรับ span/tail_rate ที่จริงจังขึ้น

#     # fallback กันเคสข้อมูลน้อย
#     if len(d_edge) < 50:
#         d_edge = d[d["edge_count"] >= 1]
#     if len(d_persist) < 50:
#         d_persist = d[d["edge_count"] >= 1]

#     early = float(np.quantile(d_edge["first_edge_pos_ratio"], 0.10))
#     late  = float(np.quantile(d_edge["first_edge_pos_ratio"], 0.90))
#     persist = float(np.quantile(d_persist["edge_span_ratio"], 0.70))
#     tail_hi = float(np.quantile(d_persist["tail_edge_rate"], 0.70))

#     # clamp กันค่าหลุดโลก
#     early = float(np.clip(early, 0.02, 0.30))     # ไม่ให้ 0.001 / ไม่ให้ใหญ่เกิน
#     late  = float(np.clip(late,  0.40, 0.95))
#     persist = float(np.clip(persist, 0.15, 0.95))
#     tail_hi = float(max(tail_hi, 0.01))           # กัน 0.0

#     out = {
#         "EARLY_MAX": early,
#         "LATE_MIN": late,
#         "PERSIST_MIN": persist,
#         "TAIL_RATE_MIN": tail_hi,
#         "EDGES_MIN": EDGES_MIN,
#     }
#     return out

# def derive_glitch_phase_thresholds(df: pd.DataFrame) -> dict:
#     """
#     Derive thresholds for glitch_phase_from_edges_dd.
#     Works even if edge_count is missing (fallback to edge_rate).
#     """
#     EDGES_MIN = 3

#     # required cols check
#     need_cols = ["first_edge_pos_ratio", "edge_span_ratio", "tail_edge_rate"]
#     for c in need_cols:
#         if c not in df.columns:
#             # fallback defaults (safe, not too aggressive)
#             return {
#                 "EARLY_MAX": 0.25,
#                 "LATE_MIN": 0.60,
#                 "PERSIST_MIN": 0.50,
#                 "TAIL_RATE_MIN": 0.05,
#                 "EDGES_MIN": EDGES_MIN,
#             }

#     d = df.copy()

#     # choose edge presence signal
#     if "edge_count" in d.columns:
#         edge_present = d["edge_count"].astype(float) >= 1
#         edge_persist = d["edge_count"].astype(float) >= EDGES_MIN
#     elif "edge_rate" in d.columns:
#         # edge_rate is [0,1] ratio; pick a small threshold as "has edges"
#         edge_present = d["edge_rate"].astype(float) > 0.0
#         edge_persist = d["edge_rate"].astype(float) >= 0.02
#     else:
#         return {
#             "EARLY_MAX": 0.25,
#             "LATE_MIN": 0.60,
#             "PERSIST_MIN": 0.50,
#             "TAIL_RATE_MIN": 0.05,
#             "EDGES_MIN": EDGES_MIN,
#         }

#     # keep finite rows
#     for c in need_cols:
#         d = d[np.isfinite(d[c].to_numpy(float))]

#     d_edge = d[edge_present].copy()
#     d_persist = d[edge_persist].copy()

#     # fallback if too few
#     if len(d_edge) < 50:
#         d_edge = d[edge_present].copy()
#     if len(d_persist) < 50:
#         d_persist = d_edge.copy()

#     if len(d_edge) == 0:
#         return {
#             "EARLY_MAX": 0.25,
#             "LATE_MIN": 0.60,
#             "PERSIST_MIN": 0.50,
#             "TAIL_RATE_MIN": 0.05,
#             "EDGES_MIN": EDGES_MIN,
#         }

#     early = float(np.quantile(d_edge["first_edge_pos_ratio"], 0.10))
#     late  = float(np.quantile(d_edge["first_edge_pos_ratio"], 0.90))

#     # persistence/tail thresholds from persist subset (or edge subset)
#     persist = float(np.quantile(d_persist["edge_span_ratio"], 0.70)) if len(d_persist) else 0.50
#     tail_hi = float(np.quantile(d_persist["tail_edge_rate"], 0.70)) if len(d_persist) else 0.05

#     # clamp (กันหลุดโลก)
#     early = float(np.clip(early, 0.02, 0.30))
#     late  = float(np.clip(late,  0.40, 0.95))
#     persist = float(np.clip(persist, 0.15, 0.95))
#     tail_hi = float(max(tail_hi, 0.01))

#     return {
#         "EARLY_MAX": early,
#         "LATE_MIN": late,
#         "PERSIST_MIN": persist,
#         "TAIL_RATE_MIN": tail_hi,
#         "EDGES_MIN": EDGES_MIN,
#     }


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

    df_all = df.dropna(subset=[label]).reset_index(drop=True)
    if len(df_all) == 0:
        raise ValueError(f"No rows with label '{label}' found in training file.")

    # stage-A label: is_fast
    df_all["is_fast"] = (df_all[label].astype(float) <= FAST_MS + 1e-12).astype(int)

    # drop identity/meta
    #df_all = df_all.drop(columns=DROP_ALWAYS, errors="ignore")

    u = df_all["is_fast"].unique()
    if len(u) < 2:
        raise ValueError(f"is_fast has only 1 class in training data: {u}")

    # ---- HOLDOUT SPLIT (train/calib) ----
    df_train, df_calib = stratified_holdout_split(df_all, "is_fast", CALIB_FRAC, SPLIT_SEED)
    
    df_train = df_train.drop(columns=DROP_ALWAYS, errors="ignore")
    df_calib = df_calib.drop(columns=DROP_ALWAYS, errors="ignore")
    save_path = model_dir or DEFAULT_SAVE_PATH
    zero_model_path = os.path.join(save_path, "zero_clf")
    reg_model_path  = os.path.join(save_path, "wait_reg")
    _ensure_dir(save_path)

    log_path = os.path.join(save_path, f"train_log_{ts}.txt")
    append_log(log_path, f"=== TRAIN LOG START {ts} ===")

    gpu_count = 1 if torch.cuda.is_available() else 0
    log_print(log_path, f"🚀 Training device: {'GPU (CUDA)' if gpu_count > 0 else 'CPU'}")
    log_print(log_path, f"data_path={data_path}")
    log_print(log_path, f"rows(all)={len(df_all)} cols={len(df_all.columns)}")
    log_print(log_path, f"rows(train)={len(df_train)} rows(calib)={len(df_calib)} | CALIB_FRAC={CALIB_FRAC} seed={SPLIT_SEED}")
    log_print(log_path, f"presets={presets} time_limit={time_limit}s")
    log_print(log_path, f"FAST_MS={FAST_MS}")
    log_print(log_path, f"policy: MIN_FAST_PRECISION={MIN_FAST_PRECISION} MAX_FAST_FP_RATE={MAX_FAST_FP_RATE} MIN_FAST_RECALL={MIN_FAST_RECALL}")

    fast_cnt = int((df_train["is_fast"] == 1).sum())
    slow_cnt = int((df_train["is_fast"] == 0).sum())
    log_print(log_path, f"[train] label distribution: fast={fast_cnt} slow={slow_cnt} fast_ratio={fast_cnt/max(len(df_train),1):.4f}")

    fast_cnt_c = int((df_calib["is_fast"] == 1).sum())
    slow_cnt_c = int((df_calib["is_fast"] == 0).sum())
    log_print(log_path, f"[calib] label distribution: fast={fast_cnt_c} slow={slow_cnt_c} fast_ratio={fast_cnt_c/max(len(df_calib),1):.4f}")

    # split time
    t_zero = max(30, int(time_limit * 0.25))
    t_reg  = max(60, int(time_limit * 0.75))
    log_print(log_path, f"stage time: clf={t_zero}s reg={t_reg}s")

    cols_to_drop_found = [c for c in COLS_TO_DROP if c in df_train.columns]
    if cols_to_drop_found:
        log_print(log_path, f"Dropping meta columns: {cols_to_drop_found}")

    # glitch_thr = derive_glitch_phase_thresholds(df_calib)   # แนะนำ derive จาก CALIB เหมือน threshold fast
    # save_json(os.path.join(save_path, "glitch_phase_thresholds.json"), glitch_thr)
    # log_print(log_path, f"glitch_thr: {glitch_thr}")


    # =========================
    # Stage A: is_fast classifier (TRAIN ONLY)
    # =========================
    log_print(log_path, "\n=== Stage A: is_fast classifier (TRAIN) ===")

    df_zero = df_train.drop(columns=cols_to_drop_found, errors="ignore").copy()

    # remove label & ids from X
    df_zero = df_zero.drop(columns=[label, "wave_id"], errors="ignore")

    # weight: punish FP on slow
    df_zero["sample_weight"] = np.where(df_zero["is_fast"] == 0, 1.5, 1.0)

    # IMPORTANT: ลด manual drop → เอาออกเฉพาะ identity/debug แน่ ๆ
    ZERO_DROP_EXTRA = [
        "dbg_label_reason",
        "logic_is_periodic",
        "logic_is_ringing",
        "sd", "low_limit", "high_limit",
    ]
    # -------------------------
    # Strong drops from feature audit (v27)
    # -------------------------
    DROP_CONSTANT_OR_DUP = [
        "base_n", "base_t_end_ms",
        #"base_energy",
        # "tail_p2p",
        # "base_max",
        # "base_mean_abs_slope",
        #"post_head_std",
        "edge_count",          # dup of edge_rate
        "tail_edge_count",     # dup of tail_edge_rate
        "shape_undershoot_norm",  # dup of shape_min_to_end_norm
        "edge_span_ratio",     # near-dup of last_edge_pos_ratio
        "ring_peak_count",     # near-dup of ring_mean_peak_spacing_ms
        "meta_abs_step",       # near-dup of meta_denom_used
        "meta_step_raw",       # redundant / makes splits hard
        "glitch_phase",        # rule-derived, do NOT feed model
    ]

    # extra risky for classifier only (avoid FP jumps / quantized proba)
    ZERO_DROP_RISKY = [
        "base_max_slope",
        "base_mean_abs_slope",
        "tail_mean_abs_slope",
        "meta_abs_step_to_noise",
        "edge_max_ratio",
        "ring_peak_decay_ratio",
    ]

    df_zero = df_zero.drop(columns=ZERO_DROP_EXTRA, errors="ignore")
    df_zero = df_zero.drop(columns=DROP_CONSTANT_OR_DUP, errors="ignore")
    df_zero = df_zero.drop(columns=ZERO_DROP_RISKY, errors="ignore")

    # auto drop constant columns
    nuniq = df_zero.nunique(dropna=True)
    const_cols = [c for c in nuniq.index if (nuniq[c] <= 1) and (c not in ["is_fast", "sample_weight"])]
    if const_cols:
        log_print(log_path, f"[zero_clf] auto-drop constant cols: {const_cols}")
        df_zero = df_zero.drop(columns=const_cols, errors="ignore")

    # save feature cols
    zero_feature_cols = [c for c in df_zero.columns if c not in ["is_fast", "sample_weight"]]
    save_json(os.path.join(save_path, "zero_feature_cols.json"), zero_feature_cols)
    log_print(log_path, f"zero_clf feature cols = {len(zero_feature_cols)} (saved zero_feature_cols.json)")

    zero_clf = TabularPredictor(
        label="is_fast",
        path=zero_model_path,
        problem_type="binary",
        eval_metric="precision",           # FP-safe
        verbosity=2,
        sample_weight="sample_weight",
    ).fit(
        train_data=df_zero,
        presets=presets,
        time_limit=t_zero,
        num_gpus=gpu_count,
        dynamic_stacking=False,
    )

    log_print(log_path, "\n=== Leaderboard: zero_clf (classifier) on CALIB ===")

    # df_calib ต้องมี is_fast อยู่แล้ว
    df_zero_calib = df_calib.drop(columns=cols_to_drop_found, errors="ignore").copy()
    df_zero_calib = df_zero_calib.drop(columns=[label, "wave_id"], errors="ignore")

    # ต้อง drop extra เหมือน train เพื่อให้คอลัมน์ตรง
    df_zero_calib = df_zero_calib.drop(columns=ZERO_DROP_EXTRA, errors="ignore")
    df_zero_calib = df_zero_calib.drop(columns=DROP_CONSTANT_OR_DUP, errors="ignore")
    df_zero_calib = df_zero_calib.drop(columns=ZERO_DROP_RISKY, errors="ignore")
    df_zero_calib = align_columns(df_zero_calib, ["is_fast"] + zero_feature_cols)  # หรือ align เฉพาะ zero_feature_cols + label

    lb_zero = zero_clf.leaderboard(
        data=df_zero_calib,
        silent=True,
        extra_metrics=["precision", "recall", "f1"],
    )

    log_print(
        log_path,
        lb_zero[
            ["model", "score_val", "precision", "recall", "f1", "fit_time", "pred_time_val", "stack_level"]
        ].to_string(index=False)
    )
    lb0 = lb_zero.copy()
    best_zero = (
        lb0.sort_values(
            ["precision", "recall", "f1", "pred_time_val"],
            ascending=[False, False, False, True]
        )
        .iloc[0]["model"]
    )

    log_print(log_path, f"\nBest zero_clf model = {best_zero}")
    zero_clf.set_model_best(best_zero)
    save_json(os.path.join(save_path, "zero_best_model.json"), {"best_model": best_zero})


    # =========================
    # Stage B: wait_reg (regression on slow only) (TRAIN ONLY)
    # =========================
    log_print(log_path, "\n=== Stage B: wait_reg (regression on slow only) (TRAIN) ===")

    df_slow = df_train[df_train[label].astype(float) > FAST_MS].copy()
    log_print(log_path, f"slow rows for regressor = {len(df_slow)}")
    if len(df_slow) < 5:
        raise ValueError("slow samples too few to train regressor.")

    df_slow["wait_time_log"] = np.log1p(df_slow[label].astype(float))

    df_reg = df_slow.drop(columns=cols_to_drop_found, errors="ignore").copy()

    REG_DROP_EXTRA = [
        "dbg_label_reason",
        "sd", "low_limit", "high_limit",
        "logic_is_periodic",
        "logic_is_ringing",
    ]
    df_reg = df_reg.drop(columns=REG_DROP_EXTRA, errors="ignore")
    df_reg = df_reg.drop(columns=DROP_CONSTANT_OR_DUP, errors="ignore")

    df_reg_for_fit = df_reg.drop(columns=["wave_id", label, "is_fast"], errors="ignore")

    nuniq = df_reg_for_fit.nunique(dropna=True)
    const_cols = [c for c in nuniq.index if (nuniq[c] <= 1) and (c != "wait_time_log")]
    if const_cols:
        log_print(log_path, f"[wait_reg] auto-drop constant cols: {const_cols}")
        df_reg_for_fit = df_reg_for_fit.drop(columns=const_cols, errors="ignore")


    reg_feature_cols = [c for c in df_reg_for_fit.columns if c != "wait_time_log"]
    reg_feature_cols = [c for c in reg_feature_cols if c not in REG_DROP_EXTRA]
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

    log_print(log_path, "\n=== Feature Importance: wait_reg (permutation) ===")
    try:
        sub_n = min(3000, len(df_reg_for_fit))
        df_reg_sub = df_reg_for_fit.sample(n=sub_n, random_state=0) if len(df_reg_for_fit) > sub_n else df_reg_for_fit

        fi_reg = wait_reg.feature_importance(
            data=df_reg_sub,
            subsample_size=sub_n,
            num_shuffle_sets=5,
            include_confidence_band=True,
        )

        fi_reg_path = os.path.join(save_path, f"feature_importance_reg_{ts}.csv")
        fi_reg.to_csv(fi_reg_path, index=False)
        log_print(log_path, f"saved: {fi_reg_path}")

        topk = 30
        log_print(
            log_path,
            f"\nTop-{topk} important features (wait_reg):\n" +
            fi_reg.sort_values("importance", ascending=False).head(topk).to_string(index=False)
        )
    except Exception as e:
        log_print(log_path, f"[WARN] reg feature_importance failed: {e}")

    # =========================
    # Threshold selection (dual) + AUTO sanity gate (CALIB ONLY)
    # =========================
    log_print(log_path, "\n=== Threshold selection & AUTO sanity gate (CALIB holdout) ===")

    X_all_cal = df_calib.drop(columns=[label], errors="ignore").copy()
    y_true_cal = df_calib[label].to_numpy(dtype=float)

    X_feat_cal = X_all_cal.drop(columns=cols_to_drop_found, errors="ignore").copy()

    # proba fast (calib)
    X_zero_in_cal = X_feat_cal.drop(columns=["wave_id"], errors="ignore").copy()
    X_zero_in_cal = align_columns(X_zero_in_cal, zero_feature_cols)
    proba_is_fast_cal = proba_class1(zero_clf.predict_proba(X_zero_in_cal))

    # reg pred (calib)
    X_reg_in_cal = X_feat_cal.drop(columns=["wave_id", "is_fast", label, "wait_time_log"], errors="ignore").copy()
    X_reg_in_cal = align_columns(X_reg_in_cal, reg_feature_cols)
    wait_log_pred_cal = wait_reg.predict(X_reg_in_cal)
    wait_pred_cal = np.expm1(np.asarray(wait_log_pred_cal, dtype=float))
    wait_pred_cal = np.clip(wait_pred_cal, 0, None)

    # coarse scan on calib
    scan = threshold_scan_report(proba_is_fast_cal, wait_pred_cal, y_true_cal, thr_min=0.10, thr_max=0.90, steps=161)
    scan_path = os.path.join(save_path, f"threshold_scan_calib_{ts}.csv")
    scan.to_csv(scan_path, index=False)
    log_print(log_path, f"saved: {scan_path}")

    thr_high, thr_low, dual_reason = pick_dual_thresholds(
        scan,
        min_precision_high=MIN_FAST_PRECISION,
        max_fp_rate_high=MAX_FAST_FP_RATE,
        min_recall_low=0.995,
        max_fp_rate_low=0.0020,
    )

    # refine around thr_high (calib)
    ref_min = max(0.0, thr_high - 0.03)
    ref_max = min(1.0, thr_high + 0.03)
    scan_ref = threshold_scan_report(proba_is_fast_cal, wait_pred_cal, y_true_cal, thr_min=ref_min, thr_max=ref_max, steps=601)
    scan_ref_path = os.path.join(save_path, f"threshold_scan_ref_calib_{ts}.csv")
    scan_ref.to_csv(scan_ref_path, index=False)
    log_print(log_path, f"saved: {scan_ref_path}")

    thr_high, thr_low, dual_reason = pick_dual_thresholds(
        scan_ref,
        min_precision_high=MIN_FAST_PRECISION,
        max_fp_rate_high=MAX_FAST_FP_RATE,
        min_recall_low=0.995,
        max_fp_rate_low=0.0020,
    )
    if thr_low > thr_high:
        thr_low = thr_high

    thresholds = {
        "thr_high": float(thr_high),
        "thr_low": float(thr_low),
        "fast_ms": float(FAST_MS),
        "policy": "dual_threshold",
        "reason": str(dual_reason),
        "ts": ts,
        "calib_frac": CALIB_FRAC,
        "split_seed": SPLIT_SEED,
    }
    save_json(os.path.join(save_path, "fast_thresholds.json"), thresholds)
    log_print(log_path, f"dual_thresholds(calib): high={thr_high:.6f} low={thr_low:.6f} | {dual_reason}")

    # =========================
    # Feature importance (zero_clf) - FP-safe @ thr_high (CALIB)
    # =========================
    log_print(log_path, "\n=== Feature Importance: zero_clf (FP-safe @thr_high) on CALIB ===")
    try:
        X_zero_imp = X_feat_cal.drop(columns=["wave_id"], errors="ignore").copy()
        X_zero_imp = align_columns(X_zero_imp, zero_feature_cols)

        imp_fp = fp_safe_feature_importance(
            zero_clf,
            X_zero_imp,
            y_true_cal,
            thr_high=thr_high,
            fast_ms=FAST_MS,
            n_repeat=5,
            subsample=3000,
            random_state=0,
        )

        imp_fp_path = os.path.join(save_path, f"fp_safe_importance_zero_calib_{ts}.csv")
        imp_fp.to_csv(imp_fp_path, index=False)
        log_print(log_path, f"saved: {imp_fp_path}")

        log_print(log_path, "\nTop-25 FP-risk features (zero_clf, calib):\n" +
                imp_fp.head(25).to_string(index=False))

    except Exception as e:
        log_print(log_path, f"[WARN] fp-safe importance (zero_clf) failed: {e}")

    # --- AUTO sanity gate derived from CALIB (not train) ---
    sanity = derive_sanity_gate_from_train(
        df_train=df_calib,
        y_true_ms=y_true_cal,
        proba_is_fast=proba_is_fast_cal,
        wait_pred=wait_pred_cal,
        thr_high=thr_high,
        thr_low=thr_low,
    )

    X_sanity_cal = X_feat_cal.drop(columns=["wave_id"], errors="ignore").copy()

    save_json(os.path.join(save_path, "sanity_gate.json"), sanity)
    log_print(
        log_path,
        f"sanity_gate(auto,calib): step_max={sanity.get('step_max', float('nan')):.6f} "
        f"ac_max={sanity.get('ac_max', float('nan')):.6f} "
        f"tail_std_max={sanity.get('tail_std_max', float('nan')):.6f} "
        f"late_activity_max={sanity.get('late_activity_max', float('nan')):.6f} "
        f"reg_fp_ms={sanity.get('reg_fp_ms', float('nan')):.6f}"
    )

    # =========================
    # Final CALIB metrics (dual + sanity)
    # =========================
    is_fast_pred_cal = apply_dual_threshold_with_sanity(
        proba_is_fast=proba_is_fast_cal,
        wait_pred=wait_pred_cal,
        df_feats=X_sanity_cal,
        thr_high=thr_high,
        thr_low=thr_low,
        sanity=sanity,
    )
    y_hat_cal = np.where(is_fast_pred_cal, FAST_MS, wait_pred_cal)

    mae_all_cal = float(np.mean(np.abs(y_hat_cal - y_true_cal)))
    slow_mask_cal = (y_true_cal > FAST_MS + 1e-12)
    mae_slow_cal = float(np.mean(np.abs(y_hat_cal[slow_mask_cal] - y_true_cal[slow_mask_cal]))) if np.any(slow_mask_cal) else float("nan")

    fast_true_cal = (y_true_cal <= FAST_MS + 1e-12)
    tp = int(np.sum(is_fast_pred_cal & fast_true_cal))
    fn = int(np.sum((~is_fast_pred_cal) & fast_true_cal))
    fp = int(np.sum(is_fast_pred_cal & (~fast_true_cal)))
    tn = int(np.sum((~is_fast_pred_cal) & (~fast_true_cal)))

    fast_precision_cal = tp / max(tp + fp, 1)
    fast_recall_cal = tp / max(int(np.sum(fast_true_cal)), 1)
    fast_fp_rate_cal = fp / max(int(np.sum(~fast_true_cal)), 1)

    cm_path = os.path.join(save_path, f"confusion_heatmap_calib_{ts}.png")
    plot_confusion_heatmap(
        tp, fp, fn, tn,
        title=f"CALIB FAST Confusion @ high={thr_high:.3f}, low={thr_low:.3f}",
        save_path=cm_path
    )
    log_print(log_path, f"[saved] {cm_path}")

    log_print(log_path, f"\nCALIB MAE(all)={mae_all_cal:.6f} | MAE(slow-only)={mae_slow_cal:.6f}")
    log_print(log_path, f"CALIB FAST precision={fast_precision_cal:.6f} recall={fast_recall_cal:.6f} fp_rate={fast_fp_rate_cal:.6f}")
    log_print(
        log_path,
        f"CALIB proba_is_fast min/mean/max = "
        f"{proba_is_fast_cal.min():.6f} {proba_is_fast_cal.mean():.6f} {proba_is_fast_cal.max():.6f}"
    )
    log_print(log_path, f"CALIB predicted fast = {int(is_fast_pred_cal.sum())} / {len(is_fast_pred_cal)}")

    # =========================
    # (Optional) Also report TRAIN metrics using CALIB-derived thresholds/sanity
    # (ช่วยดูว่ามี overfit ไหม)
    # =========================
    log_print(log_path, "\n=== (FYI) Metrics on TRAIN using CALIB-derived thresholds/sanity ===")

    X_all_tr = df_train.drop(columns=[label], errors="ignore").copy()
    y_true_tr = df_train[label].to_numpy(dtype=float)
    X_feat_tr = X_all_tr.drop(columns=cols_to_drop_found, errors="ignore").copy()

    X_zero_in_tr = X_feat_tr.drop(columns=["wave_id"], errors="ignore").copy()
    X_zero_in_tr = align_columns(X_zero_in_tr, zero_feature_cols)
    proba_is_fast_tr = proba_class1(zero_clf.predict_proba(X_zero_in_tr))

    X_reg_in_tr = X_feat_tr.drop(columns=["wave_id", "is_fast", label, "wait_time_log"], errors="ignore").copy()
    X_reg_in_tr = align_columns(X_reg_in_tr, reg_feature_cols)
    wait_log_pred_tr = wait_reg.predict(X_reg_in_tr)
    wait_pred_tr = np.expm1(np.asarray(wait_log_pred_tr, dtype=float))
    wait_pred_tr = np.clip(wait_pred_tr, 0, None)

    X_sanity_tr = X_feat_tr.drop(columns=["wave_id"], errors="ignore").copy()

    is_fast_pred_tr = apply_dual_threshold_with_sanity(
        proba_is_fast=proba_is_fast_tr,
        wait_pred=wait_pred_tr,
        df_feats=X_sanity_tr,
        thr_high=thr_high,
        thr_low=thr_low,
        sanity=sanity,
    )
    y_hat_tr = np.where(is_fast_pred_tr, FAST_MS, wait_pred_tr) # ผลลัพธ์สุดท้ายของระบบบน TRAIN เมื่อเอา threshold + sanity

    mae_all_tr = float(np.mean(np.abs(y_hat_tr - y_true_tr)))
    slow_mask_tr = (y_true_tr > FAST_MS + 1e-12)
    mae_slow_tr = float(np.mean(np.abs(y_hat_tr[slow_mask_tr] - y_true_tr[slow_mask_tr]))) if np.any(slow_mask_tr) else float("nan")

    fast_true_tr = (y_true_tr <= FAST_MS + 1e-12)
    tp_t = int(np.sum(is_fast_pred_tr & fast_true_tr))
    fn_t = int(np.sum((~is_fast_pred_tr) & fast_true_tr))
    fp_t = int(np.sum(is_fast_pred_tr & (~fast_true_tr)))
    tn_t = int(np.sum((~is_fast_pred_tr) & (~fast_true_tr)))

    fast_precision_tr = tp_t / max(tp_t + fp_t, 1)
    fast_recall_tr = tp_t / max(int(np.sum(fast_true_tr)), 1)
    fast_fp_rate_tr = fp_t / max(int(np.sum(~fast_true_tr)), 1)

    log_print(log_path, f"TRAIN MAE(all)={mae_all_tr:.6f} | MAE(slow-only)={mae_slow_tr:.6f}")
    log_print(log_path, f"TRAIN FAST precision={fast_precision_tr:.6f} recall={fast_recall_tr:.6f} fp_rate={fast_fp_rate_tr:.6f}")
    log_print(log_path, f"TRAIN predicted fast = {int(is_fast_pred_tr.sum())} / {len(is_fast_pred_tr)}")

    # =========================
    # Diagnostics CSVs
    # =========================
    diag_cal = df_calib[["wave_id", label, "is_fast"]].copy()
    diag_cal["proba_is_fast"] = proba_is_fast_cal
    diag_cal["reg_wait_pred_ms"] = wait_pred_cal
    diag_cal["pred_is_fast"] = is_fast_pred_cal.astype(int)
    diag_cal["pred_wait_ms"] = y_hat_cal
    diag_cal["abs_error"] = np.abs(diag_cal["pred_wait_ms"].to_numpy(float) - diag_cal[label].to_numpy(float))

    diag_cal_path = os.path.join(save_path, f"calib_diagnostics_{ts}.csv")
    diag_cal.sort_values("abs_error", ascending=False).to_csv(diag_cal_path, index=False)
    log_print(log_path, f"saved: {diag_cal_path}")
    log_print(log_path, "\nWorst 15 CALIB rows (by abs_error):\n" +
              diag_cal.sort_values("abs_error", ascending=False).head(15).to_string(index=False))

    diag_tr = df_train[["wave_id", label, "is_fast"]].copy()
    diag_tr["proba_is_fast"] = proba_is_fast_tr
    diag_tr["reg_wait_pred_ms"] = wait_pred_tr
    diag_tr["pred_is_fast"] = is_fast_pred_tr.astype(int)
    diag_tr["pred_wait_ms"] = y_hat_tr
    diag_tr["abs_error"] = np.abs(diag_tr["pred_wait_ms"].to_numpy(float) - diag_tr[label].to_numpy(float))

    diag_tr_path = os.path.join(save_path, f"train_diagnostics_{ts}.csv")
    diag_tr.sort_values("abs_error", ascending=False).to_csv(diag_tr_path, index=False)
    log_print(log_path, f"saved: {diag_tr_path}")
    log_print(log_path, "\nWorst 15 TRAIN rows (by abs_error):\n" +
              diag_tr.sort_values("abs_error", ascending=False).head(15).to_string(index=False))

    # =========================
    # Summary
    # =========================
    log_print(log_path, "\n✅ Saved models at: " + save_path)
    log_print(log_path, f" - zero_clf: {zero_model_path}")
    log_print(log_path, f" - wait_reg: {reg_model_path}")
    log_print(log_path, f" - thresholds: fast_thresholds.json")
    log_print(log_path, f" - sanity: sanity_gate.json")
    log_print(log_path, f" - log file: {log_path}")
    log_print(log_path, "=== TRAIN LOG END ===")


def main():
    ap = argparse.ArgumentParser("Train AutoGluon v27 (holdout calib): 2-stage + FP-safe dual threshold + AUTO sanity gate")
    ap.add_argument("--data", required=True, help="train features csv (must include wait_time_ms)")
    ap.add_argument("--label", default="wait_time_ms")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--time-limit", type=int, default=300)
    ap.add_argument("--presets", default="medium_quality")
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
