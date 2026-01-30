# scripts/train_ag_v27.py
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
from typing import Any

"""
v27 goals
- ลด manual tuning: threshold + sanity gate derive จาก train data
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
    is_fast_true = (y_true <= FAST_MS + 1e-12)

    rows = []
    thr_grid = np.linspace(thr_min, thr_max, steps)

    for thr in thr_grid:
        is_fast_pred = proba_is_fast >= thr
        y_hat = np.where(is_fast_pred, FAST_MS, wait_pred)
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
    q_reg_fp: float = 0.95,
    core_frac: float = 0.20,   # ✅ data-driven: lowest 20% of strict -> EDGE, rest -> CORE
    max_strict_margin: float = 0.20,
) -> dict:
    """
    Derive sanity gate thresholds from training data.

    Policy:
    - We do NOT sanity-gate STRICT_CORE
    - We sanity-gate STRICT_EDGE + SOFT (FP firewall)
    - strict_margin is derived from the distribution of p among strict predictions.
    """

    y_true_ms = np.asarray(y_true_ms, float)
    p = np.asarray(proba_is_fast, float)
    wait_pred = np.asarray(wait_pred, float)

    global_fast = (y_true_ms <= FAST_MS + 1e-12)

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
            pd.DataFrame({"reg": wait_pred[global_fast]}),
            "reg",
            q=0.99, hard_min=0.12, hard_max=0.8, default=0.5
        ),
    }

    # -------------------------
    # ✅ strict_margin (data-driven)
    # -------------------------
    # Define a "core cut" inside strict predictions: keep top (1-core_frac) as CORE
    # and gate only the bottom core_frac part as STRICT_EDGE.
    strict_mask = (p >= float(thr_high))
    if np.any(strict_mask):
        pp = p[strict_mask]
        # core_cut >= thr_high
        core_cut = float(np.quantile(pp, np.clip(core_frac, 0.01, 0.49)))
        strict_margin = float(np.clip(core_cut - float(thr_high), 0.0, max_strict_margin))
    else:
        strict_margin = 0.0

    # -------------------------
    # build mask used to derive feature thresholds (use true-fast near boundary)
    # -------------------------
    is_fast_true = (y_true_ms <= FAST_MS + 1e-12)
    near_soft = (p >= float(thr_low)) & (p < float(thr_high))
    mask = is_fast_true & near_soft
    if not np.any(mask):
        mask = is_fast_true

    out: dict[str, Any] = {
        "enabled": True,
        "features_present": {},
        "policy": "soft+strict_edge_fp_firewall",
        "thr_high": float(thr_high),
        "thr_low": float(thr_low),
        "strict_margin": float(strict_margin),
        "strict_core_frac": float(core_frac),
        "notes": (
            f"Derived from train true-fast & near(thr_low={thr_low:.6f}); "
            f"q_feat={q_feat}, q_reg_fp={q_reg_fp}. "
            f"Sanity applies to SOFT + STRICT_EDGE only. "
            f"strict_margin derived from p>=thr_high using core_frac={core_frac:.2f}."
        ),
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

    # --- tail_std ---
    f_tail_std = "tail_std"
    if f_tail_std in df_train.columns:
        out["features_present"][f_tail_std] = True
        v = df_train.loc[mask, f_tail_std].to_numpy(float)
        out["tail_std_max"] = float(_safe_quantile(v, q_feat, DEFAULTS["tail_std_max"]))
    else:
        out["features_present"][f_tail_std] = False
        out["tail_std_max"] = DEFAULTS["tail_std_max"]

    # --- late_activity ---
    f_late = "late_activity"
    if f_late in df_train.columns:
        out["features_present"][f_late] = True
        v = df_train.loc[mask, f_late].to_numpy(float)
        out["late_activity_max"] = float(_safe_quantile(v, q_feat, DEFAULTS["late_activity_max"]))
    else:
        out["features_present"][f_late] = False
        out["late_activity_max"] = DEFAULTS["late_activity_max"]

    # --- reg veto (FP firewall) derived from reg_pred of true-fast near boundary
    reg_vals = wait_pred[mask]
    reg_fp = _safe_quantile(reg_vals, q_reg_fp, DEFAULTS["reg_fp_ms"])
    out["reg_fp_ms"] = float(np.clip(reg_fp, 0.12, 0.60))

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
    """
    3-zone policy:
      - STRICT_CORE: p >= thr_high + strict_margin         -> always FAST (no sanity)
      - STRICT_EDGE: thr_high <= p < thr_high + margin     -> FAST only if sanity_ok
      - SOFT:        thr_low  <= p < thr_high              -> FAST only if sanity_ok
      - else SLOW
    """
    p = np.asarray(proba_is_fast, float)

    strict_margin = float(sanity.get("strict_margin", 0.0))
    strict_margin = float(np.clip(strict_margin, 0.0, 0.5))

    is_fast_core = (p >= (thr_high + strict_margin))
    is_fast_edge = (p >= thr_high) & (p < (thr_high + strict_margin))
    is_fast_soft = (p >= thr_low) & (p < thr_high)

    # sanity applies ONLY to edge+soft
    need_sanity = (is_fast_edge | is_fast_soft)

    sanity_ok = np.ones_like(p, dtype=bool)

    # --- feature gates (derived) ---
    step_max = float(sanity.get("step_max", 0.35))
    ac_max   = float(sanity.get("ac_max", 0.55))

    f_step = SANITY_FEATURES.get("meta_step_to_span", "meta_step_to_span")
    if f_step in df_feats.columns:
        sanity_ok &= (df_feats[f_step].to_numpy(float) <= step_max)

    f_ac = SANITY_FEATURES.get("per_tail_ac_best", "per_tail_ac_best")
    if f_ac in df_feats.columns:
        sanity_ok &= (df_feats[f_ac].to_numpy(float) <= ac_max)

    tail_std_max = float(sanity.get("tail_std_max", np.inf))
    if "tail_std" in df_feats.columns and np.isfinite(tail_std_max):
        sanity_ok &= (df_feats["tail_std"].to_numpy(float) <= tail_std_max)

    late_activity_max = float(sanity.get("late_activity_max", np.inf))
    if "late_activity" in df_feats.columns and np.isfinite(late_activity_max):
        sanity_ok &= (df_feats["late_activity"].to_numpy(float) <= late_activity_max)

    # --- FP firewall using reg prediction (edge+soft only) ---
    reg_fp_ms = float(sanity.get("reg_fp_ms", np.inf))
    if np.isfinite(reg_fp_ms):
        sanity_ok &= (np.asarray(wait_pred, float) <= reg_fp_ms)

    # final
    return is_fast_core | (need_sanity & sanity_ok)




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

    # เลือกเฉพาะ feature cols
    if feat_cols is None:
        feat_cols = list(X0.columns)

    # drop all-NaN + constant (กันเสียเวลา)
    X0 = X0[feat_cols].copy()
    X0 = X0.dropna(axis=1, how="all")
    nunique = X0.nunique(dropna=True)
    X0 = X0.loc[:, nunique > 1]
    feat_cols = list(X0.columns)

    # subsample เพื่อความเร็ว
    if subsample is not None and len(X0) > subsample:
        # ✅ เก็บ index เดิมก่อน แล้วค่อย slice y_true_ms ตาม index เดิม
        idx = X0.sample(n=subsample, random_state=random_state).index
        X0 = X0.loc[idx].copy()
        y_true_ms = np.asarray(y_true_ms, float)[idx.to_numpy()]
        # (ถ้าอยากให้สวยค่อย reset หลังจาก slice y_true_ms แล้ว)
        X0 = X0.reset_index(drop=True)
        y_true_ms = np.asarray(y_true_ms, float)  # now aligned with X0 rows
        assert len(X0) == len(y_true_ms), "X0 and y_true_ms length mismatch"

    y_true_ms = np.asarray(y_true_ms, float)
    fast_true = (y_true_ms <= fast_ms + 1e-12)
    slow_total = int((~fast_true).sum())

    # baseline
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
    # sort ตาม FP ก่อน (สำคัญสุด)
    out = out.sort_values(["fp_rate_increase", "precision_drop"], ascending=[False, False]).reset_index(drop=True)
    return out


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

    # stage-A label: is_fast
    df_train["is_fast"] = (df_train[label].astype(float) <= FAST_MS + 1e-12).astype(int)

    # drop identity/meta
    df_train = df_train.drop(columns=DROP_ALWAYS, errors="ignore")

    u = df_train["is_fast"].unique()
    if len(u) < 2:
        raise ValueError(f"is_fast has only 1 class in training data: {u}")

    save_path = model_dir or DEFAULT_SAVE_PATH
    zero_model_path = os.path.join(save_path, "zero_clf")
    reg_model_path  = os.path.join(save_path, "wait_reg")
    _ensure_dir(save_path)

    log_path = os.path.join(save_path, f"train_log_{ts}.txt")
    append_log(log_path, f"=== TRAIN LOG START {ts} ===")

    gpu_count = 1 if torch.cuda.is_available() else 0
    log_print(log_path, f"🚀 Training device: {'GPU (CUDA)' if gpu_count > 0 else 'CPU'}")
    log_print(log_path, f"data_path={data_path}")
    log_print(log_path, f"rows={len(df_train)} cols={len(df_train.columns)}")
    log_print(log_path, f"presets={presets} time_limit={time_limit}s")
    log_print(log_path, f"FAST_MS={FAST_MS}")
    log_print(log_path, f"policy: MIN_FAST_PRECISION={MIN_FAST_PRECISION} MAX_FAST_FP_RATE={MAX_FAST_FP_RATE} MIN_FAST_RECALL={MIN_FAST_RECALL}")

    fast_cnt = int((df_train["is_fast"] == 1).sum())
    slow_cnt = int((df_train["is_fast"] == 0).sum())
    log_print(log_path, f"label distribution: fast={fast_cnt} slow={slow_cnt} fast_ratio={fast_cnt/max(len(df_train),1):.4f}")

    # split time
    t_zero = max(30, int(time_limit * 0.25))
    t_reg  = max(60, int(time_limit * 0.75))
    log_print(log_path, f"stage time: clf={t_zero}s reg={t_reg}s")

    cols_to_drop_found = [c for c in COLS_TO_DROP if c in df_train.columns]
    if cols_to_drop_found:
        log_print(log_path, f"Dropping meta columns: {cols_to_drop_found}")

    # =========================
    # Stage A: is_fast classifier
    # =========================
    log_print(log_path, "\n=== Stage A: is_fast classifier ===")

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
        "sd", "low_limit", "high_limit",  # meta ที่อาจ leak/ไม่เสถียร
    ]
    df_zero = df_zero.drop(columns=ZERO_DROP_EXTRA, errors="ignore")

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

    # =========================
    # Feature importance (zero_clf) - normal (permutation)
    # =========================
    # log_print(log_path, "\n=== Feature Importance: zero_clf (permutation) ===")

    # # X สำหรับ importance ต้องมี label ด้วย (is_fast) และ sample_weight ถ้าอยากให้สอดคล้อง train
    # # ตอนนี้ df_zero มี is_fast + sample_weight อยู่แล้ว
    # try:
    #     # กันช้าเกิน: สุ่มบางส่วน
    #     sub_n = min(3000, len(df_zero))
    #     df_zero_sub = df_zero.sample(n=sub_n, random_state=0) if len(df_zero) > sub_n else df_zero

    #     fi = zero_clf.feature_importance(
    #         data=df_zero_sub,          # ต้องมี label column "is_fast" อยู่ใน df
    #         subsample_size=sub_n,      # ให้ตรงกับ sample ที่ส่งเข้าไป
    #         num_shuffle_sets=5,        # ยิ่งมากยิ่งนิ่ง แต่ช้าขึ้น
    #         include_confidence_band=True,
    #     )

    #     fi_path = os.path.join(save_path, f"feature_importance_zero_{ts}.csv")
    #     fi.to_csv(fi_path, index=False)
    #     log_print(log_path, f"saved: {fi_path}")

    #     topk = 30
    #     log_print(
    #         log_path,
    #         f"\nTop-{topk} important features (zero_clf):\n" +
    #         fi.sort_values("importance", ascending=False).head(topk).to_string(index=False)
    #     )
    # except Exception as e:
    #     log_print(log_path, f"[WARN] feature_importance failed: {e}")


    # =========================
    # Stage B: wait_reg (slow only)
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
    # Threshold selection (dual) + AUTO sanity gate
    # =========================
    log_print(log_path, "\n=== Threshold selection & AUTO sanity gate (v27) ===")

    X_all = df_train.drop(columns=[label], errors="ignore").copy()
    y_true = df_train[label].to_numpy(dtype=float)

    X_feat = X_all.drop(columns=cols_to_drop_found, errors="ignore").copy()

    # proba fast
    X_zero_in = X_feat.drop(columns=["wave_id"], errors="ignore").copy()
    X_zero_in = align_columns(X_zero_in, zero_feature_cols)
    proba_is_fast = proba_class1(zero_clf.predict_proba(X_zero_in))

    # reg pred (all rows)
    X_reg_in = X_feat.drop(columns=["wave_id", "is_fast", label, "wait_time_log"], errors="ignore").copy()
    X_reg_in = align_columns(X_reg_in, reg_feature_cols)
    wait_log_pred = wait_reg.predict(X_reg_in)
    wait_pred = np.expm1(np.asarray(wait_log_pred, dtype=float))
    wait_pred = np.clip(wait_pred, 0, None)

    # coarse scan
    scan = threshold_scan_report(proba_is_fast, wait_pred, y_true, thr_min=0.10, thr_max=0.90, steps=161)
    scan_path = os.path.join(save_path, f"threshold_scan_{ts}.csv")
    scan.to_csv(scan_path, index=False)
    log_print(log_path, f"saved: {scan_path}")

    thr_high, thr_low, dual_reason = pick_dual_thresholds(
        scan,
        min_precision_high=MIN_FAST_PRECISION,
        max_fp_rate_high=MAX_FAST_FP_RATE,
        min_recall_low=0.995,
        max_fp_rate_low=0.0020,
    )

    # refine around thr_high
    ref_min = max(0.0, thr_high - 0.03)
    ref_max = min(1.0, thr_high + 0.03)
    scan_ref = threshold_scan_report(proba_is_fast, wait_pred, y_true, thr_min=ref_min, thr_max=ref_max, steps=601)
    scan_ref_path = os.path.join(save_path, f"threshold_scan_ref_{ts}.csv")
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
    }
    save_json(os.path.join(save_path, "fast_thresholds.json"), thresholds)
    log_print(log_path, f"dual_thresholds: high={thr_high:.6f} low={thr_low:.6f} | {dual_reason}")

    # =========================
    # Feature importance (zero_clf) - FP-safe @ thr_high
    # =========================
    log_print(log_path, "\n=== Feature Importance: zero_clf (FP-safe @thr_high) ===")
    y_true_ms_all = y_true
    log_print(log_path, f"[DBG] y_true_ms_all range: {y_true_ms_all.min():.3f}..{y_true_ms_all.max():.3f}")
    try:
        # X เหมือนตอน predict_proba (align แล้ว)
        X_zero_imp = X_feat.drop(columns=["wave_id"], errors="ignore").copy()
        X_zero_imp = align_columns(X_zero_imp, zero_feature_cols)

        imp_fp = fp_safe_feature_importance(
            zero_clf,
            X_zero_imp,
            y_true_ms_all,
            thr_high=thr_high,
            fast_ms=FAST_MS,
            n_repeat=5,
            subsample=3000,
            random_state=0,
        )

        imp_fp_path = os.path.join(save_path, f"fp_safe_importance_zero_{ts}.csv")
        imp_fp.to_csv(imp_fp_path, index=False)
        log_print(log_path, f"saved: {imp_fp_path}")

        log_print(log_path, "\nTop-25 FP-risk features (zero_clf):\n" +
                imp_fp.head(25).to_string(index=False))

    except Exception as e:
        log_print(log_path, f"[WARN] fp-safe importance (zero_clf) failed: {e}")


    # --- AUTO sanity gate (no manual X/Y) ---
    sanity = derive_sanity_gate_from_train(
        df_train=df_train,
        y_true_ms=y_true,
        proba_is_fast=proba_is_fast,
        wait_pred=wait_pred,
        thr_high=thr_high,
        thr_low=thr_low,
    )

    X_sanity_train = X_feat.drop(columns=["wave_id"], errors="ignore")
    
    save_json(os.path.join(save_path, "sanity_gate.json"), sanity)
    log_print(log_path, f"sanity_gate(auto): step_max={sanity['step_max']:.6f} ac_max={sanity['ac_max']:.6f}")

    # =========================
    # Final train metrics with dual+sanity
    # =========================
    is_fast_pred = apply_dual_threshold_with_sanity(
        proba_is_fast=proba_is_fast,
        wait_pred=wait_pred,
        #df_feats=X_zero_in,   # contains sanity features if exist
        df_feats=X_sanity_train,
        thr_high=thr_high,
        thr_low=thr_low,
        sanity=sanity,
    )
    y_hat = np.where(is_fast_pred, FAST_MS, wait_pred)

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

    cm_path = os.path.join(save_path, f"confusion_heatmap_{ts}.png")
    plot_confusion_heatmap(tp, fp, fn, tn, title=f"FAST Confusion @ high={thr_high:.3f}, low={thr_low:.3f}", save_path=cm_path)
    log_print(log_path, f"[saved] {cm_path}")

    log_print(log_path, f"\nTrain MAE(all)={mae_all:.6f} | MAE(slow-only)={mae_slow:.6f}")
    log_print(log_path, f"FAST precision={fast_precision:.6f} recall={fast_recall:.6f} fp_rate={fast_fp_rate:.6f}")
    log_print(log_path, f"proba_is_fast min/mean/max = {proba_is_fast.min():.6f} {proba_is_fast.mean():.6f} {proba_is_fast.max():.6f}")
    log_print(log_path, f"predicted fast = {int(is_fast_pred.sum())} / {len(is_fast_pred)}")

    diag = df_train[["wave_id", label, "is_fast"]].copy()
    diag["proba_is_fast"] = proba_is_fast
    diag["reg_wait_pred_ms"] = wait_pred
    diag["pred_is_fast"] = is_fast_pred.astype(int)
    diag["pred_wait_ms"] = y_hat
    diag["abs_error"] = np.abs(diag["pred_wait_ms"].to_numpy(float) - diag[label].to_numpy(float))

    diag_path = os.path.join(save_path, f"train_diagnostics_{ts}.csv")
    diag.sort_values("abs_error", ascending=False).to_csv(diag_path, index=False)
    log_print(log_path, f"saved: {diag_path}")
    log_print(log_path, "\nWorst 15 rows (by abs_error):\n" + diag.sort_values("abs_error", ascending=False).head(15).to_string(index=False))

    log_print(log_path, "\n✅ Saved models at: " + save_path)
    log_print(log_path, f" - zero_clf: {zero_model_path}")
    log_print(log_path, f" - wait_reg: {reg_model_path}")
    log_print(log_path, f" - thresholds: fast_thresholds.json")
    log_print(log_path, f" - sanity: sanity_gate.json")
    log_print(log_path, f" - log file: {log_path}")
    log_print(log_path, "=== TRAIN LOG END ===")

def main():
    ap = argparse.ArgumentParser("Train AutoGluon v27 (2-stage: fast-clf + slow-reg) + FP-safe dual threshold + AUTO sanity gate")
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
