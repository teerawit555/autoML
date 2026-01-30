from __future__ import annotations

import argparse
import json
import os
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

# Config columns to drop
COLS_TO_DROP = ["force_mA", "range_V", "temp_C"]
DROP_ALWAYS = ["type"]

# ---------------------------------------------------------
# 1. Heuristic Flag Logic (New Function)
# ---------------------------------------------------------
def settle_need_more_sample_flags(
    df_feat: pd.DataFrame,
    pred_ms: np.ndarray,
    window_ms: float = 10.0,
) -> pd.DataFrame:
    """
    Flag ว่า 'อาจยังไม่ settle ใน window' แบบไม่มั่ว:
    - จะพิจารณาเฉพาะเคสที่ pred ใกล้ท้าย window ก่อน (gating)
    - แล้วค่อยดู tail creep / slope / std / late crossing
    - ตัด tail_monotonicity ออกเพราะทำให้ flag มั่วใน 10ms window
    """
    def g(col, default=0.0):
        return (
            df_feat[col].to_numpy(dtype=float)
            if col in df_feat.columns
            else np.full(len(df_feat), default, dtype=float)
        )

    pred_ms = np.asarray(pred_ms, dtype=float)

    # features
    tail_creep_norm = g("tail_creep_norm", 0.0)
    tail100_slope_norm = g("tail100_slope_norm", 0.0)
    std_tail_50 = g("std_tail_50", 0.0)
    last_cross_tail_ratio = g("last_cross_tail_ratio", 1.0)
    envelope_ratio = g("envelope_ratio", 0.0)

    # optional periodic detectors (ถ้ามีใน feature set)
    logic_per_score = g("logic_per_score", 0.0)             # periodic score
    logic_crossing_rate = g("logic_crossing_rate", 0.0)     # tail crossing rate
    ring_peak_count = g("ring_peak_count", 0.0)             # peak count

    # -------------------------
    # 1) Gate ก่อน (กันมั่ว)
    # -------------------------
    # ถ้า pred ยังต่ำ ๆ (< 70% ของ window) ส่วนใหญ่ไม่ควร flag
    gate = (pred_ms >= 0.70 * window_ms)

    # ถ้า pred ใกล้สุด ๆ (ชนท้าย) ให้ gate ผ่านเลย
    gate |= (pred_ms >= (window_ms - 0.25))  # 9.75ms สำหรับ 10ms

    # -------------------------
    # 2) Exclude periodic-ish (มักไม่ใช่ need more sample)
    # -------------------------
    is_periodicish = (
        (logic_per_score > 0.45) |
        (logic_crossing_rate > 0.20) |
        (ring_peak_count >= 6)
    )

    # -------------------------
    # 3) Tail instability tests
    # -------------------------
    TH_CREEP = 0.015
    TH_SLOPE = 0.006
    TH_STD_TAIL = 0.0030
    TH_LAST_CROSS = 0.035
    TH_ENV = 3.0

    c1 = (np.abs(tail_creep_norm) > TH_CREEP)
    c2 = (np.abs(tail100_slope_norm) > TH_SLOPE)
    c3 = (std_tail_50 > TH_STD_TAIL)
    c4 = (last_cross_tail_ratio < TH_LAST_CROSS)
    c5 = (envelope_ratio > TH_ENV)

    score = c1.astype(float) + c2.astype(float) + c3.astype(float) + c4.astype(float) + c5.astype(float)

    # -------------------------
    # 4) Final decision
    # -------------------------
    # ต้องผ่าน gate + ไม่ periodic-ish + มีอาการ >= 2
    need = gate & (~is_periodicish) & (score >= 2.0)

    reasons = []
    for i in range(len(df_feat)):
        if not need[i]:
            reasons.append("")
            continue
        r = []
        if c1[i]: r.append("tail_creep")
        if c2[i]: r.append("tail_slope")
        if c3[i]: r.append("tail_std_high")
        if c4[i]: r.append("late_crossing")
        if c5[i]: r.append("mid_env_high")
        reasons.append("|".join(r))

    return pd.DataFrame({
        "need_more_sample": need.astype(int),
        "need_more_reason": reasons,
        "need_more_score": score.astype(float),
    })


# ---------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_json(path: str, default=None):
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

def proba_class1(p) -> np.ndarray:
    if hasattr(p, "columns") and (1 in list(p.columns)):
        return p[1].to_numpy()
    arr = np.asarray(p)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    return arr.astype(float)

# ---------------------------------------------------------
# 3. Main Prediction Logic
# ---------------------------------------------------------
def predict(model_path: str, input_csv: str, out_csv: str, min_output_ms: float | None) -> None:
    zero_model_path = os.path.join(model_path, "zero_clf")
    reg_model_path = os.path.join(model_path, "wait_reg")

    if not os.path.exists(zero_model_path) or not os.path.exists(reg_model_path):
        raise FileNotFoundError(f"Missing models at {zero_model_path} or {reg_model_path}")

    zero_feature_cols = load_json(os.path.join(model_path, "zero_feature_cols.json"))
    reg_feature_cols = load_json(os.path.join(model_path, "reg_feature_cols.json"))

    # Load Threshold
    threshold = float(load_text(os.path.join(model_path, "zero_threshold.txt"), default="0.5"))
    print(f"🔮 Predicting: {input_csv}")
    print(f"   Using zero threshold = {threshold:.3f}")

    # Load Models
    zero_clf = TabularPredictor.load(zero_model_path)
    wait_reg = TabularPredictor.load(reg_model_path)

    # Load Data
    df = pd.read_csv(input_csv)
    df = df.drop(columns=DROP_ALWAYS, errors="ignore")
    wave_id_backup = df["wave_id"].copy() if "wave_id" in df.columns else None

    cols_to_drop_found = [c for c in COLS_TO_DROP if c in df.columns]
    df_feat = df.drop(columns=cols_to_drop_found, errors="ignore").copy()
    df_feat = df_feat.drop(columns=["wait_time_ms", "wait_time_log", "is_zero"], errors="ignore")

    # ---- Stage A: Zero Classifier ----
    X_zero_in = df_feat.drop(columns=["wave_id"], errors="ignore").copy()
    X_zero_in = align_columns(X_zero_in, zero_feature_cols)

    proba = zero_clf.predict_proba(X_zero_in)
    proba_is_zero = proba_class1(proba)
    is_zero_pred = proba_is_zero >= threshold

    # ---- Stage B: Wait Regressor ----
    X_reg_in = df_feat.drop(columns=["wave_id"], errors="ignore").copy()
    X_reg_in = align_columns(X_reg_in, reg_feature_cols)

    wait_log_pred = wait_reg.predict(X_reg_in)
    wait_pred = np.expm1(np.asarray(wait_log_pred, dtype=float))
    wait_pred = np.clip(wait_pred, 0, None)

    # 🔒 Safety Clamp (Logic Guard)
    # ถ้า Model มั่นใจว่าเป็น 0 ไม่มาก (<0.85) แต่ Regressor บอกว่าต้องรอนาน (>1ms) -> เชื่อ Regressor (ไม่เป็น 0)
    is_zero_pred = np.where(
        (is_zero_pred == 1) & (proba_is_zero < 0.85) & (wait_pred > 1.0),
        False,
        is_zero_pred.astype(bool),
    )

    # Combine
    ai_pred = np.where(is_zero_pred, 0.0, wait_pred)
    ai_pred = np.clip(ai_pred, 0.0, None)

    # Final Adjustment (0.0 -> 0.1ms)
    eps = 1e-12
    final_pred = np.where(np.abs(ai_pred) <= eps, 0.1, ai_pred)
    final_pred = np.maximum(final_pred, 0.1)

    if min_output_ms is not None:
        final_pred = np.maximum(final_pred, float(min_output_ms))

    
    # -----------------------------------------------------
    # Prepare Output
    # -----------------------------------------------------
    out = df_feat.copy()
    if wave_id_backup is not None:
        out["wave_id"] = wave_id_backup

    out["pred_wait_time_ms"] = final_pred
    flags_df = settle_need_more_sample_flags(df_feat, final_pred, window_ms=10.0)
    # เอา flag แปะเข้าไปใน output dataframe
    out["need_more_sample"] = flags_df["need_more_sample"]
    out["need_more_reason"] = flags_df["need_more_reason"]
    out["need_more_score"]  = flags_df["need_more_score"]

    

    # Reorder columns: wave_id, pred, flags, ... rest
    first_cols = ["wave_id", "pred_wait_time_ms", "need_more_sample", "need_more_reason", "need_more_score"]
    valid_first_cols = [c for c in first_cols if c in out.columns]
    other_cols = [c for c in out.columns if c not in valid_first_cols]
    
    out = out[valid_first_cols + other_cols]

    _ensure_dir(os.path.dirname(out_csv) or ".")
    out.to_csv(out_csv, index=False)
    print(f"✅ Saved: {out_csv} | rows={len(out)}")
    
    # Print summary of flags
    flagged_count = out["need_more_sample"].sum()
    if flagged_count > 0:
        print(f"⚠️  WARNING: {flagged_count} records flagged as 'need_more_sample' (unstable tail).")


def main():
    ap = argparse.ArgumentParser("Predict AutoGluon v22 (inference only)")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--in", dest="input_csv", required=True)
    ap.add_argument("--out", default="data/processed/prediction/predicted_wait_time.csv")
    ap.add_argument("--min-output-ms", type=float, default=None, help="e.g. 0.1 if you never want 0.0")

    args = ap.parse_args()
    predict(
        model_path=args.model_path,
        input_csv=args.input_csv,
        out_csv=args.out,
        min_output_ms=args.min_output_ms,
    )

if __name__ == "__main__":
    main()