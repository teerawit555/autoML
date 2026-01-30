from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

FAST_MS = 0.1
COLS_TO_DROP = ["force_mA", "range_V", "temp_C"]
#DROP_ALWAYS = ["type"]
LABEL_LEAK_COLS = ["wait_time_ms", "wait_time_log", "is_fast", "is_zero"]


# =========================
# IO helpers
# =========================
def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def load_json(path: str, default: Any = None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def align_columns(df, required_cols):
    out = df.copy()
    missing = [c for c in required_cols if c not in out.columns]
    for c in missing:
        out[c] = 0.0
    if missing:
        print(f"[WARN] missing {len(missing)} cols filled with 0.0 (sample): {missing[:8]}")
    return out[required_cols]

# def extract_proba_class1(p) -> np.ndarray:
#     if isinstance(p, pd.DataFrame):
#         return p[1].to_numpy(float)
#     arr = np.asarray(p, float)
#     return arr[:, 1]


# =========================
# Sanity gate (same as train v27)
# =========================
def extract_proba_class1(p) -> np.ndarray:
    if hasattr(p, "columns"):
        cols = list(p.columns)
        if 1 in cols: return p[1].to_numpy(float)
        if "1" in cols: return p["1"].to_numpy(float)
        return p[cols[-1]].to_numpy(float)
    arr = np.asarray(p, float)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    return arr.astype(float)


def apply_dual_threshold_with_sanity(
    *,
    proba_is_fast: np.ndarray,
    wait_pred: np.ndarray,
    X_feat: pd.DataFrame,
    thr_high: float,
    thr_low: float,
    sanity: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match train v27 policy (3 zones):

      STRICT_CORE: p >= thr_high + strict_margin    -> FAST (no sanity)
      STRICT_EDGE: thr_high <= p < thr_high+margin  -> FAST only if sanity_ok
      SOFT:        thr_low  <= p < thr_high         -> FAST only if sanity_ok
      else SLOW

    fast_zone output:
      FAST_CORE / FAST_EDGE / FAST_SOFT / SOFT_BLOCKED / EDGE_BLOCKED / SLOW
    """
    p = np.asarray(proba_is_fast, float)
    wait_pred = np.asarray(wait_pred, float)

    strict_margin = float(sanity.get("strict_margin", 0.0))
    strict_margin = float(np.clip(strict_margin, 0.0, 0.5))

    is_core = (p >= (thr_high + strict_margin))
    is_edge = (p >= thr_high) & (p < (thr_high + strict_margin))
    is_soft = (p >= thr_low) & (p < thr_high)

    need_gate = (is_edge | is_soft)

    # default = ผ่าน
    ok = np.ones_like(p, dtype=bool)

    # --- feature gates (derived) ---
    step_max = sanity.get("step_max", None)
    if step_max is not None and "meta_step_to_span" in X_feat.columns:
        ok &= (X_feat["meta_step_to_span"].to_numpy(float) <= float(step_max))

    ac_max = sanity.get("ac_max", None)
    if ac_max is not None and "per_tail_ac_best" in X_feat.columns:
        ok &= (X_feat["per_tail_ac_best"].to_numpy(float) <= float(ac_max))

    tail_std_max = sanity.get("tail_std_max", None)
    if tail_std_max is not None and "tail_std" in X_feat.columns:
        ok &= (X_feat["tail_std"].to_numpy(float) <= float(tail_std_max))

    late_activity_max = sanity.get("late_activity_max", None)
    if late_activity_max is not None and "late_activity" in X_feat.columns:
        ok &= (X_feat["late_activity"].to_numpy(float) <= float(late_activity_max))

    # reg firewall (edge + soft)
    reg_fp_ms = sanity.get("reg_fp_ms", None)
    if reg_fp_ms is not None:
        ok &= (wait_pred <= float(reg_fp_ms))

    is_fast = is_core | (need_gate & ok)

    # zone labels
    fast_zone = np.where(
        is_core & is_fast, "FAST_CORE",
        np.where(
            is_edge,
            np.where(ok & is_fast, "FAST_EDGE", "EDGE_BLOCKED"),
            np.where(
                is_soft,
                np.where(ok & is_fast, "FAST_SOFT", "SOFT_BLOCKED"),
                "SLOW",
            )
        )
    )

    return is_fast, fast_zone





# =========================
# Prediction
# =========================
def predict(
    model_path: str,
    input_csv: str,
    out_csv: str,
    *,
    debug_topk: int = 20,
) -> None:
    zero_model_path = os.path.join(model_path, "zero_clf")
    reg_model_path  = os.path.join(model_path, "wait_reg")

    zero_feature_cols = load_json(os.path.join(model_path, "zero_feature_cols.json"))
    reg_feature_cols  = load_json(os.path.join(model_path, "reg_feature_cols.json"))
    thresholds        = load_json(os.path.join(model_path, "fast_thresholds.json"))
    sanity            = load_json(os.path.join(model_path, "sanity_gate.json"))

    if not zero_feature_cols or not reg_feature_cols:
        raise RuntimeError("missing feature cols json")

    thr_high = float(thresholds["thr_high"])
    thr_low  = float(thresholds.get("thr_low", thr_high))

    print(f"FAST thresholds: high={thr_high:.6f}, low={thr_low:.6f}")
    print(f"SANITY gate: {sanity}")

    zero_clf = TabularPredictor.load(zero_model_path)
    wait_reg = TabularPredictor.load(reg_model_path)

    df = pd.read_csv(input_csv)
    wave_id = df["wave_id"] if "wave_id" in df.columns else pd.Series(np.arange(len(df)))

    # ✅ keep type for debug (not used by model)
    type_series = df["type"].astype(str) if "type" in df.columns else pd.Series([""] * len(df))

    # drop leaks + meta (แต่ไม่ต้อง drop type ตรงนี้ เพราะเราแยกเก็บแล้ว)
    df_feat = df.drop(
        columns=COLS_TO_DROP + LABEL_LEAK_COLS + ["type"],   # ✅ ใส่ "type" ตรงนี้แทน
        errors="ignore",
    ).copy()


    # -------------------------
    # Stage A: classifier
    # -------------------------
    X_zero = align_columns(
        df_feat.drop(columns=["wave_id"], errors="ignore"),
        zero_feature_cols,
    )
    proba_is_fast = extract_proba_class1(zero_clf.predict_proba(X_zero))

    # -------------------------
    # Stage B: reg (ALL rows)
    # -------------------------
    X_reg = align_columns(
        df_feat.drop(columns=["wave_id"], errors="ignore"),
        reg_feature_cols,
    )

    X_sanity = df_feat.drop(columns=["wave_id"], errors="ignore")

    wait_log = wait_reg.predict(X_reg)
    wait_pred = np.expm1(np.asarray(wait_log, float))
    wait_pred = np.clip(wait_pred, 0.0, None)

    # -------------------------
    # Final decision (v27)
    # -------------------------
    is_fast, fast_zone = apply_dual_threshold_with_sanity(
        proba_is_fast=proba_is_fast,
        wait_pred=wait_pred,
        X_feat=X_sanity,
        #df_feats=X_zero,    # หรือ df_feats=df_feat.drop(columns=["wave_id"], errors="ignore")
        thr_high=thr_high,
        thr_low=thr_low,
        sanity=sanity,
    )

    pred_ms = np.where(is_fast, FAST_MS, wait_pred)

    out = pd.DataFrame(
        {
            "wave_id": wave_id,
            "type_debug": type_series, 
            "fast_zone": fast_zone,
            "pred_is_fast": is_fast.astype(int),
            "pred_wait_time_ms": pred_ms,
            "proba_is_fast": proba_is_fast,
            "reg_wait_pred_ms": wait_pred,
        }
    )

    # -------------------------
    # Debug
    # -------------------------
    if debug_topk > 0:
        # (A) borderline FAST_STRICT: proba ต่ำสุดในกลุ่มที่ pred_is_fast=1
        fast_only = out[out["pred_is_fast"] == 1].copy()
        if len(fast_only) > 0:
            borderline_fast = fast_only.sort_values("proba_is_fast").head(debug_topk)
            print(f"\n🧪 Top-{debug_topk} borderline FAST (lowest proba among FAST):")
            print(borderline_fast.to_string(index=False))

        # (B) SOFT_BLOCKED: ตัวที่อยู่ใน soft zone แต่ไม่ผ่าน sanity
        blocked = out[out["fast_zone"] == "SOFT_BLOCKED"].copy()
        if len(blocked) > 0:
            print("\n🚫 SOFT_BLOCKED rows (inspect why failed sanity):")
            show_cols = ["wave_id","fast_zone","proba_is_fast","reg_wait_pred_ms","pred_wait_time_ms"]
            # ถ้ามี feature sanity ก็แปะให้ดู
            extra_cols = []
            for c in ["meta_step_to_span","per_tail_ac_best","tail_std","late_activity"]:
                if c in X_sanity.columns:
                    blocked[c] = X_sanity.loc[blocked.index, c].to_numpy(float)
                    extra_cols.append(c)

            show_cols = ["wave_id","fast_zone","proba_is_fast"] + extra_cols + ["reg_wait_pred_ms","pred_wait_time_ms"]

            print(blocked[show_cols].sort_values("proba_is_fast").to_string(index=False))

        print("\n📊 fast_zone distribution:")
        print(out["fast_zone"].value_counts().to_string())

        print("\n📊 type distribution among predicted fast:")
        print(out[out["pred_is_fast"] == 1]["type_debug"].value_counts().to_string())

        print("\n📊 zone distribution among predicted fast:")
        print(out[out["pred_is_fast"] == 1]["fast_zone"].value_counts().to_string())



    _ensure_dir(os.path.dirname(out_csv) or ".")
    out.to_csv(out_csv, index=False)
    print(f"\n✅ Saved {out_csv} | rows={len(out)}")


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser("predict_ag_v27")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--in", dest="input_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug-topk", type=int, default=20)
    args = ap.parse_args()

    predict(
        model_path=args.model_path,
        input_csv=args.input_csv,
        out_csv=args.out,
        debug_topk=args.debug_topk,
    )


if __name__ == "__main__":
    main()
