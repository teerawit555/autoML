# from __future__ import annotations

# import argparse
# import json
# import os
# from typing import Any

# import numpy as np
# import pandas as pd
# from autogluon.tabular import TabularPredictor

# # =========================
# # Config
# # =========================
# COLS_TO_DROP = ["force_mA", "range_V", "temp_C"]
# DROP_ALWAYS = ["type"]

# FAST_MS = 0.1  # policy: fast output


# # =========================
# # 1) Heuristic Flag Logic (optional debug / QA)
# # =========================
# def settle_need_more_sample_flags(
#     df_feat: pd.DataFrame,
#     pred_ms: np.ndarray,
#     window_ms: float = 10.0,
# ) -> pd.DataFrame:
#     """
#     Flag cases where prediction is near window end AND tail seems unstable,
#     excluding periodic-ish patterns.

#     This function is SAFE even if some columns do not exist.
#     """
#     def g(col: str, default: float = 0.0) -> np.ndarray:
#         if col in df_feat.columns:
#             return df_feat[col].to_numpy(dtype=float)
#         return np.full(len(df_feat), default, dtype=float)

#     pred_ms = np.asarray(pred_ms, dtype=float)

#     tail_creep_norm       = g("tail_creep_norm", 0.0)
#     tail100_slope_norm    = g("tail100_slope_norm", 0.0)
#     std_tail_50           = g("std_tail_50", 0.0)
#     last_cross_tail_ratio = g("last_cross_tail_ratio", 1.0)
#     envelope_ratio        = g("envelope_ratio", 0.0)

#     # periodic-ish detectors (may not exist in inference -> safe fallback)
#     logic_per_score     = g("logic_per_score", 0.0)
#     logic_crossing_rate = g("logic_crossing_rate", 0.0)
#     ring_peak_count     = g("ring_peak_count", 0.0)

#     gate = (pred_ms >= 0.70 * window_ms) | (pred_ms >= (window_ms - 0.25))

#     is_periodicish = (
#         (logic_per_score > 0.45)
#         | (logic_crossing_rate > 0.20)
#         | (ring_peak_count >= 6)
#     )

#     TH_CREEP      = 0.015
#     TH_SLOPE      = 0.006
#     TH_STD_TAIL   = 0.0030
#     TH_LAST_CROSS = 0.035
#     TH_ENV        = 3.0

#     c1 = (np.abs(tail_creep_norm) > TH_CREEP)
#     c2 = (np.abs(tail100_slope_norm) > TH_SLOPE)
#     c3 = (std_tail_50 > TH_STD_TAIL)
#     c4 = (last_cross_tail_ratio < TH_LAST_CROSS)
#     c5 = (envelope_ratio > TH_ENV)

#     score = c1.astype(float) + c2.astype(float) + c3.astype(float) + c4.astype(float) + c5.astype(float)
#     need = gate & (~is_periodicish) & (score >= 2.0)

#     reasons: list[str] = []
#     for i in range(len(df_feat)):
#         if not need[i]:
#             reasons.append("")
#             continue
#         r = []
#         if c1[i]: r.append("tail_creep")
#         if c2[i]: r.append("tail_slope")
#         if c3[i]: r.append("tail_std_high")
#         if c4[i]: r.append("late_crossing")
#         if c5[i]: r.append("mid_env_high")
#         reasons.append("|".join(r))

#     return pd.DataFrame({
#         "need_more_sample": need.astype(int),
#         "need_more_reason": reasons,
#         "need_more_score": score.astype(float),
#     })


# # =========================
# # 2) IO / helpers
# # =========================
# def _ensure_dir(path: str) -> None:
#     if path:
#         os.makedirs(path, exist_ok=True)

# def load_json(path: str, default: Any = None):
#     if not os.path.exists(path):
#         return default
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

# def load_text(path: str, default: str = "") -> str:
#     if not os.path.exists(path):
#         return default
#     with open(path, "r", encoding="utf-8") as f:
#         return f.read().strip()

# def align_columns(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
#     out = df.copy()
#     for c in required_cols:
#         if c not in out.columns:
#             out[c] = 0.0
#     return out[required_cols]

# def extract_proba_class1(p) -> np.ndarray:
#     """
#     Robustly extract probability of class=1 from AutoGluon predict_proba output.
#     Prints columns in caller (recommended).
#     """
#     if isinstance(p, pd.DataFrame):
#         cols = list(p.columns)
#         if 1 in cols:
#             return p[1].to_numpy(dtype=float)
#         # sometimes columns are strings
#         cols_str = [str(c) for c in cols]
#         if "1" in cols_str:
#             return p.loc[:, [cols[cols_str.index("1")]]].iloc[:, 0].to_numpy(dtype=float)
#         # fallback: last column
#         return p.iloc[:, -1].to_numpy(dtype=float)

#     arr = np.asarray(p, dtype=float)
#     if arr.ndim == 2 and arr.shape[1] >= 2:
#         return arr[:, 1]
#     return arr.astype(float)


# # =========================
# # 3) Prediction
# # =========================
# def predict(
#     model_path: str,
#     input_csv: str,
#     out_csv: str,
#     *,
#     min_output_ms: float | None,
#     no_safety_clamp: bool,
# ) -> None:
#     fast_model_path = os.path.join(model_path, "zero_clf")
#     reg_model_path  = os.path.join(model_path, "wait_reg")

#     if not os.path.exists(fast_model_path):
#         raise FileNotFoundError(f"Missing classifier model folder: {fast_model_path}")
#     if not os.path.exists(reg_model_path):
#         raise FileNotFoundError(f"Missing regressor model folder: {reg_model_path}")

#     fast_feature_cols = load_json(os.path.join(model_path, "zero_feature_cols.json"), default=None)
#     reg_feature_cols  = load_json(os.path.join(model_path, "reg_feature_cols.json"), default=None)

#     if not fast_feature_cols or not isinstance(fast_feature_cols, list):
#         raise FileNotFoundError("zero_feature_cols.json missing or invalid.")
#     if not reg_feature_cols or not isinstance(reg_feature_cols, list):
#         raise FileNotFoundError("reg_feature_cols.json missing or invalid.")

#     fast_thr = float(load_text(os.path.join(model_path, "zero_threshold.txt"), default="0.5"))

#     print(f"🔮 Predicting: {input_csv}")
#     print(f"   Model: {model_path}")
#     print(f"   Using FAST threshold = {fast_thr:.3f} | FAST_MS={FAST_MS}")
#     if no_safety_clamp:
#         print("   Safety clamp: DISABLED (--no-safety-clamp)")

#     fast_clf = TabularPredictor.load(fast_model_path)
#     wait_reg = TabularPredictor.load(reg_model_path)

#     df = pd.read_csv(input_csv)
#     df = df.drop(columns=DROP_ALWAYS, errors="ignore")

#     wave_id_backup = df["wave_id"].copy() if "wave_id" in df.columns else pd.Series(np.arange(len(df)), name="wave_id")

#     cols_to_drop_found = [c for c in COLS_TO_DROP if c in df.columns]
#     df_feat = df.drop(columns=cols_to_drop_found, errors="ignore").copy()

#     # prevent leaked columns if exist
#     df_feat = df_feat.drop(columns=["wait_time_ms", "wait_time_log", "is_fast", "is_zero"], errors="ignore")

#     # -------------------------
#     # Stage A: FAST classifier
#     # -------------------------
#     X_fast_in = df_feat.drop(columns=["wave_id"], errors="ignore").copy()
#     X_fast_in = align_columns(X_fast_in, fast_feature_cols)

#     p = fast_clf.predict_proba(X_fast_in)
#     print("predict_proba columns =", getattr(p, "columns", None))

#     proba_is_fast = extract_proba_class1(p)
#     is_fast_raw = (proba_is_fast >= fast_thr)

#     print(
#         f"DEBUG fast(before clamp) = {int(is_fast_raw.sum())}/{len(is_fast_raw)} | "
#         f"proba min/mean/max = {proba_is_fast.min():.4f} {proba_is_fast.mean():.4f} {proba_is_fast.max():.4f}"
#     )
#     print("DEBUG top10 proba =", np.sort(proba_is_fast)[-10:])

#     # sanity: if max > thr but count==0 => logic bug (should never happen)
#     if (proba_is_fast.max() > fast_thr + 1e-9) and (int(is_fast_raw.sum()) == 0):
#         print("❗SANITY FAIL: max(proba) > thr but predicted_fast=0. Check class mapping / threshold loading.")

#     # -------------------------
#     # Stage B: Regressor
#     # -------------------------
#     X_reg_in = df_feat.drop(columns=["wave_id"], errors="ignore").copy()
#     X_reg_in = align_columns(X_reg_in, reg_feature_cols)

#     wait_log_pred = wait_reg.predict(X_reg_in)
#     wait_pred = np.expm1(np.asarray(wait_log_pred, dtype=float))
#     wait_pred = np.clip(wait_pred, 0.0, None)

#     # -------------------------
#     # Safety clamp (optional)
#     # -------------------------
#     #if no_safety_clamp:
#     is_fast_final = is_fast_raw
#     # else:
#     #     # clamp only suspicious fast
#     #     is_fast_final = np.where(
#     #         (is_fast_raw == 1) & (proba_is_fast < 0.85) & (wait_pred > 1.0),
#     #         False,
#     #         is_fast_raw.astype(bool),
#     #     )

#     print(f"DEBUG fast(after clamp)  = {int(np.sum(is_fast_final))}/{len(is_fast_final)}")

#     # -------------------------
#     # Combine outputs
#     # -------------------------
#     pred_ms = np.where(is_fast_final, FAST_MS, wait_pred)
#     pred_ms = np.clip(pred_ms, 0.0, None)

#     floor_ms = FAST_MS if (min_output_ms is None) else float(min_output_ms)
#     pred_ms = np.maximum(pred_ms, floor_ms)

#     # flags (optional)
#     flags_df = settle_need_more_sample_flags(df_feat, pred_ms, window_ms=10.0)

#     out = pd.DataFrame({
#         "wave_id": wave_id_backup,
#         "pred_wait_time_ms": pred_ms,
#         "proba_is_fast": proba_is_fast,
#         "pred_is_fast": is_fast_final.astype(int),
#         "need_more_sample": flags_df["need_more_sample"],
#         "need_more_reason": flags_df["need_more_reason"],
#         "need_more_score": flags_df["need_more_score"],
#     })

#     _ensure_dir(os.path.dirname(out_csv) or ".")
#     out.to_csv(out_csv, index=False)

#     print(f"✅ Saved: {out_csv} | rows={len(out)}")
#     print(f"thr={fast_thr:.3f} predicted_fast={int(out['pred_is_fast'].sum())}/{len(out)}")


# def main():
#     ap = argparse.ArgumentParser("Predict AutoGluon v23 (fast-clf + slow-reg) with strong debug")
#     ap.add_argument("--model-path", required=True)
#     ap.add_argument("--in", dest="input_csv", required=True)
#     ap.add_argument("--out", required=True)
#     ap.add_argument("--min-output-ms", type=float, default=None, help="floor output, e.g. 0.1")
#     ap.add_argument("--no-safety-clamp", action="store_true", help="disable any post-override logic")

#     args = ap.parse_args()
#     predict(
#         model_path=args.model_path,
#         input_csv=args.input_csv,
#         out_csv=args.out,
#         min_output_ms=args.min_output_ms,
#         no_safety_clamp=args.no_safety_clamp,
#     )

# if __name__ == "__main__":
#     main()


# ## guard version -- single threshold
# from __future__ import annotations

# import argparse
# import json
# import os
# from typing import Any

# import numpy as np
# import pandas as pd
# from autogluon.tabular import TabularPredictor

# # =========================
# # Config
# # =========================
# COLS_TO_DROP = ["force_mA", "range_V", "temp_C"]
# DROP_ALWAYS = ["type"]
# LABEL_LEAK_COLS = ["wait_time_ms", "wait_time_log", "is_fast", "is_zero"]

# FAST_MS = 0.1


# # =========================
# # Helpers
# # =========================
# def _ensure_dir(path: str) -> None:
#     if path:
#         os.makedirs(path, exist_ok=True)

# def load_json(path: str, default: Any = None):
#     if not os.path.exists(path):
#         return default
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

# def load_text(path: str, default: str = "") -> str:
#     if not os.path.exists(path):
#         return default
#     with open(path, "r", encoding="utf-8") as f:
#         return f.read().strip()

# def align_columns(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
#     out = df.copy()
#     for c in required_cols:
#         if c not in out.columns:
#             out[c] = 0.0
#     return out[required_cols]

# def extract_proba_class1(p) -> np.ndarray:
#     if isinstance(p, pd.DataFrame):
#         if 1 in p.columns:
#             return p[1].to_numpy(float)
#         return p.iloc[:, -1].to_numpy(float)
#     arr = np.asarray(p, float)
#     return arr[:, 1] if arr.ndim == 2 else arr

# def load_type_map(meta_csv: str) -> dict[int, str]:
#     m = pd.read_csv(meta_csv)
#     out = {}
#     for _, r in m[["wave_id", "type"]].dropna().iterrows():
#         try:
#             out[int(r["wave_id"])] = str(r["type"])
#         except Exception:
#             pass
#     return out


# # =========================
# # Rule (log-only)
# # =========================
# def rule_steady_sine(df_feat: pd.DataFrame) -> np.ndarray:
#     need = [
#         "tail_ac_best",
#         "fft_spectral_entropy",
#         "fft_peak_to_2nd_ratio",
#         "tail_amp_cv",
#         "tail_env_slope_abs",
#         "edge_count",
#     ]
#     if any(c not in df_feat.columns for c in need):
#         return np.zeros(len(df_feat), dtype=bool)

#     return (
#         (df_feat["tail_ac_best"] >= 0.6) &
#         (df_feat["fft_spectral_entropy"] <= 0.6) &
#         (df_feat["fft_peak_to_2nd_ratio"] >= 1.6) &
#         (df_feat["tail_amp_cv"] <= 0.08) &
#         (df_feat["tail_env_slope_abs"] <= 0.05) &
#         (df_feat["edge_count"] <= 8)
#     ).to_numpy(bool)

# import numpy as np
# import pandas as pd


# # =========================
# # Prediction
# # =========================
# def predict(
#     model_path: str,
#     input_csv: str,
#     out_csv: str,
#     *,
#     meta_csv: str | None,
#     debug_topk: int,
# ) -> None:
#     fast_model_path = os.path.join(model_path, "zero_clf")
#     reg_model_path  = os.path.join(model_path, "wait_reg")

#     fast_feature_cols = load_json(os.path.join(model_path, "zero_feature_cols.json"))
#     reg_feature_cols  = load_json(os.path.join(model_path, "reg_feature_cols.json"))
#     fast_thr = float(load_text(os.path.join(model_path, "zero_threshold.txt"), "0.5"))

#     print(f"🔮 Predicting {input_csv}")
#     print(f"FAST threshold = {fast_thr:.3f} | FAST_MS={FAST_MS}")

#     fast_clf = TabularPredictor.load(fast_model_path)
#     wait_reg = TabularPredictor.load(reg_model_path)

#     df = pd.read_csv(input_csv)
#     wave_id = df["wave_id"] if "wave_id" in df.columns else pd.Series(np.arange(len(df)))

#     # type debug
#     if "type" in df.columns:
#         type_series = df["type"].astype(str)
#     elif meta_csv:
#         type_map = load_type_map(meta_csv)
#         type_series = wave_id.apply(lambda x: type_map.get(int(x), ""))
#     else:
#         type_series = pd.Series([""] * len(df))

#     df = df.drop(columns=DROP_ALWAYS + COLS_TO_DROP + LABEL_LEAK_COLS, errors="ignore")
#     df_feat = df.copy()

#     # ---------- Stage A ----------
#     X_fast = align_columns(df_feat.drop(columns=["wave_id"], errors="ignore"), fast_feature_cols)

#     # ===== DEBUG: X_fast health =====
#     nuniq = X_fast.nunique(dropna=False)
#     dead = nuniq[nuniq <= 1]
#     print(f"[DEBUG] X_fast shape = {X_fast.shape}")
#     print(f"[DEBUG] constant features = {len(dead)}")
#     if len(dead) > 0:
#         print("[DEBUG] sample constant cols:", dead.index.tolist()[:30])

#     # optional: ดู top variance
#     print("[DEBUG] top-10 std:", X_fast.std(numeric_only=True).sort_values(ascending=False).head(10).to_string())
#     print("[DEBUG] bottom-10 std:", X_fast.std(numeric_only=True).sort_values(ascending=True).head(10).to_string())


#     proba_is_fast = extract_proba_class1(fast_clf.predict_proba(X_fast))
#     is_fast = proba_is_fast >= fast_thr

#     # raw decision จากโมเดล
#     is_fast_raw = (proba_is_fast >= fast_thr)

#     # gate จาก rule
#     #allowed = fast_allowed_rule(df_feat)

#     # final decision (deploy-safe)
#     #is_fast_final = is_fast_raw & allowed

#     # ===== DEBUG: proba distribution =====
#     uniq = np.unique(np.round(proba_is_fast, 6))
#     print(f"[DEBUG] proba unique count = {len(uniq)}")
#     print(f"[DEBUG] proba min/mean/max = {proba_is_fast.min():.6f} / {proba_is_fast.mean():.6f} / {proba_is_fast.max():.6f}")
#     print("[DEBUG] proba unique sample:", uniq[:20])

#     # ---------- Stage B (slow only) ----------
#     wait_pred = np.full(len(df_feat), np.nan, dtype=float)

#     slow_mask = ~is_fast_raw
#     if slow_mask.any():
#         X_reg = align_columns(
#             df_feat.loc[slow_mask].drop(columns=["wave_id"], errors="ignore"),
#             reg_feature_cols,
#         )
#         wait_log = wait_reg.predict(X_reg)
#         wait_pred[slow_mask] = np.expm1(wait_log)

#     # ---------- Combine ----------
#     pred_ms = np.where(is_fast, FAST_MS, wait_pred)
#     pred_ms = np.clip(pred_ms, 0.0, None)

#     #rule_is_steady = rule_steady_sine(df_feat)

#     out = pd.DataFrame({
#         "wave_id": wave_id,
#         "type_debug": type_series,
#         "pred_wait_time_ms": pred_ms,
#         "pred_is_fast": is_fast.astype(int),
#         "proba_is_fast": proba_is_fast,
#         "reg_wait_pred_ms": wait_pred,
#         #"rule_is_steady": rule_is_steady.astype(int),
#     })


#     # ---------- Debug ----------
#     if debug_topk > 0:
#         suspicious = out[out["pred_is_fast"] == 1].copy()
#         suspicious = suspicious.sort_values("proba_is_fast").head(debug_topk)

#         print(f"\n🧪 Top-{debug_topk} borderline FAST (low confidence):")
#         print(
#             suspicious[
#                 ["wave_id","type_debug","proba_is_fast","pred_wait_time_ms"]
#             ].to_string(index=False)
#         )

#         print("\n📊 type distribution among predicted fast:")
#         print(out[out["pred_is_fast"] == 1]["type_debug"].value_counts().to_string())

#     _ensure_dir(os.path.dirname(out_csv) or ".")
#     out.to_csv(out_csv, index=False)
#     print(f"\n✅ Saved {out_csv} | rows={len(out)}")


# # =========================
# # CLI
# # =========================
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model-path", required=True)
#     ap.add_argument("--in", dest="input_csv", required=True)
#     ap.add_argument("--out", required=True)
#     ap.add_argument("--meta-csv", default=None)
#     ap.add_argument("--debug-topk", type=int, default=20)
#     args = ap.parse_args()

#     predict(
#         model_path=args.model_path,
#         input_csv=args.input_csv,
#         out_csv=args.out,
#         meta_csv=args.meta_csv,
#         debug_topk=args.debug_topk,
#     )

# if __name__ == "__main__":
#     main()

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

# =========================
# Config
# =========================
COLS_TO_DROP = ["force_mA", "range_V", "temp_C"]
DROP_ALWAYS = ["type"]
LABEL_LEAK_COLS = ["wait_time_ms", "wait_time_log", "is_fast", "is_zero"]

FAST_MS = 0.1


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


def load_type_map(meta_csv: str) -> dict[int, str]:
    m = pd.read_csv(meta_csv)
    out: dict[int, str] = {}
    for _, r in m[["wave_id", "type"]].dropna().iterrows():
        try:
            out[int(r["wave_id"])] = str(r["type"])
        except Exception:
            pass
    return out

def load_fast_thresholds(model_path: str, default_thr: float):
    p = os.path.join(model_path, "fast_thresholds.json")
    if os.path.exists(p):
        jj = load_json(p)
        thr_high = jj["thr_high"]
        thr_low  = jj.get("thr_low", thr_high)
        return thr_high, thr_low
    else:
        return default_thr, default_thr


# def load_fast_thresholds(model_path: str, default_thr: float) -> tuple[float, float]:
#     """
#     Return (thr_high, thr_low)
#     - If fast_thresholds.json exists -> use it
#     - else fallback -> use zero_threshold.txt as single threshold for both
#     """
#     p = os.path.join(model_path, "fast_thresholds.json")
#     if os.path.exists(p):
#         jj = load_json(p, default={}) or {}
#         thr_high = float(jj.get("thr_high", default_thr))
#         thr_low = float(jj.get("thr_low", thr_high))
#         thr_low = float(min(thr_low, thr_high))
#         print("[INFO] threshold src=fast_thresholds.json")
#         return thr_high, thr_low
#     else:
#         print("[INFO] threshold src=zero_threshold.txt (fallback)")
#     return default_thr, default_thr


# =========================
# Rule (log-only)
# =========================
def rule_steady_sine(df_feat: pd.DataFrame) -> np.ndarray:
    need = [
        "tail_ac_best",
        "fft_spectral_entropy",
        "fft_peak_to_2nd_ratio",
        "tail_amp_cv",
        "tail_env_slope_abs",
        "edge_count",
    ]
    if any(c not in df_feat.columns for c in need):
        return np.zeros(len(df_feat), dtype=bool)

    return (
        (df_feat["tail_ac_best"] >= 0.6)
        & (df_feat["fft_spectral_entropy"] <= 0.65)
        & (df_feat["fft_peak_to_2nd_ratio"] >= 1.6)
        & (df_feat["tail_amp_cv"] <= 0.08)
        & (df_feat["tail_env_slope_abs"] <= 0.05)
        & (df_feat["edge_count"] <= 8)
    ).to_numpy(bool)

## Try
# def rule_inband_from_start(df_feat: pd.DataFrame) -> np.ndarray:
#     if "t_enter_band_ms" not in df_feat.columns:
#         return np.zeros(len(df_feat), dtype=bool)

#     t = pd.to_numeric(df_feat["t_enter_band_ms"], errors="coerce").fillna(1e9).to_numpy(float)
#     # ✅ ถ้าเข้ากรอบตั้งแต่ <=0.1ms ให้ถือว่า fast เลย
#     return (t <= 0.10)

# def soft_fast_gate(df_feat: pd.DataFrame) -> np.ndarray:
#     allow = np.zeros(len(df_feat), dtype=bool)

#     # (A) อยู่ใน band ตั้งแต่ต้น → allow (ครอบคลุม sine-fast)
#     allow |= rule_inband_from_start(df_feat)

#     # (B) ของเดิม (pulse/glitch-like) จะคงไว้ก็ได้
#     if ("edge_count" in df_feat.columns) and ("last_edge_pos_ratio" in df_feat.columns):
#         ec = pd.to_numeric(df_feat["edge_count"], errors="coerce").fillna(0).to_numpy(float)
#         ler = pd.to_numeric(df_feat["last_edge_pos_ratio"], errors="coerce").fillna(0).to_numpy(float)
#         pulse_like = (ec >= 3) & (ec <= 30) & (ler >= 0.60)
#         allow |= pulse_like

#     return allow

import numpy as np
import pandas as pd

def soft_fast_gate(df_feat: pd.DataFrame) -> np.ndarray:
    """
    SOFT gate for mos-features (no edge_count/fft/tail_ac_*).
    Goal: allow FAST_SOFT เฉพาะ waveform ที่ 'นิ่งเร็ว/เข้า band เร็ว/สัญญาณท้ายเงียบ'
    Conservative: ถ้าขาด feature หลัก -> return False ทั้งหมด (กัน FP)
    """
    n = len(df_feat)
    allow = np.zeros(n, dtype=bool)

    # ---- required-ish columns for MOS gate ----
    must_have = ["t_enter_norm", "tail_noise_ratio", "slope_norm"]
    if any(c not in df_feat.columns for c in must_have):
        return allow  # conservative: no signal -> don't allow soft-fast

    # helper to numeric arrays
    def num(col: str, default: float) -> np.ndarray:
        return pd.to_numeric(df_feat.get(col, default), errors="coerce").fillna(default).to_numpy(float)

    t_enter_norm     = num("t_enter_norm", 1.0)          # 0..1 (เข้า band เร็ว -> เล็ก)
    enter_margin_ms  = num("enter_margin_ms", 0.0)       # ยิ่งมาก = เข้า band เร็ว (เพราะ t_end - t_enter)
    tail_noise_ratio = num("tail_noise_ratio", 999.0)    # tail_std/std_value (ยิ่งต่ำ = ท้ายเงียบ)
    slope_norm       = num("slope_norm", 999.0)          # max_slope/p2p (ยิ่งต่ำ = ไม่กระชาก)
    enter_vs_settle  = num("enter_vs_settle", 999.0)     # t_enter / t_est (ใกล้ 1 ดี)
    t_est_settle_ms  = num("t_est_settle_ms", 999.0)
    tail_mas         = num("tail_mean_abs_slope", 999.0)
    band_to_p2p      = num("band_to_p2p", 0.0)

    # ---- (A) "entered band early" signal ----
    # ถ้า t_enter_norm ต่ำ แปลว่าเข้า band เร็วมาก
    early_enter = (t_enter_norm <= 0.20) | (enter_margin_ms >= 2.0)  # 2ms ปรับได้ตาม window

    # ---- (B) "tail is quiet" signal ----
    quiet_tail = (tail_noise_ratio <= 0.35) & (tail_mas <= 2000.0)   # tail_mas scale-dependent -> ปรับได้

    # ---- (C) "not too impulsive" signal ----
    # slope_norm สูงมากมักเป็น step/edge-heavy ที่เสี่ยง FP
    not_impulsive = (slope_norm <= 0.35)

    # ---- (D) "enter & settle consistent" (optional but helpful) ----
    consistent = (enter_vs_settle <= 1.25) | (t_est_settle_ms <= 0.25)

    # ---- (E) If no band info at all (band_to_p2p ~ 0) -> be stricter ----
    has_band = (band_to_p2p > 0.02)

    # Final allow:
    # - ต้อง early_enter + quiet_tail + not_impulsive + consistent
    # - ถ้าไม่มี band ให้ require t_est_settle_ms สั้นมากเพิ่ม
    allow = early_enter & quiet_tail & not_impulsive & consistent
    allow = allow & (has_band | (t_est_settle_ms <= 0.15))

    # guard NaN weirdness (already filled) — keep as bool
    return allow.astype(bool)

#-------------------
# TT version
#-------------------
# def soft_fast_gate(df_feat: pd.DataFrame) -> np.ndarray:
#     n = len(df_feat)
#     allow = np.zeros(n, dtype=bool)

#     # (A) steady sine / stable periodic tail (ถ้ามี feature ครบ)
#     try:
#         allow |= rule_steady_sine(df_feat)
#     except Exception:
#         pass

#     # (B) pulse/glitch-like
#     if ("edge_count" in df_feat.columns) and ("last_edge_pos_ratio" in df_feat.columns):
#         ec  = pd.to_numeric(df_feat["edge_count"], errors="coerce").fillna(0).to_numpy(float)
#         ler = pd.to_numeric(df_feat["last_edge_pos_ratio"], errors="coerce").fillna(0).to_numpy(float)
#         allow |= (ec >= 3) & (ec <= 30) & (ler >= 0.60)

#     return allow

# def soft_fast_gate(df_feat: pd.DataFrame) -> np.ndarray:
#     """
#     FAST_SOFT allow only for:
#     A) steady sine (rule_steady_sine)
#     B) pulse/glitch-like: edges are not too many (cap) AND last edge is very late
#     """
#     n = len(df_feat)
#     allow = np.zeros(n, dtype=bool)

#     # (A) steady sine
#     try:
#         allow |= rule_steady_sine(df_feat)
#     except Exception:
#         pass

#     # (B) pulse/glitch (FIX: cap edge_count + late last edge)
#     if ("edge_count" in df_feat.columns) and ("last_edge_pos_ratio" in df_feat.columns):
#         ec = pd.to_numeric(df_feat["edge_count"], errors="coerce").fillna(0).to_numpy(float)
#         ler = pd.to_numeric(df_feat["last_edge_pos_ratio"], errors="coerce").fillna(0).to_numpy(float)

#         # ✅ tune here
#         EC_MIN = 3
#         EC_MAX = 30        # กัน Step ที่ edge_count บวมแบบ 189
#         LER_MIN = 0.60     # ต้องมี edge “ท้ายคลื่น” จริง ๆ

#         pulse_like = (ec >= EC_MIN) & (ec <= EC_MAX) & (ler >= LER_MIN)
#         allow |= pulse_like

#     return allow



# =========================
# Prediction
# =========================
def predict(
    model_path: str,
    input_csv: str,
    out_csv: str,
    *,
    meta_csv: str | None,
    debug_topk: int,
) -> None:
    fast_model_path = os.path.join(model_path, "zero_clf")
    reg_model_path = os.path.join(model_path, "wait_reg")

    fast_feature_cols = load_json(os.path.join(model_path, "zero_feature_cols.json"))
    reg_feature_cols = load_json(os.path.join(model_path, "reg_feature_cols.json"))

    if not fast_feature_cols:
        raise FileNotFoundError(f"missing zero_feature_cols.json in: {model_path}")
    if not reg_feature_cols:
        raise FileNotFoundError(f"missing reg_feature_cols.json in: {model_path}")

    # legacy single-thr fallback
    single_thr = float(load_text(os.path.join(model_path, "zero_threshold.txt"), "0.5"))
    thr_high, thr_low = load_fast_thresholds(model_path, single_thr)

    if abs(thr_high - thr_low) < 1e-12:
        thr_low = max(thr_high - 0.005, 0.10)


    print(f"[DBG] thr_high={thr_high:.18f} thr_low={thr_low:.18f} (single_thr={single_thr:.18f})")

    # enforce ordering
    thr_low = float(min(thr_low, thr_high))

    dual_mode = (thr_low < thr_high - 1e-12)

    print(f"🔮 Predicting {input_csv}")
    if dual_mode:
        print(f"FAST thresholds(dual): high={thr_high:.3f} low={thr_low:.3f} | FAST_MS={FAST_MS}")
    else:
        print(f"FAST threshold(single) = {thr_high:.3f} | FAST_MS={FAST_MS}")

    # if thr_high == thr_low:
    #     thr_low = float(single_thr)       # เอา 0.285/0.275 จาก train มาเป็น low
    #     thr_low = min(thr_low, thr_high)  # ensure ordering

    print(f"🔮 Predicting {input_csv}")
    if thr_high == thr_low:
        print(f"FAST threshold(single) = {thr_high:.3f} | FAST_MS={FAST_MS}")
    else:
        print(f"FAST thresholds(dual): high={thr_high:.3f} low={thr_low:.3f} | FAST_MS={FAST_MS}")

    fast_clf = TabularPredictor.load(fast_model_path)
    wait_reg = TabularPredictor.load(reg_model_path)

    df = pd.read_csv(input_csv)
    wave_id = df["wave_id"] if "wave_id" in df.columns else pd.Series(np.arange(len(df)))

    # type debug (not used by model)
    if "type" in df.columns:
        type_series = df["type"].astype(str)
    elif meta_csv:
        type_map = load_type_map(meta_csv)
        type_series = wave_id.apply(lambda x: type_map.get(int(x), ""))
    else:
        type_series = pd.Series([""] * len(df))

    # drop meta + leak cols
    df_feat = df.drop(columns=DROP_ALWAYS + COLS_TO_DROP + LABEL_LEAK_COLS, errors="ignore").copy()

    # ---------- Stage A: fast proba ----------
    X_fast = align_columns(df_feat.drop(columns=["wave_id"], errors="ignore"), fast_feature_cols)

    # ===== DEBUG: X_fast health =====
    try:
        nuniq = X_fast.nunique(dropna=False)
        dead = nuniq[nuniq <= 1]
        print(f"[DEBUG] X_fast shape = {X_fast.shape}")
        print(f"[DEBUG] constant features = {len(dead)}")
        if len(dead) > 0:
            print("[DEBUG] sample constant cols:", dead.index.tolist()[:30])

        stds = X_fast.std(numeric_only=True)
        print("[DEBUG] top-10 std:", stds.sort_values(ascending=False).head(10).to_string())
        print("[DEBUG] bottom-10 std:", stds.sort_values(ascending=True).head(10).to_string())
    except Exception:
        pass

    proba_is_fast = extract_proba_class1(fast_clf.predict_proba(X_fast))
    EPS_THR = 1e-6
    target = 441
    mask = (wave_id.to_numpy() == target)
    if mask.any():
        p = float(proba_is_fast[mask][0])
        print(f"[DBG] wave={target} proba={p:.18f} proba+eps={p+EPS_THR:.18f} diff(thr-p)={(thr_high-p):.18e}")


    # --- zones ---
    #is_fast_strict = (proba_is_fast >= thr_high)
    EPS_THR = 1e-6
    is_fast_strict = (proba_is_fast + EPS_THR >= thr_high)
    is_fast_soft_zone = (proba_is_fast >= thr_low) & (~is_fast_strict)
    is_slow = (proba_is_fast < thr_low)

    # --- gate for SOFT only ---
    allow_soft = soft_fast_gate(df_feat)

    # if is_fast_soft_zone.any():
    #     dbg_cols = [c for c in ["edge_count","last_edge_pos_ratio","tail_ac_best","fft_spectral_entropy"] if c in df_feat.columns]
    #     tmp = df_feat.loc[is_fast_soft_zone, dbg_cols].copy()
    #     tmp["wave_id"] = wave_id.loc[is_fast_soft_zone].to_numpy()
    #     tmp["type_debug"] = type_series.loc[is_fast_soft_zone].to_numpy()
    #     tmp["proba"] = proba_is_fast[is_fast_soft_zone]
    #     tmp["allow_soft"] = allow_soft[is_fast_soft_zone]
    #     print("\n[DEBUG] soft-zone rows:")
    #     print(tmp.sort_values("proba").to_string(index=False))


    # final fast decision
    is_fast = is_fast_strict | (is_fast_soft_zone & allow_soft)

    # ✅ IMPORTANT: run reg for everything that is NOT fast (includes SOFT_BLOCKED + SLOW)
    need_reg_mask = ~is_fast
    wait_pred = np.full(len(df_feat), np.nan, dtype=float)

    if need_reg_mask.any():
        X_reg = align_columns(
            df_feat.loc[need_reg_mask].drop(columns=["wave_id"], errors="ignore"),
            reg_feature_cols,
        )
        wait_log = wait_reg.predict(X_reg)
        wait_pred[need_reg_mask] = np.expm1(np.asarray(wait_log, dtype=float))

    # final output
    pred_ms = np.where(is_fast, FAST_MS, wait_pred)
    pred_ms = np.clip(pred_ms, 0.0, None)


    # zone tag for debug (note: SOFT here means "in soft probability band", not necessarily final fast)
    fast_zone = np.where(
        is_fast_strict,
        "FAST_STRICT",
        np.where(
            is_fast_soft_zone,
            np.where(allow_soft, "FAST_SOFT", "SOFT_BLOCKED"),
            "SLOW",
        ),
    )


    # ---------- Stage B (reg only for SLOW) ----------
    # wait_pred = np.full(len(df_feat), np.nan, dtype=float)
    # if is_slow.any():
    #     X_reg = align_columns(
    #         df_feat.loc[is_slow].drop(columns=["wave_id"], errors="ignore"),
    #         reg_feature_cols,
    #     )
    #     wait_log = wait_reg.predict(X_reg)
    #     wait_pred[is_slow] = np.expm1(np.asarray(wait_log, dtype=float))

    # # ---------- Combine ----------
    # pred_ms = np.where(is_fast, FAST_MS, wait_pred)
    # pred_ms = np.clip(pred_ms, 0.0, None)


    out = pd.DataFrame(
        {
            "wave_id": wave_id,
            "type_debug": type_series,
            "fast_zone": fast_zone,
            "pred_wait_time_ms": pred_ms,
            "pred_is_fast": is_fast.astype(int),
            "proba_is_fast": proba_is_fast,
            "reg_wait_pred_ms": wait_pred,
        }
    )
    if thr_high != thr_low:
        print("\n[DEBUG] soft-zone count =", int(is_fast_soft_zone.sum()))
        print("[DEBUG] soft-zone allowed =", int((is_fast_soft_zone & allow_soft).sum()))
        print("[DEBUG] soft-zone blocked =", int((is_fast_soft_zone & (~allow_soft)).sum()))


    # ---------- Debug ----------
    if debug_topk > 0:
        suspicious = out[out["pred_is_fast"] == 1].copy()
        suspicious = suspicious.sort_values("proba_is_fast").head(debug_topk)

        print(f"\n🧪 Top-{debug_topk} borderline FAST (low confidence):")
        print(
            suspicious[
                ["wave_id", "type_debug", "fast_zone", "proba_is_fast", "pred_wait_time_ms", "reg_wait_pred_ms"]
            ].to_string(index=False)
        )

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--in", dest="input_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta-csv", default=None)
    ap.add_argument("--debug-topk", type=int, default=20)
    args = ap.parse_args()

    predict(
        model_path=args.model_path,
        input_csv=args.input_csv,
        out_csv=args.out,
        meta_csv=args.meta_csv,
        debug_topk=args.debug_topk,
    )


if __name__ == "__main__":
    main()
