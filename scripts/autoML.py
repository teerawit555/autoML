from __future__ import annotations

from datetime import datetime
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from autogluon.tabular import TabularPredictor


# =========================
# CONFIG
# =========================
# Meta columns ที่ไม่อยากให้โมเดลใช้เทรน (ยังคง wave_id ไว้ทำ report)
COLS_TO_DROP = ["force_mA", "range_V", "temp_C"]
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_SAVE_PATH = f"AutogluonModels/ag-{ts}"


# =========================
# HELPERS
# =========================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_text(path: str, default: str = "") -> str:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def align_columns(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    """
    ทำให้คอลัมน์ "ตรงกับตอนเทรน" แบบไม่พัง:
    - ถ้าขาด: เติม 0
    - ถ้าเกิน: ตัดทิ้ง
    - จัดเรียงตาม required_cols
    """
    out = df.copy()
    for c in required_cols:
        if c not in out.columns:
            out[c] = 0.0
    out = out[required_cols]
    return out


def pick_best_zero_threshold(
    proba_is_zero: np.ndarray,
    wait_pred: np.ndarray,
    y_true: np.ndarray,
    thr_min: float = 0.20,
    thr_max: float = 0.80,
    steps: int = 121,
) -> tuple[float, float]:
    """
    เลือก threshold ที่ทำให้ MAE ต่ำสุดบนชุด train (ทั้งชุด)
    y_hat = 0 ถ้า proba >= thr, ไม่งั้นใช้ wait_pred
    """
    best_thr = 0.5
    best_mae = float("inf")

    thr_grid = np.linspace(thr_min, thr_max, steps)
    for thr in thr_grid:
        is_zero = proba_is_zero >= thr
        y_hat = np.where(is_zero, 0.0, wait_pred)
        mae = float(np.mean(np.abs(y_hat - y_true)))
        if mae < best_mae:
            best_mae = mae
            best_thr = float(thr)

    return best_thr, best_mae


def proba_class1(p) -> np.ndarray:
    """
    AutoGluon บางเวอร์ชันคืน DataFrame(0/1), บางทีคืน ndarray/Series
    เราดึง proba ของ class=1 ให้ชัวร์
    """
    if hasattr(p, "columns") and (1 in list(p.columns)):
        return p[1].to_numpy()
    arr = np.asarray(p)
    # ถ้าเป็น shape (n,2) ให้เอาคอลัมน์ 1
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    return arr.astype(float)


# =========================
# TRAIN
# =========================
def train(
    data_path: str,
    label: str = "wait_time_ms",
    model_dir: str | None = None,
    presets: str = "medium_quality",
    time_limit: int = 120,
) -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Input file not found: "{data_path}"')

    df = pd.read_csv(data_path)

    # --- 1) basic sanity ---
    if "wave_id" not in df.columns:
        df["wave_id"] = np.arange(len(df), dtype=int)
    wave_id_backup = df["wave_id"].copy()

    # ตัดแถว label ว่าง
    df_train = df.dropna(subset=[label]).reset_index(drop=True)

    # สร้าง label ของ zero classifier
    df_train["is_zero"] = (df_train[label] == 0).astype(int)

    u = df_train["is_zero"].unique()
    if len(u) < 2:
        raise ValueError(
            f"is_zero has only 1 class in training data: {u}. "
            f"เช็ค label wait_time_ms แล้ว regenerate train_features ใหม่"
        )

    # เวลาแบ่งให้ 2-stage
    t_zero = max(30, int(time_limit * 0.25))
    t_reg = max(60, int(time_limit * 0.75))

    save_path = model_dir or DEFAULT_SAVE_PATH
    zero_model_path = os.path.join(save_path, "zero_clf")
    reg_model_path = os.path.join(save_path, "wait_reg")
    _ensure_dir(save_path)

    gpu_count = 1 if torch.cuda.is_available() else 0
    print(f"🚀 Training device: {'GPU (CUDA)' if gpu_count > 0 else 'CPU'}")
    print(f"presets={presets} time_limit={time_limit}s (zero={t_zero}s reg={t_reg}s)")

    # cols_to_drop_found (meta)
    cols_to_drop_found = [c for c in COLS_TO_DROP if c in df_train.columns]
    if cols_to_drop_found:
        print(f"Dropping meta columns: {cols_to_drop_found}")

    # =========================
    # Stage A: Zero Classifier
    # =========================
    df_zero = df_train.drop(columns=cols_to_drop_found, errors="ignore").copy()
    # ห้ามให้เห็น label จริง และ wave_id
    df_zero = df_zero.drop(columns=[label, "wave_id"], errors="ignore")

    # บันทึกคอลัมน์ที่ใช้เทรน (เพื่อ align ตอน predict)
    zero_feature_cols = [c for c in df_zero.columns if c != "is_zero"]
    save_json(os.path.join(save_path, "zero_feature_cols.json"), zero_feature_cols)

    zero_clf = TabularPredictor(
        label="is_zero",
        path=zero_model_path,
        problem_type="binary",
        eval_metric="f1",
        verbosity=2,
    ).fit(
        train_data=df_zero,
        presets=presets,
        time_limit=t_zero,
        num_gpus=gpu_count,
        dynamic_stacking=False,
    )

    # =========================
    # Stage B: Wait Regressor (non-zero only)
    # =========================
    df_nonzero = df_train[df_train[label] > 0].copy()
    if len(df_nonzero) < 5:
        raise ValueError(
            f"non-zero samples too few ({len(df_nonzero)}). "
            f"ต้องมีเคส wait>0 มากกว่านี้เพื่อเทรน regressor"
        )

    # log1p เพื่อให้สเกลนิ่งขึ้น
    df_nonzero["wait_time_log"] = np.log1p(df_nonzero[label].astype(float))

    df_reg = df_nonzero.drop(columns=cols_to_drop_found, errors="ignore").copy()
    # ห้ามใช้ wave_id / label เดิม / is_zero
    df_reg_for_fit = df_reg.drop(columns=["wave_id", label, "is_zero"], errors="ignore")

    # บันทึกคอลัมน์ที่ reg ใช้เทรน
    reg_feature_cols = [c for c in df_reg_for_fit.columns if c != "wait_time_log"]
    save_json(os.path.join(save_path, "reg_feature_cols.json"), reg_feature_cols)

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

    # =========================
    # 3) ANALYSIS (คงโครงเดิม + เพิ่ม auto-threshold)
    # =========================
    print("\n" + "=" * 60)
    print(" DEEP MODEL ANALYSIS & DIAGNOSIS")
    print("=" * 60)

    # A. Feature Importance (Regressor)
    print("\n[1] Calculating Feature Importance (Regressor)...")
    importance = wait_reg.feature_importance(df_reg_for_fit)
    print(importance.head(15))

    # B. Leaderboard (Regressor)
    print("\n[2] Model Leaderboard (Regressor):")
    leaderboard = wait_reg.leaderboard(df_reg_for_fit, silent=True)
    cols_lb = [c for c in ["model", "score_val", "pred_time_val", "fit_time"] if c in leaderboard.columns]
    print(leaderboard[cols_lb].head(5))

    # C. Residual Analysis (2-stage)
    X_all = df_train.drop(columns=[label])
    y_actual = df_train[label].to_numpy(dtype=float)

    # ลบ meta
    X_feat = X_all.drop(columns=cols_to_drop_found, errors="ignore").copy()

    # --- Zero proba ---
    X_zero_in = X_feat.drop(columns=["wave_id"], errors="ignore").copy()
    # align ให้ตรงกับตอนเทรน
    X_zero_in = align_columns(X_zero_in, zero_feature_cols)
    proba = zero_clf.predict_proba(X_zero_in)
    proba_is_zero = proba_class1(proba)

    # --- Regressor prediction (ให้ pred ได้ “ทุกแถว” เพื่อใช้เลือก threshold) ---
    X_reg_in = X_feat.drop(columns=["wave_id", "is_zero", label, "wait_time_log"], errors="ignore").copy()
    X_reg_in = align_columns(X_reg_in, reg_feature_cols)
    wait_log_pred = wait_reg.predict(X_reg_in)
    wait_pred = np.expm1(np.asarray(wait_log_pred, dtype=float))
    wait_pred = np.clip(wait_pred, 0, None)

    # --- Auto pick threshold จาก train ---
    best_thr, best_mae = pick_best_zero_threshold(
        proba_is_zero=proba_is_zero,
        wait_pred=wait_pred,
        y_true=y_actual,
        thr_min=0.20,
        thr_max=0.80,
        steps=121,
    )
    save_text(os.path.join(save_path, "zero_threshold.txt"), str(best_thr))

    print(f"\n[2.5] Auto-picked zero threshold = {best_thr:.3f} (train MAE={best_mae:.3f})")
    print(f"proba_is_zero min/mean/max = {proba_is_zero.min():.6f} {proba_is_zero.mean():.6f} {proba_is_zero.max():.6f}")

    is_zero_pred = proba_is_zero >= best_thr
    y_pred = np.where(is_zero_pred, 0.0, wait_pred)
    print(f"predicted zeros = {int(is_zero_pred.sum())} / {len(is_zero_pred)}")

    # Report DF
    out = df_train.copy()
    out["wave_id"] = wave_id_backup.iloc[df_train.index].values
    out["pred_wait_time_ms"] = y_pred
    out["error_ms"] = out["pred_wait_time_ms"] - out[label]
    out["abs_error_ms"] = out["error_ms"].abs()

    print("\n[3] TOP 10 WORST PREDICTIONS (Review these Wave IDs!):")
    cols_to_show = ["wave_id", label, "pred_wait_time_ms", "error_ms"]
    cols_to_show = [c for c in cols_to_show if c in out.columns]
    worst_10 = out.sort_values(by="abs_error_ms", ascending=False).head(10)
    print(worst_10[cols_to_show])

    # =========================
    # 4) SAVE DIAGNOSIS DATA (คงโครงเดิม)
    # =========================
    _ensure_dir("data/processed/analysis")
    _ensure_dir("data/processed/train")

    diag_path = f"data/processed/analysis/diagnosis_report_{ts}.csv"
    out.to_csv(diag_path, index=False)

    feat_imp_path = f"data/processed/analysis/feature_importance_{ts}.csv"
    importance.to_csv(feat_imp_path)

    train_out_path = f"data/processed/train/train_with_predictions_{ts}.csv"
    out.drop(columns=["abs_error_ms"], errors="ignore").to_csv(train_out_path, index=False)

    print(f"\n✅ Analysis Report Saved: {diag_path}")
    print(f"✅ Feature Importance Saved: {feat_imp_path}")
    print(f"✅ Models saved at: {save_path}")
    print(f"   - Zero classifier: {zero_model_path}")
    print(f"   - Wait regressor:  {reg_model_path}")
    print(f"   - Zero threshold:  {best_thr:.3f}  (saved: {os.path.join(save_path, 'zero_threshold.txt')})")
    print("=" * 60)


# =========================
# PREDICT
# =========================
def predict(
    model_path: str,
    input_csv: str,
    out_csv: str = "predictions.csv",
) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found at: "{model_path}"')

    zero_model_path = os.path.join(model_path, "zero_clf")
    reg_model_path = os.path.join(model_path, "wait_reg")
    if not os.path.exists(zero_model_path) or not os.path.exists(reg_model_path):
        raise FileNotFoundError(
            "Expected sub-models not found.\n"
            f"Missing: {zero_model_path} or {reg_model_path}\n"
            "Train must create both folders: zero_clf + wait_reg"
        )

    zero_feature_cols = load_json(os.path.join(model_path, "zero_feature_cols.json"), default=None)
    reg_feature_cols = load_json(os.path.join(model_path, "reg_feature_cols.json"), default=None)
    if not zero_feature_cols or not reg_feature_cols:
        raise FileNotFoundError(
            "Missing feature column metadata.\n"
            "Expected: zero_feature_cols.json and reg_feature_cols.json in model folder."
        )

    threshold = float(load_text(os.path.join(model_path, "zero_threshold.txt"), default="0.5"))
    print(f"🔮 Loading models and predicting: {input_csv}")
    print(f"using zero threshold = {threshold:.3f}")

    zero_clf = TabularPredictor.load(zero_model_path)
    wait_reg = TabularPredictor.load(reg_model_path)

    df = pd.read_csv(input_csv)

    # backup wave_id เพื่อแปะคืน
    wave_id_backup = df["wave_id"].copy() if "wave_id" in df.columns else None

    # drop meta columns (ให้เหมือนตอนเทรน)
    cols_to_drop_found = [c for c in COLS_TO_DROP if c in df.columns]
    df_feat = df.drop(columns=cols_to_drop_found, errors="ignore").copy()

    # ---- Stage A ----
    X_zero_in = df_feat.drop(columns=["wave_id"], errors="ignore").copy()
    X_zero_in = align_columns(X_zero_in, zero_feature_cols)

    proba = zero_clf.predict_proba(X_zero_in)
    proba_is_zero = proba_class1(proba)
    is_zero_pred = proba_is_zero >= threshold

    print(f"proba_is_zero min/mean/max = {proba_is_zero.min():.6f} {proba_is_zero.mean():.6f} {proba_is_zero.max():.6f}")
    print(f"predicted zeros = {int(is_zero_pred.sum())} / {len(is_zero_pred)}")

    # ---- Stage B ----
    X_reg_in = df_feat.drop(columns=["wave_id", "is_zero", "wait_time_ms", "wait_time_log"], errors="ignore").copy()
    X_reg_in = align_columns(X_reg_in, reg_feature_cols)

    wait_log_pred = wait_reg.predict(X_reg_in)
    wait_pred = np.expm1(np.asarray(wait_log_pred, dtype=float))
    wait_pred = np.clip(wait_pred, 0, None)

    # ถ้า Model ทายว่าเป็น Zero Class (0.0) ให้ปัดเป็น 0.1 (Default)
    final_pred = np.where(is_zero_pred, 0.1, wait_pred)
    final_pred = np.maximum(final_pred, 0.1)

    # ======================================================
    # HYBRID LOGIC OVERRIDE
    # ======================================================
    # ใช้ Logic Flag จาก df ต้นฉบับมาบังคับค่า (Override AI)
    # Sine/Pulse หรือ Glitch ที่ Logic จับได้ จะถูกบังคับเป็น 0.0ms ทันที

    if "logic_flag_continuous" in df.columns:
        mask_cont = df["logic_flag_continuous"] == 1
        count_cont = mask_cont.sum()
        if count_cont > 0:
            print(f"⚡ Applying Logic Override for Continuous Waves: {count_cont} items forced to 0.0ms")
            final_pred[mask_cont] = 0.0

    if "logic_flag_glitch" in df.columns:
        mask_glitch = df["logic_flag_glitch"] == 1
        count_glitch = mask_glitch.sum()
        if count_glitch > 0:
            print(f"⚡ Applying Logic Override for Glitch: {count_glitch} items forced to 0.0ms")
            final_pred[mask_glitch] = 0.0
    # ======================================================

    # Output (คงโครง: wave_id + pred ก่อน)
    out = df_feat.copy()
    if wave_id_backup is not None:
        out["wave_id"] = wave_id_backup

    out["pred_wait_time_ms"] = final_pred

    # แถม logic flag ติดไปด้วยเพื่อความชัวร์ตอน check results
    if "logic_flag_continuous" in df.columns:
        out["logic_flag_continuous"] = df["logic_flag_continuous"]

    first_cols = [c for c in ["wave_id", "pred_wait_time_ms", "logic_flag_continuous"] if c in out.columns]
    other_cols = [c for c in out.columns if c not in first_cols]
    out = out[first_cols + other_cols]

    _ensure_dir(os.path.dirname(out_csv) or ".")
    out.to_csv(out_csv, index=False)
    print(f"✅ Prediction Results saved: {out_csv}")


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="AutoGluon 2-stage (zero clf + wait reg) with auto threshold")
    ap.add_argument("--mode", default="train", choices=["train", "predict"])
    ap.add_argument("--data", default="data/processed/train/train_features.csv")
    ap.add_argument("--label", default="wait_time_ms")
    ap.add_argument("--model-dir", default=None, help="optional custom save dir for train")
    ap.add_argument("--model-path", default=None, help="required for predict mode")
    ap.add_argument("--inference-csv", default="data/processed/inference/wide.csv")
    ap.add_argument("--out", default="data/processed/prediction/predicted_wait_time.csv")
    ap.add_argument("--time-limit", type=int, default=120)
    ap.add_argument("--presets", default="medium_quality", help="e.g. medium_quality, high_quality, best_quality")

    args = ap.parse_args()

    try:
        if args.mode == "train":
            train(
                data_path=args.data,
                label=args.label,
                model_dir=args.model_dir,
                presets=args.presets,
                time_limit=args.time_limit,
            )
        else:
            if not args.model_path:
                print("❌ Error: Please specify --model-path for prediction mode.")
                return
            predict(
                model_path=args.model_path,
                input_csv=args.inference_csv,
                out_csv=args.out,
            )
    except Exception as e:
        print(f" ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


#---------------------------------------------
# V-14
#---------------------------------------------
# from __future__ import annotations

# from datetime import datetime
# from pathlib import Path
# import argparse
# import os
# import sys

# import pandas as pd
# import numpy as np
# import torch

# from autogluon.tabular import TabularPredictor


# # Meta columns ที่เราจะไม่ใช้เทรน (เพื่อให้ Model โฟกัสที่ลักษณะคลื่น)
# # NOTE: ไม่ drop wave_id ใน list นี้แล้ว (จะเอาไว้ทำ report/diagnosis)
# COLS_TO_DROP = ['force_mA', 'range_V', 'temp_C']
# ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# DEFAULT_SAVE_PATH = f"AutogluonModels/ag-{ts}"


# def train(
#     data_path: str,
#     label: str = "wait_time_ms",
#     model_dir: str | None = None,
#     presets: str = "medium_quality",
#     time_limit: int = 60,
# ) -> None:
#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f'Input file not found: "{data_path}"')

#     df = pd.read_csv(data_path)

#     # --- 1. PREPROCESSING ---
#     if "wave_id" not in df.columns:
#         df["wave_id"] = np.arange(len(df), dtype=int)

#     wave_id_backup = df["wave_id"].copy()

#     cols_to_drop_found = [c for c in COLS_TO_DROP if c in df.columns]
#     if cols_to_drop_found:
#         print(f"Dropping meta columns: {cols_to_drop_found}")

#     df_train = df.dropna(subset=[label]).reset_index(drop=True)

#     # classifier label: is_zero
#     df_train["is_zero"] = (df_train[label] == 0).astype(int)

#     u = df_train["is_zero"].unique()
#     if len(u) < 2:
#         raise ValueError(
#             f"is_zero has only 1 class in training data: {u}. "
#             f"Check your labeling (wait_time_ms) and regenerate train_features."
#         )

#     t_zero = max(30, int(time_limit * 0.25))
#     t_reg = max(60, int(time_limit * 0.75))

#     save_path = model_dir or DEFAULT_SAVE_PATH
#     gpu_count = 1 if torch.cuda.is_available() else 0
#     print(f"🚀 Training device: {'GPU (CUDA)' if gpu_count > 0 else 'CPU'}")

#     # --- 2. FIT MODELS (2-STAGE) ---
#     # Stage A: Zero Classifier
#     zero_model_path = os.path.join(save_path, "zero_clf")

#     df_zero = df_train.drop(columns=cols_to_drop_found, errors="ignore").copy()
#     # ห้าม classifier เห็น label จริง + ห้ามใช้ wave_id
#     df_zero = df_zero.drop(columns=[label, "wave_id"], errors="ignore")

#     zero_clf = TabularPredictor(
#         label="is_zero",
#         path=zero_model_path,
#         problem_type="binary",
#         eval_metric="f1",
#         verbosity=2,
#     ).fit(
#         train_data=df_zero,
#         presets=presets,
#         time_limit=t_zero,
#         num_gpus=gpu_count,
#     )

#     # Stage B: Regressor (เฉพาะ wait>0)
#     reg_model_path = os.path.join(save_path, "wait_reg")

#     df_nonzero = df_train[df_train[label] > 0].copy()
#     df_nonzero["wait_time_log"] = np.log1p(df_nonzero[label].astype(float))

#     df_reg = df_nonzero.drop(columns=cols_to_drop_found, errors="ignore").copy()
#     # regressor ไม่ให้เห็น wave_id / label เดิม / is_zero
#     df_reg_for_fit = df_reg.drop(columns=["wave_id", label, "is_zero"], errors="ignore")

#     predictor = TabularPredictor(
#         label="wait_time_log",
#         path=reg_model_path,
#         problem_type="regression",
#         eval_metric="mean_absolute_error",
#         verbosity=2,
#     ).fit(
#         train_data=df_reg_for_fit,
#         presets=presets,
#         time_limit=t_reg,
#         num_gpus=gpu_count,
#     )

#     # --- 3. MODEL ANALYSIS (คงโครงเดิม) ---
#     print("\n" + "=" * 60)
#     print("🔍 DEEP MODEL ANALYSIS & DIAGNOSIS")
#     print("=" * 60)

#     # A. Feature Importance (Regressor)
#     print("\n[1] Calculating Feature Importance (Regressor)...")
#     importance = predictor.feature_importance(df_reg_for_fit)
#     print(importance.head(15))

#     # B. Leaderboard (Regressor)
#     print("\n[2] Model Leaderboard (Regressor):")
#     leaderboard = predictor.leaderboard(df_reg_for_fit, silent=True)
#     cols_lb = [c for c in ["model", "score_val", "pred_time_val", "fit_time"] if c in leaderboard.columns]
#     print(leaderboard[cols_lb].head(5))

#     # C. Residual Analysis (2-stage combine)  ✅ FIX จุดนี้
#     X_test = df_train.drop(columns=[label])
#     y_actual = df_train[label].astype(float)

#     # เตรียม feature base (drop meta)
#     X_feat = X_test.drop(columns=cols_to_drop_found, errors="ignore").copy()

#     # ---- Stage A: predict proba zero ----
#     X_zero_in = X_feat.drop(columns=["wave_id"], errors="ignore").copy()
#     proba = zero_clf.predict_proba(X_zero_in)

#     if hasattr(proba, "columns") and (1 in list(proba.columns)):
#         proba_is_zero = proba[1].values
#     else:
#         proba_is_zero = np.array(proba)

#     threshold = 0.5  # เริ่มจาก 0.45 ก่อน (ค่อยจูน)
#     is_zero_pred = proba_is_zero >= threshold

#     print("proba_is_zero min/mean/max =",
#           float(np.min(proba_is_zero)), float(np.mean(proba_is_zero)), float(np.max(proba_is_zero)))
#     print("predicted zeros =", int(is_zero_pred.sum()), "/", len(is_zero_pred))

#     # ---- Stage B: predict wait (log -> ms) ----
#     # ✅ FIX: drop แค่ wave_id + is_zero ให้ feature set ตรงกับตอนเทรน regressor
#     X_reg_in = X_feat.drop(columns=["wave_id", "is_zero"], errors="ignore").copy()

#     wait_log_pred = predictor.predict(X_reg_in)
#     wait_pred = np.expm1(wait_log_pred.astype(float))
#     wait_pred = np.clip(wait_pred, 0, None)

#     # ---- Combine ----
#     y_pred = np.where(is_zero_pred, 0.0, wait_pred)

#     # Report dataframe
#     out = df_train.copy()
#     out["wave_id"] = wave_id_backup.iloc[df_train.index].values
#     out["pred_wait_time_ms"] = y_pred
#     out["error_ms"] = out["pred_wait_time_ms"] - y_actual
#     out["abs_error_ms"] = out["error_ms"].abs()

#     print("\n[3] TOP 10 WORST PREDICTIONS (Review these Wave IDs!):")
#     cols_to_show = ["wave_id", label, "pred_wait_time_ms", "error_ms"]
#     cols_to_show = [c for c in cols_to_show if c in out.columns]
#     worst_10 = out.sort_values(by="abs_error_ms", ascending=False).head(10)
#     print(worst_10[cols_to_show])

#     # --- 4. SAVE DIAGNOSIS DATA (คงโครงเดิม) ---
#     os.makedirs("data/processed/analysis", exist_ok=True)
#     os.makedirs("data/processed/train", exist_ok=True)

#     diag_path = f"data/processed/analysis/diagnosis_report_{ts}.csv"
#     out.to_csv(diag_path, index=False)

#     feat_imp_path = f"data/processed/analysis/feature_importance_{ts}.csv"
#     importance.to_csv(feat_imp_path)

#     train_out_path = f"data/processed/train/train_with_predictions_{ts}.csv"
#     out.drop(columns=["abs_error_ms"], errors="ignore").to_csv(train_out_path, index=False)

#     print(f"\n✅ Analysis Report Saved: {diag_path}")
#     print(f"✅ Feature Importance Saved: {feat_imp_path}")
#     print(f"✅ Models saved at: {save_path}")
#     print(f"   - Zero classifier: {zero_model_path}")
#     print(f"   - Wait regressor:  {reg_model_path}")
#     print("=" * 60)

# def predict(
#     model_path: str,
#     input_csv: str,
#     out_csv: str = "predictions.csv",
# ) -> None:
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f'Model not found at: "{model_path}"')

#     zero_model_path = os.path.join(model_path, "zero_clf")
#     reg_model_path = os.path.join(model_path, "wait_reg")
#     if not os.path.exists(zero_model_path) or not os.path.exists(reg_model_path):
#         raise FileNotFoundError(
#             f"Expected sub-models not found.\n"
#             f"Missing: {zero_model_path} or {reg_model_path}\n"
#             f"Train must create both folders: zero_clf + wait_reg"
#         )

#     print(f"🔮 Loading models and predicting: {input_csv}")
#     zero_clf = TabularPredictor.load(zero_model_path)
#     wait_reg = TabularPredictor.load(reg_model_path)

#     df = pd.read_csv(input_csv)

#     # backup wave_id เพื่อแปะคืน
#     wave_id_backup = df['wave_id'].copy() if 'wave_id' in df.columns else None

#     # drop meta columns
#     cols_to_drop_found = [c for c in COLS_TO_DROP if c in df.columns]
#     df_feat = df.drop(columns=cols_to_drop_found, errors="ignore").copy()

#     # Stage A: predict proba zero
#     X_zero_in = df_feat.drop(columns=["wave_id"], errors="ignore").copy()
#     proba = zero_clf.predict_proba(X_zero_in)
#     if hasattr(proba, "columns") and (1 in list(proba.columns)):
#         proba_is_zero = proba[1].values
#     else:
#         proba_is_zero = np.array(proba)

#     threshold = 0.5
#     is_zero_pred = proba_is_zero >= threshold

#     print("proba_is_zero min/mean/max =", float(np.min(proba_is_zero)), float(np.mean(proba_is_zero)), float(np.max(proba_is_zero)))
#     print("predicted zeros =", int(is_zero_pred.sum()), "/", len(is_zero_pred))

#     # Stage B: regression (log -> ms)
#     X_reg_in = df_feat.drop(columns=["wave_id", "is_zero", "wait_time_ms", "wait_time_log"], errors="ignore").copy()
#     wait_log_pred = wait_reg.predict(X_reg_in)
#     wait_pred = np.expm1(wait_log_pred.astype(float))
#     wait_pred = np.clip(wait_pred, 0, None)

#     final_pred = np.where(is_zero_pred, 0.0, wait_pred)

#     # Output (คงโครงแบบเดิม: มี wave_id + pred)
#     out = df_feat.copy()
#     if wave_id_backup is not None:
#         out["wave_id"] = wave_id_backup

#     out["pred_wait_time_ms"] = final_pred

#     # Reorder columns ให้ ID และผลทายอยู่หน้าสุดเพื่อให้ดูง่าย
#     first_cols = [c for c in ["wave_id", "pred_wait_time_ms"] if c in out.columns]
#     other_cols = [c for c in out.columns if c not in first_cols]
#     out = out[first_cols + other_cols]

#     os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
#     out.to_csv(out_csv, index=False)
#     print(f"✅ Prediction Results saved: {out_csv}")


# def main():
#     ap = argparse.ArgumentParser(description="AutoGluon Workflow with Analysis")
#     ap.add_argument("--mode", default="train", choices=["train", "predict"])
#     ap.add_argument("--data", default="data/processed/train/train_features.csv")
#     ap.add_argument("--label", default="wait_time_ms")
#     ap.add_argument("--model-path", default=None)  # สำหรับโหมด predict
#     ap.add_argument("--inference-csv", default="data/processed/inference/features_test_2.csv")
#     ap.add_argument("--out", default="data/processed/prediction/final_results.csv")
#     ap.add_argument("--time-limit", type=int, default=120)  # เพิ่มเวลาเทรนเริ่มต้น
#     ap.add_argument("--presets", default="medium_quality")

#     args = ap.parse_args()

#     try:
#         if args.mode == "train":
#             train(
#                 data_path=args.data,
#                 label=args.label,
#                 presets=args.presets,
#                 time_limit=args.time_limit,
#             )
#         else:
#             if not args.model_path:
#                 print("❌ Error: Please specify --model-path for prediction mode.")
#                 return
#             predict(
#                 model_path=args.model_path,
#                 input_csv=args.inference_csv,
#                 out_csv=args.out,
#             )
#     except Exception as e:
#         print(f"💥 ERROR: {e}")
#         sys.exit(1)


# if __name__ == "__main__":
#     main()
