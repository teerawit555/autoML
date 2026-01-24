#scripts/autoML.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

# ==========================================================
# Fixed Feature Set (ใช้เหมือนกันทั้ง TRAIN และ PRED)
# ==========================================================
FEATURE_COLS = [
    "t_enter_band_ms",
    "enter_margin_ms",
    "t_enter_norm",
    "enter_vs_settle",
    "band_to_p2p",
    "band_width",
    "t_est_settle_ms",
    "tail_p2p",
    "std_value",   
    "max_slope",
]



# ==========================================================
# Constants
# ==========================================================
EPS = 1e-12
MIN_PRED_MS = 0.1

WATCH_COLS = [
    "t_enter_band_ms",
    "t_est_settle_ms",
    "band_width",
    "tail_p2p",   
    "tail_std",
    "max_slope",
]


# ==========================================================
# Logging (datalog)
# ==========================================================
def setup_logger(log_path: str) -> logging.Logger:
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("runlog")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ==========================================================
# Helpers
# ==========================================================
def ensure_required_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{context}: missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def log_df_describe(logger: logging.Logger, df: pd.DataFrame, cols: list[str], title: str) -> None:
    present = [c for c in cols if c in df.columns]
    if not present:
        logger.info(f"{title}: (no columns found)")
        return
    logger.info(f"{title}:\n{df[present].describe().to_string()}")


def save_and_log_worst(
    logger: logging.Logger,
    out_df: pd.DataFrame,
    name: str,
    topn: int = 20
) -> None:
    if "abs_error_ms" not in out_df.columns:
        return

    logger.info(f"{name}: abs_error summary:\n{out_df['abs_error_ms'].describe().to_string()}")
    worst = out_df.sort_values("abs_error_ms", ascending=False).head(topn)
    logger.info(f"{name}: Top-{topn} worst:\n{worst.to_string(index=False)}")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ทำ feature engineering ให้เหมือนเดิม (TRAIN/PRED ทำเหมือนกัน)
    NOTE: ตอนนี้ FEATURE_COLS อาจจะไม่ได้ใช้ทุกตัวที่สร้างไว้ แต่คง behavior เดิมไว้
    """
    out = df.copy()

    # ต้องมีคอลัมน์พื้นฐานที่ใช้คำนวณ
    needed = ["band_width", "p2p_value", "tail_std", "std_value", "t_enter_band_ms", "t_est_settle_ms", "max_slope"]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        # ไม่ raise เพื่อไม่เปลี่ยน behavior แบบเดิมมากเกินไป
        # แต่โดยปกติถ้าขาด แปลว่า features_csv ไม่ถูกต้อง
        return out

    out["band_to_p2p"] = out["band_width"] / (out["p2p_value"] + EPS)
    out["tail_noise_ratio"] = out["tail_std"] / (out["std_value"] + EPS)
    out["enter_vs_settle"] = out["t_enter_band_ms"] / (out["t_est_settle_ms"] + 1e-6)
    out["slope_norm"] = out["max_slope"] / (out["p2p_value"] + EPS)
    return out


# ==========================================================
# TRAIN
# ==========================================================
def run_train(features_csv: str, out_dir: str, seed: int, log_path: str) -> None:
    logger = setup_logger(log_path)

    model_dir = Path(out_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📂 Loading TRAIN features: {features_csv}")
    df = pd.read_csv(features_csv)

    # ทำ feature engineering (เอาไว้เหมือนเดิม แต่ไม่ทำซ้ำสองรอบแล้ว)
    df = add_engineered_features(df)

    # log engineered summary (เหมือนเดิม)
    engineered_cols = ["band_to_p2p", "tail_noise_ratio", "enter_vs_settle", "slope_norm"]
    present_eng = [c for c in engineered_cols if c in df.columns]
    if present_eng:
        logger.info("Engineered feature summary:\n" + df[present_eng].describe().to_string())

    ensure_required_columns(df, ["wave_id", "wait_time_ms"], "TRAIN")
    ensure_required_columns(df, FEATURE_COLS, "TRAIN")

    logger.info(f"Rows={len(df)} | Cols={len(df.columns)}")
    logger.info("Label (wait_time_ms) summary:\n" + df["wait_time_ms"].describe().to_string())

    log_df_describe(logger, df, WATCH_COLS, "Key feature summary")

    # สร้าง data สำหรับ AutoGluon
    X = df[FEATURE_COLS].copy()
    y = df["wait_time_ms"].astype(float)
    wave_id = df["wave_id"].copy()

    data = X.copy()
    data["wait_time_ms"] = y
    data["wave_id"] = wave_id  # เก็บไว้ export (โมเดลไม่เห็น)

    # ----------------------------
    # Split 80 / 20
    # ----------------------------
    rng = np.random.default_rng(seed)
    idx = np.arange(len(data))
    rng.shuffle(idx)

    split_at = int(0.8 * len(idx))
    train_idx = idx[:split_at]
    val_idx = idx[split_at:]

    train_data = data.iloc[train_idx].reset_index(drop=True)
    val_data = data.iloc[val_idx].reset_index(drop=True)

    logger.info(f"Split: train={len(train_data)} | val={len(val_data)}")
    logger.info(f"Using {len(FEATURE_COLS)} fixed features: {FEATURE_COLS}")
    logger.info(f"[DEBUG][TRAIN] FEATURE_COLS = {FEATURE_COLS}")

    # ----------------------------
    # Train AutoGluon
    # ----------------------------
    logger.info("🚀 Training AutoGluon...")
    predictor = TabularPredictor(
        label="wait_time_ms",
        problem_type="regression",
        eval_metric="mean_absolute_error",
        path=str(model_dir),
    )

    predictor.fit(
        train_data=train_data.drop(columns=["wave_id"]),
        tuning_data=val_data.drop(columns=["wave_id"]),
        presets="medium_quality",
        verbosity=1,
    )

    # ==========================================================
    # Predict ALL training data (100%)
    # ==========================================================
    logger.info("📤 Predicting ALL training data (100%)...")
    preds_all = predictor.predict(data[FEATURE_COLS].copy())
    preds_all = np.maximum(np.asarray(preds_all, dtype=float), MIN_PRED_MS)

    extra_cols = [c for c in WATCH_COLS if c in df.columns]
    out_all = df[["wave_id", "wait_time_ms"] + extra_cols].copy()
    out_all["pred_wait_time_ms"] = preds_all
    out_all["error_ms"] = out_all["pred_wait_time_ms"] - out_all["wait_time_ms"]
    out_all["abs_error_ms"] = out_all["error_ms"].abs()
    out_all = out_all.sort_values("wave_id").reset_index(drop=True)

    out_all_path = Path("data/processed/analysis/train_predictions_all.csv")
    out_all_path.parent.mkdir(parents=True, exist_ok=True)
    out_all.to_csv(out_all_path, index=False)

    logger.info(f"✅ Saved ALL-train predictions to: {out_all_path}")
    save_and_log_worst(logger, out_all, "ALL-TRAIN", topn=20)

    # ----------------------------
    # Evaluate (VAL)
    # ----------------------------
    logger.info("📊 Evaluating on validation...")
    perf = predictor.evaluate(val_data.drop(columns=["wave_id"]))
    logger.info("Validation performance:")
    for k, v in perf.items():
        try:
            logger.info(f"  {k}: {float(v):.6f}")
        except Exception:
            logger.info(f"  {k}: {v}")

    # ------------------------------
    # Save validation predictions
    # ------------------------------
    logger.info("📤 Predicting VAL (20%)...")
    preds_val = predictor.predict(val_data[FEATURE_COLS].copy())
    preds_val = np.maximum(np.asarray(preds_val, dtype=float), MIN_PRED_MS)

    out_val = val_data[["wave_id", "wait_time_ms"]].copy()
    out_val["pred_wait_time_ms"] = preds_val
    out_val["error_ms"] = out_val["pred_wait_time_ms"] - out_val["wait_time_ms"]
    out_val["abs_error_ms"] = out_val["error_ms"].abs()
    out_val = out_val.sort_values("wave_id").reset_index(drop=True)

    out_val_path = Path("data/processed/analysis/val_predictions.csv")
    out_val_path.parent.mkdir(parents=True, exist_ok=True)
    out_val.to_csv(out_val_path, index=False)

    logger.info(f"✅ Saved validation predictions to: {out_val_path}")
    save_and_log_worst(logger, out_val, "VAL", topn=20)

    # -------------------------------------------------
    # Feature Importance
    # -------------------------------------------------
    logger.info("🔍 Computing feature importance...")
    try:
        lb = predictor.leaderboard(val_data.drop(columns=["wave_id"]), silent=True)
        best_model = lb.iloc[0]["model"]
        logger.info(f"Best model for FI: {best_model}")

        fi = predictor.feature_importance(
            val_data.drop(columns=["wave_id"]),
            model=best_model
        )
        fi = fi.reset_index().rename(columns={"index": "feature"})
        fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)

        fi_path = Path("data/processed/analysis/feature_importance.csv")
        fi_path.parent.mkdir(parents=True, exist_ok=True)
        fi.to_csv(fi_path, index=False)

        logger.info(f"✅ Feature importance saved to: {fi_path}")
        logger.info("Top-15 FI:\n" + fi.head(15).to_string(index=False))

    except Exception as e:
        logger.exception(f"❌ Feature importance failed: {e}")


# ==========================================================
# PREDICT
# ==========================================================
def run_pred(features_csv: str, model_dir: str, out_csv: str, log_path: str) -> None:
    logger = setup_logger(log_path)

    logger.info(f"📂 Loading PRED features: {features_csv}")
    df = pd.read_csv(features_csv)

    df = add_engineered_features(df)

    engineered_cols = ["band_to_p2p", "tail_noise_ratio", "enter_vs_settle", "slope_norm"]
    present_eng = [c for c in engineered_cols if c in df.columns]
    if present_eng:
        logger.info("PRED engineered feature summary:\n" + df[present_eng].describe().to_string())

    ensure_required_columns(df, ["wave_id"], "PRED")
    ensure_required_columns(df, FEATURE_COLS, "PRED")

    X = df[FEATURE_COLS].copy()
    predictor = TabularPredictor.load(model_dir)
    logger.info(f"[DEBUG][PRED] FEATURE_COLS = {FEATURE_COLS}")
    logger.info(f"[DEBUG][PRED] df.columns = {df.columns.tolist()}")
    logger.info(f"[DEBUG][PRED] Model expects = {predictor.feature_generator.features_in}")

    preds = predictor.predict(X)
    preds = np.maximum(np.asarray(preds, dtype=float), MIN_PRED_MS)

    out = pd.DataFrame({
        "wave_id": df["wave_id"].copy(),
        "pred_wait_time_ms": preds,
    }).sort_values("wave_id").reset_index(drop=True)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    logger.info(f"✅ Saved predictions to: {out_path}")


# ==========================================================
# Main
# ==========================================================
def main():
    ap = argparse.ArgumentParser("autoML (fixed feature set) - train or predict wait_time_ms")
    ap.add_argument("--mode", required=True, choices=["train", "pred"])
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", default="data/logs/run.log")

    # train
    ap.add_argument("--out_dir", default="models/autoML_wait_time")

    # pred
    ap.add_argument("--model_dir", default="models/autoML_wait_time")
    ap.add_argument("--pred_out", default="data/processed/analysis/predictions.csv")

    args = ap.parse_args()

    if args.mode == "train":
        run_train(args.features_csv, args.out_dir, args.seed, args.log)
    else:
        run_pred(args.features_csv, args.model_dir, args.pred_out, args.log)


if __name__ == "__main__":
    main()
