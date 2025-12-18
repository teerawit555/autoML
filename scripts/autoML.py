from __future__ import annotations
from datetime import datetime
from pathlib import Path
import argparse
import os
import sys
import pandas as pd
import torch # Added for GPU detection

from autogluon.tabular import TabularPredictor

COLS_TO_DROP = ['force_mA', 'range_V', 'temp_C']
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# Default save path logic
DEFAULT_SAVE_PATH = f"AutogluonModels/ag-{ts}"

def train(
    data_path: str,
    label: str = "wait_time_ms",
    model_dir: str | None = None,
    presets: str = "medium_quality",
    time_limit: int = 60,
) -> None:
    """
    Train a TabularPredictor, show comparison table, and save results.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Input file not found: "{data_path}"')

    df = pd.read_csv(data_path)
    if label not in df.columns:
        raise ValueError(f'Label column "{label}" not found in {data_path}')

    # --- 1. PREPROCESSING ---
    cols_to_drop_found = [c for c in COLS_TO_DROP if c in df.columns]
    if cols_to_drop_found:
        print(f"Dropping columns not present in inference data: {cols_to_drop_found}")
        df = df.drop(columns=cols_to_drop_found)

    df = df.dropna(subset=[label]).reset_index(drop=True)
    
    save_path = model_dir or DEFAULT_SAVE_PATH
    gpu_count = 1 if torch.cuda.is_available() else 0
    print(f"Training device: {'GPU' if gpu_count > 0 else 'CPU'}")

    # --- 2. FIT MODEL ---
    predictor = TabularPredictor(
        label=label,
        path=save_path,
        problem_type="regression",
        eval_metric="mean_absolute_error",
        verbosity=2,
    )

    predictor.fit(
        train_data=df,
        presets=presets,
        time_limit=time_limit,
        num_gpus=gpu_count,
    )

    # --- 3. EVALUATION LOGS ---
    print("\n=== Leaderboard (val) ===")
    print(predictor.leaderboard(df, silent=True)[["model", "score_val", "pred_time_val", "fit_time"]])

    print("\n=== Full Evaluation on Training Data ===")
    eval_metrics = predictor.evaluate(df, silent=True)
    mae = abs(eval_metrics.get("mean_absolute_error", 0))
    r2 = eval_metrics.get("r2", 0)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R^2): {r2:.4f}")

    # --- 4. PREDICTION & COMPARISON TABLE ---
    Xall = df.drop(columns=[label])
    preds = predictor.predict(Xall)

    out = df.copy()
    out["pred_wait_time_ms"] = preds

    # Reorder columns: ID, Actual, Prediction first
    first_cols = ["wave_id", label, "pred_wait_time_ms"]
    other_cols = [c for c in out.columns if c not in first_cols]
    out = out[first_cols + other_cols]

    # Display Detailed Comparison Table on Terminal
    print("\n=== Detailed Comparison: Actual vs Predicted (Top 20) ===")
    comparison = out[first_cols].copy()
    comparison["error_ms"] = comparison["pred_wait_time_ms"] - comparison[label]
    print(comparison.head(20).to_string(index=False, formatters={
        label: '{:.4f}'.format, 
        'pred_wait_time_ms': '{:.4f}'.format,
        'error_ms': '{:.4f}'.format
    }))

    # --- 5. SAVE OUTPUTS ---
    os.makedirs("data/processed/train", exist_ok=True)
    os.makedirs("data/processed/prediction", exist_ok=True)

    train_out_path = f"data/processed/train/train_with_predictions_{ts}.csv"
    out.to_csv(train_out_path, index=False)

    summary = out.groupby("wave_id", as_index=False)["pred_wait_time_ms"].mean().sort_values("wave_id")
    summary_out_path = f"data/processed/prediction/pred_wait_by_wave_id_{ts}.csv"
    summary.to_csv(summary_out_path, index=False)

    print(f"\nSaved Summary: {summary_out_path}")
    print(f"Saved Detailed: {train_out_path}")

    predictor.save()
    print("\nDone. Predictor saved at:", predictor.path)

def predict(
    model_path: str,
    input_csv: str,
    out_csv: str = "predictions.csv",
) -> None:
    """
    Load model and predict with ordered columns for inference.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model path not found: "{model_path}"')
    
    predictor = TabularPredictor.load(model_path)
    df = pd.read_csv(input_csv)

    # Drop metadata
    cols_to_drop_found = [c for c in COLS_TO_DROP if c in df.columns]
    if cols_to_drop_found:
        df = df.drop(columns=cols_to_drop_found)

    preds = predictor.predict(df)
    out = df.copy()
    out["pred_wait_time_ms"] = preds

    # Reorder for inference
    first_cols = ["wave_id", "pred_wait_time_ms"]
    other_cols = [c for c in out.columns if c not in first_cols]
    out = out[first_cols + other_cols]

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote predictions to: {out_csv}")

# def predict(
#     model_path: str,
#     input_csv: str,
#     out_csv: str = "predictions.csv",
# ) -> None:
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f'Model path not found: "{model_path}"')
#     if not os.path.exists(input_csv):
#         raise FileNotFoundError(f'Input file not found: "{input_csv}"')

#     predictor = TabularPredictor.load(model_path)
#     df = pd.read_csv(input_csv)

#     # Drop metadata columns before prediction to match training features
#     cols_to_drop_found = [c for c in COLS_TO_DROP if c in df.columns]
#     if cols_to_drop_found:
#         df = df.drop(columns=cols_to_drop_found)

#     preds = predictor.predict(df)
#     out = df.copy()
#     out["pred_wait_time_ms"] = preds

#     # สำหรับ Inference: เอา ID และ ค่าทำนาย ไว้หน้าสุด
#     first_cols = ["wave_id", "pred_wait_time_ms"]
#     other_cols = [c for c in out.columns if c not in first_cols]
#     out = out[first_cols + other_cols]

#     os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
#     out.to_csv(out_csv, index=False)
#     print(f"Wrote predictions: {out_csv}")

def main():
    ap = argparse.ArgumentParser(description="AutoGluon training/prediction")
    ap.add_argument("--mode", default="train", choices=["train", "predict"])

    # Train args
    ap.add_argument("--data", default="data/processed/train/train_features.csv")
    ap.add_argument("--label", default="wait_time_ms")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--presets", default="medium_quality")
    ap.add_argument("--time-limit", type=int, default=60)

    # Predict args
    ap.add_argument("--model-path", default="AutogluonModels")
    ap.add_argument("--inference-csv", default="data/processed/inference/train_features_1000_x.csv")
    ap.add_argument("--out", default="data/processed/prediction/predicted_wait_time_1000_x.csv")

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
            predict(
                model_path=args.model_path,
                input_csv=args.inference_csv,
                out_csv=args.out,
            )
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()