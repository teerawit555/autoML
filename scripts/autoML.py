# autoML.py
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import argparse
import os
import sys
import pandas as pd

from autogluon.tabular import TabularPredictor

from datetime import datetime

COLS_TO_DROP = ['force_mA', 'range_V', 'temp_C']

#run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

save_path = f"../models/AutogluonModels/ag-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def train(
    data_path: str,
    label: str = "wait_time_ms",
    model_dir: str | None = None,
    presets: str = "medium_quality",
    time_limit: int = 60,
) -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Input file not found: "{data_path}"')

    df = pd.read_csv(data_path)
    if label not in df.columns:
        raise ValueError(f'Label column "{label}" not found in {data_path}')

    # --- NEW LOGIC: DROP COLUMNS BEFORE TRAINING ---
    cols_to_drop_found = [c for c in COLS_TO_DROP if c in df.columns]
    
    if cols_to_drop_found:
        print(f"⚠️ Dropping columns not present in inference data: {cols_to_drop_found}")
        df = df.drop(columns=cols_to_drop_found)
    else:
        print(f"✅ Training data does not contain excluded Meta Data columns: {COLS_TO_DROP}")
    # --- END NEW LOGIC ---

    # Drop rows with missing label
    df = df.dropna(subset=[label]).reset_index(drop=True)
    if len(df) < 10:
        print(f" Warning: very small dataset ({len(df)} rows). Metrics may look unstable/overfit.")

    save_path = model_dir or "AutogluonModels"
    print(f'No path specified. Models will be saved in: "{save_path}"')

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
    )

    print("\n=== Leaderboard (val) ===")
    print(predictor.leaderboard(df, silent=True)[["model", "score_val", "pred_time_val", "fit_time"]])

    print("\n=== Feature Importance (quick) ===")
    # For tiny datasets, this can be noisy; still useful as a sanity check
    fi = predictor.feature_importance(df, num_shuffle_sets=5, silent=True) # Permutation Importance
    print(fi.head(30))

    # ===== NEW: Calculate and print evaluation metrics (MAE, RMSE, R^2) =====
    print("\n=== Full Evaluation on Training Data (for reference) ===")
    # Evaluate calculates standard regression metrics like MAE, RMSE, and R^2.
    # Note: AutoGluon returns evaluation metrics that are *always* scores to be maximized.
    # For error metrics (like MAE, RMSE), this means they are returned as negative numbers.
    eval_metrics = predictor.evaluate(df, silent=True)

    # Use the absolute value for error metrics for human readability.
    mae = abs(eval_metrics.get("mean_absolute_error", float("nan")))
    rmse = abs(eval_metrics.get("root_mean_squared_error", float("nan")))
    r2 = eval_metrics.get("r2", float("nan"))

    print(f"Mean Absolute Error (MAE): \t\t{mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): \t{rmse:.4f}")
    print(f"R-squared (R^2): \t\t\t{r2:.4f}")
    print("-" * 50) # margin

    # ===== PREDICT ALL TRAIN DATA & SAVE =====
    sample_row = df.drop(columns=[label]).iloc[[0]]  # เอาแถวแรกเป็นตัวอย่าง
    pred_ms = predictor.predict(sample_row).iloc[0]
    print(f"\n Predicted wait_time_ms (example row) = {pred_ms:.2f} ms\n")

    Xall = df.drop(columns=[label])
    preds = predictor.predict(Xall)

    out = df.copy()
    out["pred_wait_time_ms"] = preds
    out.to_csv(f"data/processed/train/train_with_predictions_{ts}.csv", index=False)

    # สรุปเป็นราย wave_id (เอาค่า pred ของแต่ละ wave_id)
    # ถ้า 1 wave_id มี 1 แถวอยู่แล้ว ผลจะตรงๆ
    summary = (
        out.groupby("wave_id", as_index=False)["pred_wait_time_ms"]
        .mean()
        .sort_values("wave_id")
    )

    summary.to_csv(f"data/processed/prediction/pred_wait_by_wave_id_{ts}.csv", index=False)

    print("\n=== Predicted wait_time_ms by wave_id ===")
    print(summary.to_string(index=False))
    print("\n Saved file: data/processed/prediction/pred_wait_by_wave_id.csv") # Result wave_id vs pred_wait_time_ms

    print("\n Saved file: data/processed/train/train_with_predictions.csv")
    # print(out[["wait_time_ms", "pred_wait_time_ms"]].head(10))
    predictor.save()
    print("\n Done. Predictor saved at:", predictor.path)

def predict(
    model_path: str,
    input_csv: str,
    out_csv: str = "predictions.csv",
) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model path not found: "{model_path}"')
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f'Input file not found: "{input_csv}"')

    predictor = TabularPredictor.load(model_path)
    df = pd.read_csv(input_csv)

    preds = predictor.predict(df)
    out = df.copy()
    out["pred_wait_time_ms"] = preds

    out.to_csv(out_csv, index=False)
    print(f"Wrote predictions: {out_csv}")

def main():
    ap = argparse.ArgumentParser(description="AutoGluon training/prediction for settling wait time from waveform samples.")
    sub = ap.add_subparsers(dest="cmd", required=False)

    ap.add_argument("--mode", default="train", choices=["train", "predict"], help="train or predict (default: train)")

    # Train args
    ap.add_argument("--data", default="../data/processed/train_features.csv", help="Training CSV (default: train_features.csv)")
    ap.add_argument("--label", default="wait_time_ms", help="Label column (default: wait_time_ms)")
    ap.add_argument("--model-dir", default=None, help='Model output folder (default: "AutogluonModels")')
    ap.add_argument("--presets", default="medium_quality", help="AutoGluon presets (default: medium_quality)")
    ap.add_argument("--time-limit", type=int, default=60, help="Time limit seconds (default: 60)")

    # Predict args
    ap.add_argument("--model-path", default="AutogluonModels", help="Path to trained model folder")
    ap.add_argument("--inference-csv", default="inference_features.csv", help="CSV for prediction")
    ap.add_argument("--out", default="predictions.csv", help="Output predictions CSV")

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
