# import pandas as pd
# from autogluon.tabular import TabularPredictor

# # =========================
# # (1) Load data
# # =========================
# df = pd.read_csv("settling_demo.csv")

# # label = คำตอบที่เราอยากให้โมเดลทำนาย
# LABEL = "wait_time_ms"

# # =========================
# # (2) Split train/test
# # =========================
# # แนวคิด: แยกข้อมูลบางส่วนไว้ "ทดสอบ" เพื่อประเมินว่าโมเดลไม่ใช่แค่ท่องจำ
# train_df = df.sample(frac=0.8, random_state=42)
# test_df  = df.drop(train_df.index)

# # =========================
# # (3) Train AutoML
# # =========================
# # AutoGluon จะลองหลายโมเดล + จูนพารามิเตอร์ + เลือกตัวที่ดีที่สุดให้
# predictor = TabularPredictor(
#     label=LABEL,
#     problem_type="regression",
#     eval_metric="mean_absolute_error"  # MAE: พลาดเฉลี่ยกี่ ms
# ).fit(
#     train_data=train_df,
#     time_limit=60,      # เดโม: ให้ลอง 60 วินาที (งานจริงปรับเป็น 5-30 นาทีได้)
#     presets="medium_quality"
# )

# # =========================
# # (4) Evaluate on test set
# # =========================
# print("\n=== Evaluation on TEST ===")
# metrics = predictor.evaluate(test_df)
# print(metrics)

# # =========================
# # (5) Leaderboard: โมเดลไหนชนะ
# # =========================
# print("\n=== Leaderboard ===")
# lb = predictor.leaderboard(test_df, silent=True)
# print(lb[["model", "score_val", "pred_time_val", "fit_time"]].head(10))

# # =========================
# # (6) Feature importance: ฟีเจอร์ไหนสำคัญ
# # =========================
# print("\n=== Feature Importance ===")
# fi = predictor.feature_importance(test_df)
# print(fi.head(15))

# # =========================
# # (7) Predict example
# # =========================
# example = pd.DataFrame([{
#     "force_mA": 2,
#     "range_V": 10,
#     "temp_C": 25,
#     "overshoot_V": 0.090,
#     "undershoot_V": 0.065,
#     "max_slope_V_per_ms": 0.75,
#     "ringing_energy": 1.00,
#     "std_last10": 0.0032,
#     "mean_last10": 1.000,
# }])

# pred_wait = predictor.predict(example)[0]
# print(f"\nPredicted wait_time_ms = {pred_wait:.2f} ms")

# # =========================
# # (8) Save model
# # =========================
# predictor.save("settling_wait_model")
# print("\nModel saved to folder: settling_wait_model")

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

    # Drop rows with missing label
    df = df.dropna(subset=[label]).reset_index(drop=True)
    if len(df) < 10:
        print(f"⚠️ Warning: very small dataset ({len(df)} rows). Metrics may look unstable/overfit.")

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
    fi = predictor.feature_importance(df, num_shuffle_sets=5, silent=True)
    print(fi.head(30))

    # ===== PREDICT ALL TRAIN DATA & SAVE =====
    sample_row = df.drop(columns=[label]).iloc[[0]]  # เอาแถวแรกเป็นตัวอย่าง
    pred_ms = predictor.predict(sample_row).iloc[0]
    print(f"\n✅ Predicted wait_time_ms (example row) = {pred_ms:.2f} ms\n")

    Xall = df.drop(columns=[label])
    preds = predictor.predict(Xall)

    out = df.copy()
    out["pred_wait_time_ms"] = preds
    out.to_csv(f"data/processed/train_with_predictions_{ts}.csv", index=False)

    # ✅ สรุปเป็นราย wave_id (เอาค่า pred ของแต่ละ wave_id)
    # ถ้า 1 wave_id มี 1 แถวอยู่แล้ว ผลจะตรงๆ
    summary = (
        out.groupby("wave_id", as_index=False)["pred_wait_time_ms"]
        .mean()
        .sort_values("wave_id")
    )

    summary.to_csv(f"data/processed/pred_wait_by_wave_id_{ts}.csv", index=False)

    print("\n=== Predicted wait_time_ms by wave_id ===")
    print(summary.to_string(index=False))
    print("\n✅ Saved file: data/processed/pred_wait_by_wave_id.csv")

    print("\n✅ Saved file: data/processed/train_with_predictions.csv")
    # print(out[["wait_time_ms", "pred_wait_time_ms"]].head(10))
    predictor.save()
    print("\n✅ Done. Predictor saved at:", predictor.path)


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
    print(f"✅ Wrote predictions: {out_csv}")


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
        print(f"❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
