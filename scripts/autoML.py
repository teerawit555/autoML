from __future__ import annotations
from datetime import datetime
from pathlib import Path
import argparse
import os
import sys
import pandas as pd
import torch

from autogluon.tabular import TabularPredictor

# Meta columns ที่เราจะไม่ใช้เทรน (เพื่อให้ Model โฟกัสที่ลักษณะคลื่น)
COLS_TO_DROP = ['force_mA', 'range_V', 'temp_C','wave_id']
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_SAVE_PATH = f"AutogluonModels/ag-{ts}"

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
    
    # --- 1. PREPROCESSING ---
    cols_to_drop_found = [c for c in COLS_TO_DROP if c in df.columns]
    if cols_to_drop_found:
        print(f"Dropping meta columns: {cols_to_drop_found}")
        df = df.drop(columns=cols_to_drop_found)

    df = df.dropna(subset=[label]).reset_index(drop=True)
    
    save_path = model_dir or DEFAULT_SAVE_PATH
    gpu_count = 1 if torch.cuda.is_available() else 0
    print(f"🚀 Training device: {'GPU (CUDA)' if gpu_count > 0 else 'CPU'}")

    # --- 2. FIT MODEL ---
    predictor = TabularPredictor(
        label=label,
        path=save_path,
        problem_type="regression",
        eval_metric="mean_absolute_error",
        verbosity=2,
    ).fit(
        train_data=df,
        presets=presets,
        time_limit=time_limit,
        num_gpus=gpu_count,
    )

    # --- 3. MODEL ANALYSIS (ข้อมูลสำหรับ ADJUST MODEL) ---
    print("\n" + "="*60)
    print("🔍 DEEP MODEL ANALYSIS & DIAGNOSIS")
    print("="*60)

    # A. Feature Importance: ดูว่า AI ใช้ฟีเจอร์ไหนตัดสินใจ (สำคัญมากในการ Adjust)
    print("\n[1] Calculating Feature Importance...")
    importance = predictor.feature_importance(df)
    print(importance.head(15)) # โชว์ 15 อันดับแรก

    # B. Leaderboard: ดูว่าอัลกอริทึมตัวไหนฉลาดที่สุด
    print("\n[2] Model Leaderboard:")
    leaderboard = predictor.leaderboard(df, silent=True)
    print(leaderboard[["model", "score_val", "pred_time_val", "fit_time"]].head(5))

    # C. Residual Analysis: หาจุดที่ทายพลาด (Worst Case)
    X_test = df.drop(columns=[label])
    y_actual = df[label]
    y_pred = predictor.predict(X_test)

    out = df.copy()
    out["pred_wait_time_ms"] = y_pred
    out["error_ms"] = out["pred_wait_time_ms"] - y_actual
    out["abs_error_ms"] = out["error_ms"].abs()

    # ดึง 10 อันดับที่ AI ทายพลาดที่สุดมาโชว์
    worst_10 = out.sort_values(by="abs_error_ms", ascending=False).head(10)
    print("\n[3] TOP 10 WORST PREDICTIONS (Review these Wave IDs!):")
    print(worst_10[["wave_id", label, "pred_wait_time_ms", "error_ms"]])

    # --- 4. SAVE DIAGNOSIS DATA ---
    os.makedirs("data/processed/analysis", exist_ok=True)
    os.makedirs("data/processed/train", exist_ok=True)

    # ไฟล์ Diagnosis: รวม Features + Actual + Pred + Error (เอาไว้ Adjust ฟีเจอร์)
    diag_path = f"data/processed/analysis/diagnosis_report_{ts}.csv"
    out.to_csv(diag_path, index=False)

    # ไฟล์ Feature Importance: เก็บไว้ดูว่าตัวไหนไม่มีประโยชน์จะได้ตัดออก
    feat_imp_path = f"data/processed/analysis/feature_importance_{ts}.csv"
    importance.to_csv(feat_imp_path)

    # ไฟล์ผลลัพธ์ปกติ
    train_out_path = f"data/processed/train/train_with_predictions_{ts}.csv"
    out.drop(columns=["abs_error_ms"]).to_csv(train_out_path, index=False)

    print(f"\n✅ Analysis Report Saved: {diag_path}")
    print(f"✅ Feature Importance Saved: {feat_imp_path}")
    print(f"✅ Model saved at: {save_path}")
    print("="*60)

def predict(
    model_path: str,
    input_csv: str,
    out_csv: str = "predictions.csv",
) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found at: "{model_path}"')
    
    print(f"🔮 Loading model and predicting: {input_csv}")
    predictor = TabularPredictor.load(model_path)
    df = pd.read_csv(input_csv)

    # Clean meta columns
    for c in COLS_TO_DROP:
        if c in df.columns:
            df = df.drop(columns=[c])

    preds = predictor.predict(df)
    out = df.copy()
    out["pred_wait_time_ms"] = preds

    # Reorder columns ให้ ID และผลทายอยู่หน้าสุด
    first_cols = ["wave_id", "pred_wait_time_ms"]
    other_cols = [c for c in out.columns if c not in first_cols]
    out = out[first_cols + other_cols]

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"✅ Prediction Results saved: {out_csv}")

def main():
    ap = argparse.ArgumentParser(description="AutoGluon Workflow with Analysis")
    ap.add_argument("--mode", default="train", choices=["train", "predict"])
    ap.add_argument("--data", default="data/processed/train/train_features.csv")
    ap.add_argument("--label", default="wait_time_ms")
    ap.add_argument("--model-path", default=None) # สำหรับโหมด predict
    ap.add_argument("--inference-csv", default="data/processed/inference/features_test_2.csv")
    ap.add_argument("--out", default="data/processed/prediction/final_results.csv")
    ap.add_argument("--time-limit", type=int, default=120) # เพิ่มเวลาเทรนเริ่มต้น
    ap.add_argument("--presets", default="medium_quality")

    args = ap.parse_args()

    try:
        if args.mode == "train":
            train(
                data_path=args.data,
                label=args.label,
                presets=args.presets,
                time_limit=args.time_limit
            )
        else:
            if not args.model_path:
                print("❌ Error: Please specify --model-path for prediction mode.")
                return
            predict(
                model_path=args.model_path,
                input_csv=args.inference_csv,
                out_csv=args.out
            )
    except Exception as e:
        print(f"💥 ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()