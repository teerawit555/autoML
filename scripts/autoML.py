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
    # [จุดที่แก้] สำรอง wave_id ไว้ก่อนที่จะถูก drop
    wave_id_backup = df['wave_id'].copy() if 'wave_id' in df.columns else None

    cols_to_drop_found = [c for c in COLS_TO_DROP if c in df.columns]
    if cols_to_drop_found:
        print(f"Dropping meta columns: {cols_to_drop_found}")
        # เรา drop ออกจาก df ที่จะใช้เทรนเท่านั้น
        df_train = df.drop(columns=cols_to_drop_found)
    else:
        df_train = df.copy()

    df_train = df_train.dropna(subset=[label]).reset_index(drop=True)
    
    save_path = model_dir or DEFAULT_SAVE_PATH
    gpu_count = 1 if torch.cuda.is_available() else 0
    print(f"🚀 Training device: {'GPU (CUDA)' if gpu_count > 0 else 'CPU'}")

    # --- 2. FIT MODEL ---
    # ใช้ df_train ในการ fit
    predictor = TabularPredictor(
        label=label,
        path=save_path,
        problem_type="regression",
        eval_metric="mean_absolute_error",
        verbosity=2,
    ).fit(
        train_data=df_train,
        presets=presets,
        time_limit=time_limit,
        num_gpus=gpu_count,
    )

    # --- 3. MODEL ANALYSIS ---
    print("\n" + "="*60)
    print("🔍 DEEP MODEL ANALYSIS & DIAGNOSIS")
    print("="*60)

    # A. Feature Importance
    print("\n[1] Calculating Feature Importance...")
    importance = predictor.feature_importance(df_train)
    print(importance.head(15))

    # B. Leaderboard
    print("\n[2] Model Leaderboard:")
    leaderboard = predictor.leaderboard(df_train, silent=True)
    print(leaderboard[["model", "score_val", "pred_time_val", "fit_time"]].head(5))

    # C. Residual Analysis
    X_test = df_train.drop(columns=[label])
    y_actual = df_train[label]
    y_pred = predictor.predict(X_test)

    # [จุดที่แก้] สร้าง DataFrame สำหรับ Report โดยเอา wave_id กลับมาใส่
    out = df_train.copy()
    if wave_id_backup is not None:
        # ดึงเฉพาะ id ที่ตรงกับ index ของข้อมูลที่ใช้เทรน (กรณีมีการ dropna)
        out["wave_id"] = wave_id_backup.iloc[df_train.index].values
        
    out["pred_wait_time_ms"] = y_pred
    out["error_ms"] = out["pred_wait_time_ms"] - y_actual
    out["abs_error_ms"] = out["error_ms"].abs()

    # [จุดที่แก้] ตรวจสอบคอลัมน์ก่อนพิมพ์ Worst 10
    print("\n[3] TOP 10 WORST PREDICTIONS (Review these Wave IDs!):")
    cols_to_show = ["wave_id", label, "pred_wait_time_ms", "error_ms"]
    # เช็คว่าถ้าไม่มี wave_id ใน out จริงๆ ให้ตัดออกจากลิสต์แสดงผลเพื่อไม่ให้พังอีก
    cols_to_show = [c for c in cols_to_show if c in out.columns]
    
    worst_10 = out.sort_values(by="abs_error_ms", ascending=False).head(10)
    print(worst_10[cols_to_show])

    # --- 4. SAVE DIAGNOSIS DATA ---
    os.makedirs("data/processed/analysis", exist_ok=True)
    os.makedirs("data/processed/train", exist_ok=True)

    diag_path = f"data/processed/analysis/diagnosis_report_{ts}.csv"
    out.to_csv(diag_path, index=False)

    feat_imp_path = f"data/processed/analysis/feature_importance_{ts}.csv"
    importance.to_csv(feat_imp_path)

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

    # [จุดที่แก้] สำรอง wave_id ไว้ก่อน เพื่อเอาไว้แปะคืนในไฟล์ผลลัพธ์
    wave_id_backup = df['wave_id'].copy() if 'wave_id' in df.columns else None

    # สร้าง DataFrame สำหรับส่งให้ AI ทาย (ต้องลบ meta columns และ ID ออก)
    df_for_pred = df.copy()
    for c in COLS_TO_DROP:
        if c in df_for_pred.columns:
            df_for_pred = df_for_pred.drop(columns=[c])

    # AI ทำการทายผล
    preds = predictor.predict(df_for_pred)
    
    # สร้าง DataFrame ผลลัพธ์
    out = df_for_pred.copy()
    if wave_id_backup is not None:
        out["wave_id"] = wave_id_backup # เอา ID กลับคืนมา

    out["pred_wait_time_ms"] = preds

    # Reorder columns ให้ ID และผลทายอยู่หน้าสุดเพื่อให้ดูง่าย
    # ตรวจสอบก่อนว่ามีคอลัมน์ที่จะย้ายไหมป้องกัน Error ซ้ำ
    first_cols = [c for c in ["wave_id", "pred_wait_time_ms"] if c in out.columns]
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