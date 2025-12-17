# make_wide_csv.py
from __future__ import annotations

import argparse
import os
import sys
import pandas as pd
import numpy as np

def make_wide(
    in_path: str,
    out_path: str,
    id_col: str = "wave_id",
    # *** แก้ไข: ใช้ "sample" เป็น Default ***
    sample_idx_col: str = "sample",
    # *** แก้ไข: ใช้ "value" เป็น Default ***
    value_col: str = "value",
    label_col: str = "wait_time_ms",
) -> None:
    
    """ Original wide conversion logic for inference data """
    
    if not os.path.exists(in_path):
        raise FileNotFoundError(f'Input file not found: "{in_path}"')

    df = pd.read_csv(in_path)

    required = {id_col, sample_idx_col, value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input: {sorted(missing)}")

    # label is optional (for inference), but recommended
    has_label = label_col in df.columns

    # Identify meta columns (anything not id/sample/value/label)
    cols_to_exclude = {id_col, sample_idx_col, value_col, label_col}
    meta_cols = [c for c in df.columns if c not in cols_to_exclude]

    # Pivot to wide i_0..i_N
    wide = df.pivot(index=id_col, columns=sample_idx_col, values=value_col)

    # Rename columns to i_{k}
    wide.columns = [f"i_{int(c)}" for c in wide.columns]
    wide = wide.reset_index()

    # --- Attach meta columns (sd, low limit, high limit, time, force_mA, etc.) ---
    if meta_cols:
        # Groupby และเอาค่าแรกของคอลัมน์ meta data
        meta_first = df.groupby(id_col, as_index=False)[meta_cols].first()
        wide = wide.merge(meta_first, on=id_col, how="left")

    # --- Attach label (wait_time_ms) ---
    if has_label:
        y_first = df.groupby(id_col, as_index=False)[[label_col]].first()
        wide = wide.merge(y_first, on=id_col, how="left")

    # --- ปรับปรุงตรรกะการจัดเรียงคอลัมน์ (แก้ไขปัญหา KeyError/IndexError) ---
    i_cols = [c for c in wide.columns if c.startswith("i_")]
    i_cols_sorted = sorted(i_cols, key=lambda s: int(s.split("_")[1]))
    
    # สร้างรายการคอลัมน์สุดท้าย
    final_cols = [id_col]              # 1. wave_id (ID Column)
    final_cols.extend(i_cols_sorted)   # 2. i_0, i_1, i_2, ... (Waveform Data)
    
    # 3. Meta Data และ Label ที่เหลือ (ต้องไม่ซ้ำกับ ID หรือ i_cols)
    remaining_cols = [c for c in wide.columns if c not in final_cols]
    final_cols.extend(remaining_cols)
    
    wide = wide[final_cols] # จัดเรียง DataFrame ด้วยรายการคอลัมน์ใหม่

    wide.to_csv(out_path, index=False)
    print(f" Wrote wide CSV: {out_path}")
    print(f"Rows(waves): {len(wide)} | i_cols: {len(i_cols_sorted)} | meta_cols: {len(meta_cols)} | label: {has_label}")

def extract_features_and_label(group):
    """
    This function processes each wave_id to extract both features and 
    the settling time (wait_time_ms) as a label.
    """
    values = group['value'].values # Your new column name
    times = group['time_ms'].values
    
    # --- 1. SETTLING TIME CALCULATION (LABEL) ---
    last_10_pct_idx = max(1, int(len(values) * 0.1))
    mean_last = np.mean(values[-last_10_pct_idx:])
    std_last = np.std(values[-last_10_pct_idx:])
    
    # Target value and tolerance for labeling
    tolerance = abs(mean_last * 0.01) # 1% Threshold
    
    settle_idx = len(values) - 1
    for i in range(len(values) - 1, -1, -1):
        if abs(values[i] - mean_last) > tolerance:
            settle_idx = i + 1
            break
    wait_time_ms = times[min(settle_idx, len(times)-1)]

    # --- 2. ADVANCED FEATURE EXTRACTION ---
    # Slopes
    slopes = np.diff(values) if len(values) > 1 else [0]
    
    # Ringing Energy (Sum of squared differences from the final mean)
    ringing_energy = np.sum((values[-last_10_pct_idx:] - mean_last)**2)
    
    # Band 3-Sigma (3 times the standard deviation of the last portion)
    band_3std_last = 3 * std_last

    # --- 2. FEATURE EXTRACTION ---
    # Add all your existing feature calculations here
    features = {
        'wave_id': group['wave_id'].iloc[0],
        'time': times[0],                     # Start time
        'sd': np.std(values),                 # Standard deviation (same as std_all)
        'low_limit': mean_last - tolerance,    # Dynamic low limit
        'high_limit': mean_last + tolerance,   # Dynamic high limit
        'x0': values[0],                      # Initial value
        'x_end': values[-1],                  # Final value
        'mean_all': np.mean(values),
        'std_all': np.std(values),
        'mean_last': mean_last,
        'std_last': std_last,
        'peak_rel': np.max(values) - mean_last,
        'trough_rel': mean_last - np.min(values),
        'max_slope': np.max(slopes),
        'min_slope': np.min(slopes),
        'ringing_energy': ringing_energy,
        'settle_idx': settle_idx,             # The index where it settled
        'band_3std_last': band_3std_last,
        'wait_time_ms': wait_time_ms          # Target Label for Training
    }
    return pd.Series(features)

def main():

    ap = argparse.ArgumentParser(description="Convert long-format waveform CSV to wide-format (i_0..i_N).")
    # ปรับ Default Input และ Value Col
    ap.add_argument("--mode", default="train", choices=["train", "inference"], help="Processing mode")
    ap.add_argument("--in", dest="in_path", default="data/raw/data1000samples_test.csv", help="Input long CSV path (default: data1000samples_test.csv)")
    ap.add_argument("--out", dest="out_path", default="data/processed/inference/wide.csv", help="Output wide CSV path (default: wide.csv)")
    ap.add_argument("--id-col", default="wave_id")
    ap.add_argument("--sample-idx-col", default="sample") 
    ap.add_argument("--value-col", default="value")
    ap.add_argument("--label-col", default="wait_time_ms", help="Label column name (used for exclusion from features)") # เปลี่ยนคำอธิบาย
    args = ap.parse_args()

    try:
            # --- MODE 1: TRAIN (Extract Stats + Automated Label) ---
        if args.mode == "train":
            print(f"Reading raw data for training: {args.in_path}")
            df_raw = pd.read_csv(args.in_path)
            
            print("Extracting features and calculating settling times (Labeling)...")
            # Group by wave_id and apply the automated logic
            train_features = df_raw.groupby(args.id_col, group_keys=False).apply(extract_features_and_label).reset_index(drop=True)
            
            os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
            train_features.to_csv(args.out_path, index=False)
            print(f"Done! Training features saved to: {args.out_path}")

        # --- MODE 2: INFERENCE (Convert to i_0...i_N for Predictor) ---
        else:
            print(f"Converting to wide format for inference: {args.in_path}")
            make_wide(
                in_path=args.in_path,
                out_path=args.out_path,
                id_col=args.id_col,
                sample_idx_col=args.sample_idx_col,
                value_col=args.value_col,
                label_col=args.label_col,
            )
                
    except Exception as e:
        print(f" ERROR: {e}")
        sys.exit(1)

    # Load raw data
    df_raw = pd.read_csv('data/raw/data_for_train.csv')
    
    # Process each wave_id
    print("Extracting features and calculating settling times...")
    # Group by wave_id and apply the function
    train_features = df_raw.groupby('wave_id').apply(extract_features_and_label).reset_index(drop=True)
    
    # Save to the final training file
    output_path = 'data/processed/train/train_features.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    train_features.to_csv(output_path, index=False)
    print(f"Done! File saved to: {output_path}")

if __name__ == "__main__":
    main()

#------ 1 Data for training (with labels) ------#
# python scripts/make_wide_csv.py --mode train --in data/raw/data_for_train.csv --out data/processed/train/train_features.csv
#------ 2 Data for inference (without labels) ------#
# python scripts/make_wide_csv.py --mode inference --in data/raw/data_1000_samples_to_pred.csv --out data\processed\inference\wide_1000_samples_to_pred.csv