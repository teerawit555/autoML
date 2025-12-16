# make_wide_csv.py
from __future__ import annotations

import argparse
import os
import sys
import pandas as pd

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
    print(f"✅ Wrote wide CSV: {out_path}")
    print(f"   Rows(waves): {len(wide)} | i_cols: {len(i_cols_sorted)} | meta_cols: {len(meta_cols)} | label: {has_label}")


def main():
    ap = argparse.ArgumentParser(description="Convert long-format waveform CSV to wide-format (i_0..i_N).")
    # ปรับ Default Input และ Value Col
    ap.add_argument("--in", dest="in_path", default="../data/raw/data1000samples_test.csv", help="Input long CSV path (default: data1000samples_test.csv)")
    ap.add_argument("--out", dest="out_path", default="../data/processed/inference/wide.csv", help="Output wide CSV path (default: wide.csv)")
    ap.add_argument("--id-col", default="wave_id")
    ap.add_argument("--sample-idx-col", default="sample") 
    ap.add_argument("--value-col", default="value")
    ap.add_argument("--label-col", default="wait_time_ms", help="Label column name (used for exclusion from features)") # เปลี่ยนคำอธิบาย
    args = ap.parse_args()

    try:
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

if __name__ == "__main__":
    main()