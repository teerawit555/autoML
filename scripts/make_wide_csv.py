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
    sample_idx_col: str = "sample_idx",
    value_col: str = "current",
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

    # Keep meta columns (anything that is not id/sample/value/label) -> take first per wave
    meta_cols = [c for c in df.columns if c not in {id_col, sample_idx_col, value_col, label_col}]

    # Pivot to wide i_0..i_99
    wide = df.pivot(index=id_col, columns=sample_idx_col, values=value_col)

    # Rename columns to i_{k}
    wide.columns = [f"i_{int(c)}" for c in wide.columns]
    wide = wide.reset_index()

    # Attach meta columns
    if meta_cols:
        meta_first = df.groupby(id_col, as_index=False)[meta_cols].first()
        wide = wide.merge(meta_first, on=id_col, how="left")

    # Attach label (first per wave)
    if has_label:
        y_first = df.groupby(id_col, as_index=False)[[label_col]].first()
        wide = wide.merge(y_first, on=id_col, how="left")

    # Sort i_ columns if present
    i_cols = [c for c in wide.columns if c.startswith("i_")]
    i_cols_sorted = sorted(i_cols, key=lambda s: int(s.split("_")[1]))
    other_cols = [c for c in wide.columns if c not in i_cols]
    wide = wide[other_cols[:1] + i_cols_sorted + other_cols[1:]]  # keep wave_id first

    wide.to_csv(out_path, index=False)
    print(f"✅ Wrote wide CSV: {out_path}")
    print(f"   Rows(waves): {len(wide)} | i_cols: {len(i_cols_sorted)} | meta_cols: {len(meta_cols)} | label: {has_label}")


def main():
    ap = argparse.ArgumentParser(description="Convert long-format waveform CSV to wide-format (i_0..i_N).")
    ap.add_argument("--in", dest="in_path", default="../data/raw/raw_long.csv", help="Input long CSV path (default: raw_long.csv)")
    ap.add_argument("--out", dest="out_path", default="../data/processed/wide.csv", help="Output wide CSV path (default: wide.csv)")
    ap.add_argument("--id-col", default="wave_id")
    ap.add_argument("--sample-idx-col", default="sample_idx")
    ap.add_argument("--value-col", default="current")
    ap.add_argument("--label-col", default="wait_time_ms")
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
        print(f"❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
