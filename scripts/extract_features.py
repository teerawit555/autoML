# extract_features.py
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import pandas as pd


def _get_i_cols(df: pd.DataFrame) -> list[str]:
    i_cols = [c for c in df.columns if c.startswith("i_")]
    i_cols.sort(key=lambda s: int(s.split("_")[1]))
    return i_cols


def compute_features_from_row(x: np.ndarray) -> dict:
    # x: shape (N,)
    N = x.shape[0]
    if N < 5:
        raise ValueError("Need at least 5 samples per waveform")

    # Basic
    x0 = float(x[0])
    x_end = float(x[-1])
    mean_all = float(np.mean(x))
    std_all = float(np.std(x))

    # Last 10
    k = min(10, N)
    last = x[-k:]
    mean_last = float(np.mean(last))
    std_last = float(np.std(last))

    # Peak / trough relative to final mean (proxy overshoot/undershoot but derived purely from samples)
    peak = float(np.max(x))
    trough = float(np.min(x))
    peak_rel = float(peak - mean_last)
    trough_rel = float(mean_last - trough)

    # Max slope (per sample step)
    dx = np.diff(x)
    max_slope = float(np.max(dx))
    min_slope = float(np.min(dx))

    # Ringing energy proxy: sum of absolute derivative after the first few samples
    ringing_energy = float(np.sum(np.abs(dx[3:])))

    # Settling index: first index where remaining window stays within band around mean_last
    # Band = max(3*std_last, small epsilon)
    band = max(3.0 * std_last, 1e-12)
    settle_idx = N - 1
    for i in range(N):
        rem = x[i:]
        if np.all(np.abs(rem - mean_last) <= band):
            settle_idx = i
            break

    return {
        "x0": x0,
        "x_end": x_end,
        "mean_all": mean_all,
        "std_all": std_all,
        "mean_last": mean_last,
        "std_last": std_last,
        "peak_rel": peak_rel,
        "trough_rel": trough_rel,
        "max_slope": max_slope,
        "min_slope": min_slope,
        "ringing_energy": ringing_energy,
        "settle_idx": settle_idx,
        "band_3std_last": band,
    }


def extract_features(in_path: str, out_path: str, label_col: str = "wait_time_ms") -> None:
    if not os.path.exists(in_path):
        raise FileNotFoundError(f'Input file not found: "{in_path}"')

    df = pd.read_csv(in_path)
    i_cols = _get_i_cols(df)
    if not i_cols:
        raise ValueError("No i_0..i_N columns found. Did you create wide.csv first?")

    X = df[i_cols].to_numpy(dtype=float)
    feats = [compute_features_from_row(X[i]) for i in range(X.shape[0])]
    feat_df = pd.DataFrame(feats)

    # Keep meta columns (non i_) and label if present
    keep_cols = [c for c in df.columns if not c.startswith("i_")]
    out = pd.concat([df[keep_cols].reset_index(drop=True), feat_df], axis=1)

    out.to_csv(out_path, index=False)
    print(f"✅ Wrote feature CSV: {out_path}")
    print(f"   Rows: {len(out)} | i_cols: {len(i_cols)} | label_present: {label_col in out.columns}")


def main():
    ap = argparse.ArgumentParser(description="Extract ML features from wide waveform CSV (i_0..i_N).")
    ap.add_argument("--in", dest="in_path", default="../data/processed/wide.csv", help="Input wide CSV (default: wide.csv)")
    ap.add_argument("--out", dest="out_path", default="../data/processed/train_features.csv", help="Output CSV (default: train_features.csv)")
    ap.add_argument("--label-col", default="wait_time_ms")
    args = ap.parse_args()

    try:
        extract_features(args.in_path, args.out_path, label_col=args.label_col)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
