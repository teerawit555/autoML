from pathlib import Path
import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt

# BASE_DIR = Path(__file__).resolve().parent.parent
# RAW_DIR = BASE_DIR / "data" / "raw"
# PROC_DIR = BASE_DIR / "data" / "processed"
# PLOT_DIR = BASE_DIR / "plots" / "waves"

def main():
    ap = argparse.ArgumentParser(description="Plot raw waveforms with predicted wait time.")
    ap.add_argument("--raw", default="../data/raw/data1000samples_test.csv", help="Path to the original long CSV (data1000samples_test.csv)")
    ap.add_argument("--pred", default="../data/processed/prediction/predicted_wait_time_1000.csv", help="Path to the prediction CSV file (e.g., predicted_wait_time_1000.csv)")
    ap.add_argument("--out", default="../plots/waves/waves_with_pred_wait.png", help="Output path for the plot image")
    ap.add_argument("--ncols", type=int, default=4, help="Number of columns in the subplot grid")
    ap.add_argument("--sharey", action="store_true", help="Share Y-axis across all plots")
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    pred = pd.read_csv(args.pred)

    # --- sanity check columns ---
    # Edit: Changed 'current' to 'value'
    for c in ["wave_id", "time", "value"]:
        if c not in raw.columns:
            # Use 'time' instead of 'time_ms' (based on the provided raw data structure)
            raise ValueError(f"raw file missing column: {c}")

    # Must contain 'wave_id' + 'pred_wait_time_ms' in pred
    if "pred_wait_time_ms" not in pred.columns:
        raise ValueError("pred file missing column: pred_wait_time_ms")

    if "wave_id" not in pred.columns:
        raise ValueError("pred file missing column: wave_id (see note below)")

    # Map predicted wait time per wave
    # Use average of pred_wait_time_ms per wave_id if there are multiple rows per wave
    # (Case where pred comes from train_with_predictions which has multiple rows per wave_id)
    # However, since the prediction file comes from features.csv which has 1 row per wave_id, we use .iloc[0]
    
    # Warning: Prediction file (predicted_wait_time_1000.csv) already has 1 row per wave_id
    # But if using train_with_predictions.csv (from train mode), there will be multiple rows
    # For safety, we use .iloc[0] because features file has 1 row per wave_id
    pred_map = dict(zip(pred["wave_id"], pred["pred_wait_time_ms"]))

    wave_ids = sorted(raw["wave_id"].unique())
    n = len(wave_ids)
    ncols = args.ncols
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.5 * ncols, 3.2 * nrows),
        sharex=True,
        sharey=args.sharey
    )
    # Handle case with only 1 plot or multiple plots
    axes = axes.flatten() if n > 1 else [axes] if n == 1 else []

    for i, wid in enumerate(wave_ids):
        ax = axes[i]
        sub = raw[raw["wave_id"] == wid].sort_values("time")

         # Edit: Use "value" instead of "current" and add "Signal" label
        ax.plot(sub["time"], sub["value"], linewidth=1, label='Signal')
        ax.set_title(f"wave_id={wid}")

        ax.set_xlabel("time (ms)", fontsize=9)
        ax.set_ylabel("value", fontsize=9)

        # Predicted wait line
        if wid in pred_map:
            t_pred = float(pred_map[wid])
            ax.axvline(t_pred, color='orange', linestyle="--", linewidth=1.5, label='AI Predict')

            # Edit: Changed "left" to "right"
            ax.text(t_pred, sub["value"].max(), f" pred={t_pred:.2f}ms",
                    rotation=90, va="top", ha="right", fontsize=9, color='r')
        else:
            ax.text(0.02, 0.90, "no pred", transform=ax.transAxes, fontsize=9)

        ax.grid(True, alpha=0.3)

    # Remove remaining subplots (if any)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    # Edit: Added h_pad=3.0, w_pad=2.0 as parameters in tight_layout
    fig.tight_layout(h_pad=3.0, w_pad=2.0) 
    fig.savefig(args.out, dpi=200)
    print(f"saved: {args.out}")

if __name__ == "__main__":
    main()