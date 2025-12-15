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
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../data/raw/raw_long.csv")
    ap.add_argument("--pred", default="../data/processed/train_with_predictions.csv")
    ap.add_argument("--out", default="../plots/waves/waves_with_pred_wait.png")
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--sharey", action="store_true")
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    pred = pd.read_csv(args.pred)

    # --- sanity check columns ---
    for c in ["wave_id", "time_ms", "current"]:
        if c not in raw.columns:
            raise ValueError(f"raw file missing column: {c}")

    # ต้องมี wave_id + pred_wait_time_ms ใน pred
    if "pred_wait_time_ms" not in pred.columns:
        raise ValueError("pred file missing column: pred_wait_time_ms")

    if "wave_id" not in pred.columns:
        raise ValueError("pred file missing column: wave_id (ดูหมายเหตุด้านล่าง)")

    # map predicted wait time ต่อ wave
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
    axes = axes.flatten() if n > 1 else [axes]

    for i, wid in enumerate(wave_ids):
        ax = axes[i]
        sub = raw[raw["wave_id"] == wid].sort_values("time_ms")

        ax.plot(sub["time_ms"], sub["current"], linewidth=1)
        ax.set_title(f"wave_id={wid}")

        # เส้น predicted wait
        if wid in pred_map:
            t_pred = float(pred_map[wid])
            ax.axvline(t_pred, linestyle="--", linewidth=1.5)
            ax.text(t_pred, sub["current"].max(), f" pred={t_pred:.2f}ms",
                    rotation=90, va="top", ha="left", fontsize=9)
        else:
            ax.text(0.02, 0.90, "no pred", transform=ax.transAxes, fontsize=9)

        ax.grid(True, alpha=0.3)

    # ลบ subplot ที่เหลือ (ถ้ามี)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.supxlabel("time (ms)")
    fig.supylabel("current")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"✅ saved: {args.out}")

if __name__ == "__main__":
    main()
