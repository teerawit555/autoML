import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="../data/raw/raw_long.csv", help="input ../data/raw/raw_long.csv")
    ap.add_argument("--wave", type=int, default=None, help="plot only this wave_id (optional)")
    ap.add_argument("--max-waves", type=int, default=6, help="max number of waves to plot (when --wave not set)")
    ap.add_argument("--save", default=None, help="save figure path (optional)")
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)

    # expected columns: wave_id, sample_idx, time_ms, current
    required = {"wave_id", "sample_idx", "current"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"❌ Missing columns: {missing}\nFound columns: {list(df.columns)}")

    # sort for clean lines
    sort_cols = ["wave_id", "sample_idx"]
    df = df.sort_values(sort_cols)

    # choose x-axis
    x_col = "time_ms" if "time_ms" in df.columns else "sample_idx"

    if args.wave is not None:
        wave_ids = [args.wave]
    else:
        wave_ids = df["wave_id"].dropna().unique().tolist()[:args.max_waves]

    plt.figure()
    for wid in wave_ids:
        d = df[df["wave_id"] == wid]
        plt.plot(d[x_col].to_numpy(), d["current"].to_numpy(), label=f"wave_id={wid}")

    plt.xlabel(x_col)
    plt.ylabel("current")
    plt.title("Raw tester waveform (current vs time)")
    plt.legend()
    plt.grid(True)

    if args.save:
        plt.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"✅ Saved: {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
