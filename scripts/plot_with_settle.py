import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../data/raw/raw_long.csv", help="raw_long.csv")
    ap.add_argument("--feat", default="../data/processed/train_features.csv", help="train_features.csv (must contain wave_id, settle_idx)")
    ap.add_argument("--wave", type=int, default=None, help="plot only this wave_id (optional)")
    ap.add_argument("--max-waves", type=int, default=6, help="max waves to plot")
    ap.add_argument("--save", default=None, help="save figure path (optional)")
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    feat = pd.read_csv(args.feat)

    # checks
    for c in ["wave_id", "sample_idx", "current"]:
        if c not in raw.columns:
            raise SystemExit(f"❌ raw_long.csv missing column: {c}")
    for c in ["wave_id", "settle_idx"]:
        if c not in feat.columns:
            raise SystemExit(f"❌ train_features.csv missing column: {c}")

    raw = raw.sort_values(["wave_id", "sample_idx"])

    x_col = "time_ms" if "time_ms" in raw.columns else "sample_idx"

    settle_map = dict(zip(feat["wave_id"], feat["settle_idx"]))

    if args.wave is not None:
        wave_ids = [args.wave]
    else:
        wave_ids = raw["wave_id"].dropna().unique().tolist()[:args.max_waves]

    plt.figure()
    for wid in wave_ids:
        d = raw[raw["wave_id"] == wid]
        plt.plot(d[x_col].to_numpy(), d["current"].to_numpy(), label=f"wave_id={wid}")

        if wid in settle_map and pd.notna(settle_map[wid]):
            sidx = int(settle_map[wid])
            # find x position of settle index
            dd = d[d["sample_idx"] == sidx]
            if len(dd) > 0:
                x_settle = float(dd.iloc[0][x_col])
                plt.axvline(x_settle, linestyle="--")
            else:
                # fallback: if sample_idx not found, still show index on x-axis if using sample_idx
                if x_col == "sample_idx":
                    plt.axvline(sidx, linestyle="--")

    plt.xlabel(x_col)
    plt.ylabel("current")
    plt.title("Raw waveform with settle marker (vertical dashed line)")
    plt.legend()
    plt.grid(True)

    if args.save:
        plt.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"✅ Saved: {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
