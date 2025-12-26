import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

def detect_columns(df, file_label):
    """
    Detect Time, Value, and ID columns automatically to avoid KeyError.
    """
    cols = df.columns.tolist()
    # Find Time Column (check for various names)
    t_col = next((c for c in cols if c.lower() in ['time_ms', 'time', 't', 'ms', 'timestamp']), None)
    # Find Signal/Value Column
    v_col = next((c for c in cols if c.lower() in ['value', 'current', 'v', 'signal', 'volt', 'amp']), None)
    # Find Wave ID Column
    id_col = next((c for c in cols if c.lower() in ['wave_id', 'id', 'wave']), None)

    if not t_col or not v_col or not id_col:
        print(f"❌ Error in {file_label}!")
        print(f"Columns found in file: {cols}")
        raise KeyError(f"Could not find required columns in {file_label}. Check header names.")
    
    return id_col, t_col, v_col

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_train", required=True, help="Path to data_for_train.csv")
    ap.add_argument("--raw_test", required=True, help="Path to data1000samples_test.csv")
    ap.add_argument("--limit", type=int, default=100, help="Number of waves to plot")
    ap.add_argument("--ncols", type=int, default=5, help="Columns per row")
    ap.add_argument("--out_prefix", default="plots/compare_100/wave_check")
    args = ap.parse_args()

    # 1. Load Data
    print("📂 Loading Raw Data...")
    df_train = pd.read_csv(args.raw_train)
    df_test = pd.read_csv(args.raw_test)

    # 2. Detect Columns (Crucial: to handle different names in Train vs Test)
    try:
        id_tr, t_tr, v_tr = detect_columns(df_train, "TRAIN FILE")
        id_ts, t_ts, v_ts = detect_columns(df_test, "TEST FILE")
    except KeyError as e:
        print(e)
        return

    # 3. Get first N unique IDs (Sorted)
    train_ids = sorted(df_train[id_tr].unique())[:args.limit]
    test_ids = sorted(df_test[id_ts].unique())[:args.limit]
    
    num_to_plot = min(len(train_ids), len(test_ids), args.limit)
    per_page = 20  # 20 pairs (40 charts total) per image for clarity
    num_pages = math.ceil(num_to_plot / per_page)

    print(f"🎨 Drawing first {num_to_plot} waves comparison...")
    print(f"   Train setup: ID='{id_tr}', Time='{t_tr}', Val='{v_tr}'")
    print(f"   Test  setup: ID='{id_ts}', Time='{t_ts}', Val='{v_ts}'")

    for p in range(num_pages):
        start = p * per_page
        end = min(start + per_page, num_to_plot)
        batch_tr = train_ids[start:end]
        batch_ts = test_ids[start:end]
        
        n_waves = len(batch_tr)
        n_rows = math.ceil(n_waves / args.ncols)

        # Build figure: Train (Top) and Test (Bottom) paired
        fig, axes = plt.subplots(n_rows * 2, args.ncols, 
                                 figsize=(4 * args.ncols, 3.5 * n_rows * 2), 
                                 sharey=False)
        axes = axes.flatten()

        for i in range(n_waves):
            col = i % args.ncols
            row_group = i // args.ncols
            idx_tr = (row_group * 2) * args.ncols + col
            idx_ts = (row_group * 2 + 1) * args.ncols + col

            # --- Plot RAW TRAIN ---
            ax_tr = axes[idx_tr]
            sub_tr = df_train[df_train[id_tr] == batch_tr[i]].sort_values(t_tr)
            ax_tr.plot(sub_tr[t_tr], sub_tr[v_tr], color='steelblue', linewidth=1)
            ax_tr.set_title(f"TRAIN ID: {batch_tr[i]}", fontsize=9, fontweight='bold')
            ax_tr.grid(True, alpha=0.3)

            # --- Plot RAW TEST ---
            ax_ts = axes[idx_ts]
            sub_ts = df_test[df_test[id_ts] == batch_ts[i]].sort_values(t_ts)
            ax_ts.plot(sub_ts[t_ts], sub_ts[v_ts], color='seagreen', linewidth=1)
            ax_ts.set_title(f"TEST ID: {batch_ts[i]}", fontsize=9, fontweight='bold')
            ax_ts.grid(True, alpha=0.3)

        # Delete unused axes
        for ax in axes[n_waves * 2:]:
            fig.delaxes(ax)

        plt.tight_layout()
        os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
        save_path = f"{args.out_prefix}_page_{p+1}.png"
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"   ✅ Saved Page {p+1}: {save_path}")

    print("\n✨ All comparison plots saved successfully!")

if __name__ == "__main__":
    main()