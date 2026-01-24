import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["check_train", "check_pred"], 
                    help="1. check_train: เทียบ Raw Train กับ Label จริง | 2. check_pred: เทียบ Raw Test กับค่าที่ AI ทาย")
    ap.add_argument("--raw", required=True, help="Path to raw CSV (data_for_train.csv หรือ data1000samples_test_2.csv)")
    ap.add_argument( "--result", required=False, default=None,help="(optional) Path to result CSV (train_with_predictions.csv หรือ predictions.csv). ถ้าไม่ใส่ จะพล็อต raw อย่างเดียว"  )
    ap.add_argument("--out", default="plots/check_results/result.png")
    ap.add_argument("--show_filters", action="store_true", help="Plot medium/heavy filtered waveform (kernel=11,51)")
    ap.add_argument("--ncols", type=int, default=5)
    ap.add_argument("--per_page", type=int, default=20)
    args = ap.parse_args()

    # 1. โหลดข้อมูล
    print(f"📂 Loading Raw Data: {args.raw}")
    raw = pd.read_csv(args.raw)

    val_col = "value" if "value" in raw.columns else "current"
    time_col = "time_ms" if "time_ms" in raw.columns else "time"

    # 2. ตั้งค่าตามโหมด (ใช้เมื่อมี result เท่านั้น)
    if args.mode == "check_train":
        target_col = "wait_time_ms"
        line_color = "royalblue"
        line_style = "-"
        title_prefix = "CHECK TRAIN LABEL"
    else:
        target_col = "pred_wait_time_ms"
        line_color = "darkorange"
        line_style = "--"
        title_prefix = "CHECK PREDICTION"

    # 3) โหลด result แบบ optional
    val_map = {}
    if args.result:
        print(f"📂 Loading Result Data: {args.result}")
        res_df = pd.read_csv(args.result)
        if "wave_id" not in res_df.columns:
            raise ValueError("Result CSV must have 'wave_id' column")

        if target_col not in res_df.columns:
            raise ValueError(f"Result CSV missing required column '{target_col}' for mode={args.mode}")

        tmp = res_df[["wave_id", target_col]].dropna()
        val_map = tmp.groupby("wave_id")[target_col].median().to_dict()

    else:
        print("ℹ️ No --result provided → plotting RAW only (no vertical lines).")

    all_wave_ids = sorted(raw["wave_id"].unique())
    num_pages = math.ceil(len(all_wave_ids) / args.per_page)

    # 4) วนลูปสร้างกราฟ
    base_out = Path(args.out)
    base_out.parent.mkdir(parents=True, exist_ok=True)

    for page in range(num_pages):
        start_idx = page * args.per_page
        end_idx = min(start_idx + args.per_page, len(all_wave_ids))
        batch_ids = all_wave_ids[start_idx:end_idx]

        nrows = math.ceil(len(batch_ids) / args.ncols)
        fig, axes = plt.subplots(nrows, args.ncols,figsize=(4.5 * args.ncols, 3.2 * nrows),dpi=150)
        axes = axes.flatten() if len(batch_ids) > 1 else [axes]

        print(f"🎨 Drawing {args.mode} - Page {page+1}/{num_pages}...")

        for i, wid in enumerate(batch_ids):
            ax = axes[i]
            sub = raw[raw["wave_id"] == wid].sort_values(time_col)

            t = sub[time_col].to_numpy(dtype=float)
            x = sub[val_col].to_numpy(dtype=float)
            N = len(x)

            # raw
            ax.plot(
                t, x,
                color='black',       # ชัดกว่า gray
                alpha=0.85,          # ลดความจาง
                linewidth=1.3        # หนาขึ้น
            )

            # filtered
            if args.show_filters:
                x_med = medfilt(x, kernel_size=11)
                x_heavy = medfilt(x, kernel_size=(51 if N > 51 else 11))
                ax.plot(t, x_med, linewidth=1.0, alpha=0.9)
                ax.plot(t, x_heavy, linewidth=1.2, alpha=0.95)

            # ✅ วาดเส้นเฉพาะเมื่อมี result จริง ๆ
            if val_map and wid in val_map and pd.notna(val_map[wid]):
                t_val = float(val_map[wid])
                ax.axvline(t_val, color=line_color, linestyle=line_style, linewidth=2)

                post_data = sub[sub[time_col] >= t_val]
                if not post_data.empty:
                    y_max, y_min = post_data[val_col].max(), post_data[val_col].min()
                    ax.set_title(f"{title_prefix}\nID:{wid} | Δ:{y_max - y_min:.3f}", fontsize=8)
                else:
                    ax.set_title(f"{title_prefix}\nID:{wid}", fontsize=8)

                ax.text(
                    t_val, ax.get_ylim()[1], f"{t_val:.2f}ms",
                    color=line_color, fontsize=7, ha='center', va='bottom', fontweight='bold'
                )
            else:
                ax.set_title(f"RAW ONLY\nID:{wid}", fontsize=8)

            if args.show_filters and i == 0:
                ax.legend(["raw", "med k=11", f"heavy k={'51' if N > 51 else '11'}"], fontsize=6, loc="upper right")

            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=7)
            

        # ลบแกนว่าง
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        save_name = f"{args.mode}_pg{page+1}.png"
        fig.savefig(base_out.parent / save_name, dpi=200)
        plt.close(fig)

    print(f"✨ Done! Graphs saved in {base_out.parent}")


if __name__ == "__main__":
    main()