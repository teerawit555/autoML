import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["check_train", "check_pred"], 
                    help="1. check_train: เทียบ Raw Train กับ Label จริง | 2. check_pred: เทียบ Raw Test กับค่าที่ AI ทาย")
    ap.add_argument("--raw", required=True, help="Path to raw CSV (data_for_train.csv หรือ data1000samples_test_2.csv)")
    ap.add_argument("--result", required=True, help="Path to result CSV (train_with_predictions.csv หรือ predictions.csv)")
    ap.add_argument("--out", default="plots/check_results/result.png")
    ap.add_argument("--ncols", type=int, default=5)
    ap.add_argument("--per_page", type=int, default=20)
    args = ap.parse_args()

    # 1. โหลดข้อมูล
    print(f"📂 Loading Raw Data: {args.raw}")
    print(f"📂 Loading Result Data: {args.result}")
    raw = pd.read_csv(args.raw)
    res_df = pd.read_csv(args.result)

    val_col = "value" if "value" in raw.columns else "current"
    time_col = "time_ms" if "time_ms" in raw.columns else "time"

    # 2. จัดการ Mapping ตามโหมด
    if args.mode == "check_train":
        # โหมด 1: ดูว่าสูตรคำนวณ Settling Time ของเรา (Actual) มันตรงกับหน้าตาคลื่นไหม
        # ใช้ wait_time_ms จากไฟล์ผลลัพธ์การเทรน
        target_col = "wait_time_ms"
        line_color = "royalblue"
        line_style = "-"
        title_prefix = "CHECK TRAIN LABEL"
    else:
        # โหมด 2: ดูว่าที่ AI ทาย (Pred) มันตรงกับหน้าตาคลื่นไหม
        target_col = "pred_wait_time_ms"
        line_color = "darkorange"
        line_style = "--"
        title_prefix = "CHECK PREDICTION"

    val_map = dict(zip(res_df["wave_id"], res_df[target_col]))

    all_wave_ids = sorted(raw["wave_id"].unique())
    num_pages = math.ceil(len(all_wave_ids) / args.per_page)

    # 3. วนลูปสร้างกราฟ
    base_out = Path(args.out)
    base_out.parent.mkdir(parents=True, exist_ok=True)

    for page in range(num_pages):
        start_idx = page * args.per_page
        end_idx = min(start_idx + args.per_page, len(all_wave_ids))
        batch_ids = all_wave_ids[start_idx:end_idx]
        
        nrows = math.ceil(len(batch_ids) / args.ncols)
        fig, axes = plt.subplots(nrows, args.ncols, figsize=(4 * args.ncols, 3 * nrows))
        axes = axes.flatten() if len(batch_ids) > 1 else [axes]

        print(f"🎨 Drawing {args.mode} - Page {page+1}/{num_pages}...")

        for i, wid in enumerate(batch_ids):
            ax = axes[i]
            sub = raw[raw["wave_id"] == wid].sort_values(time_col)
            
            # พล็อตคลื่น
            ax.plot(sub[time_col], sub[val_col], color='gray', alpha=0.5, linewidth=0.8)
            
            # พล็อตเส้นแนวตั้ง (ค่าที่เลือกตามโหมด)
            if wid in val_map:
                t_val = float(val_map[wid])
                ax.axvline(t_val, color=line_color, linestyle=line_style, linewidth=2)
                
                # แสดงข้อมูลสถิติหลังเส้นนิ่ง
                post_data = sub[sub[time_col] >= t_val]
                if not post_data.empty:
                    y_max, y_min = post_data[val_col].max(), post_data[val_col].min()
                    ax.set_title(f"ID:{wid} | Δ:{y_max-y_min:.3f}", fontsize=8)
                
                ax.text(t_val, ax.get_ylim()[1], f"{t_val:.2f}ms", 
                        color=line_color, fontsize=7, ha='center', va='bottom', fontweight='bold')
            
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=7)

        # ลบแกนว่าง
        for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
        
        plt.tight_layout()
        save_name = f"{base_out.stem}_{args.mode}_pg{page+1}.png"
        fig.savefig(base_out.parent / save_name, dpi=120)
        plt.close(fig)

    print(f"✨ Done! Graphs saved in {base_out.parent}")

if __name__ == "__main__":
    main()