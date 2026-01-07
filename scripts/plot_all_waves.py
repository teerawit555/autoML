# import argparse
# import math
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# def main():
#     ap = argparse.ArgumentParser(description="Plot waves with comparison between Actual and Predicted values.")
#     ap.add_argument("--raw", required=True, help="Path to raw CSV (Long format)")
#     ap.add_argument("--pred", required=True, help="Path to prediction CSV")
#     # เพิ่ม Argument สำหรับไฟล์ที่มีค่าจริง (Actual)
#     ap.add_argument("--actual", help="Path to train_with_predictions CSV (to get Actual lines)")
#     ap.add_argument("--out", default="plots/waves/result.png", help="Output filename")
#     ap.add_argument("--ncols", type=int, default=5, help="Columns per row")
#     ap.add_argument("--per_page", type=int, default=50, help="Number of waves per page")
#     args = ap.parse_args()

#     # 1. โหลดข้อมูล
#     print(f"📂 Reading Raw: {args.raw}")
#     print(f"📂 Reading Pred: {args.pred}")
#     raw = pd.read_csv(args.raw)
#     pred_df = pd.read_csv(args.pred)

#     val_col = "value" if "value" in raw.columns else "current"
    
#     # --- [ส่วนที่แก้ไข] ตรรกะการ Map ข้อมูล ---
#     # 1. Map ค่าทำนาย (Orange Dashed Line)
#     pred_map = dict(zip(pred_df["wave_id"], pred_df["pred_wait_time_ms"]))
    
#     # 2. Map ค่าจริง (Blue Solid Line)
#     actual_map = {}
    
#     # กรณีที่ 1: ระบุไฟล์ --actual แยกมาให้ (เช่น ไฟล์จากการ Train)
#     if args.actual:
#         print(f"📂 Reading Actual labels from: {args.actual}")
#         act_df = pd.read_csv(args.actual)
#         if "wait_time_ms" in act_df.columns:
#             actual_map = dict(zip(act_df["wave_id"], act_df["wait_time_ms"]))
            
#     # กรณีที่ 2: ค่าจริงอยู่ในไฟล์ --pred อยู่แล้ว (Fallback)
#     if not actual_map and "wait_time_ms" in pred_df.columns:
#         actual_map = dict(zip(pred_df["wave_id"], pred_df["wait_time_ms"]))
    
#     all_wave_ids = sorted(raw["wave_id"].unique())
#     total_waves = len(all_wave_ids)
#     num_pages = math.ceil(total_waves / args.per_page)

#     print(f"📊 Total Waves: {total_waves} | Actual Labels Found: {len(actual_map)}")

#     # --- สร้าง Folder อัตโนมัติ ---
#     base_out_path = Path(args.out)
#     out_dir = base_out_path.parent
#     out_dir.mkdir(parents=True, exist_ok=True)
#     stem_name = base_out_path.stem 

#     # 2. วนลูปสร้างกราฟ
#     for page in range(num_pages):
#         start_idx = page * args.per_page
#         end_idx = min(start_idx + args.per_page, total_waves)
#         batch_ids = all_wave_ids[start_idx:end_idx]
        
#         nrows = math.ceil(len(batch_ids) / args.ncols)
#         fig, axes = plt.subplots(nrows, args.ncols, figsize=(4.5 * args.ncols, 3.5 * nrows), sharex=True, sharey=True)
        
#         if len(batch_ids) > 1: axes = axes.flatten()
#         else: axes = [axes]

#         print(f"   🎨 Drawing Page {page + 1}/{num_pages}...")

#         for i, wid in enumerate(batch_ids):
#             ax = axes[i]
#             time_col = "time" if "time" in raw.columns else "time_ms"
#             sub = raw[raw["wave_id"] == wid].sort_values(time_col)
            
#             # Plot เส้นสัญญาณหลัก
#             ax.plot(sub[time_col], sub[val_col], linewidth=1, color='steelblue', alpha=0.7)
#             ax.set_title(f"ID={wid}", fontsize=10, fontweight='bold')
            
#             stats_list = []

#             # --- [ส่วนที่แสดงผล] Plot เส้นค่าจริง (Actual - สีน้ำเงินทึบ) ---
#             if wid in actual_map:
#                 t_actual = float(actual_map[wid])
#                 ax.axvline(t_actual, color='royalblue', linestyle="-", linewidth=2, alpha=0.8)
#                 stats_list.append(f"Actual: {t_actual:.2f}ms")

#             # --- Plot เส้นทำนาย (Pred - สีส้มประ) ---
#             if wid in pred_map:
#                 t_pred = float(pred_map[wid])
#                 ax.axvline(t_pred, color='darkorange', linestyle="--", linewidth=1.8)
#                 stats_list.append(f"Pred: {t_pred:.2f}ms")
                
#                 # ส่วนคำนวณ Max/Min หลัง Prediction
#                 post_data = sub[sub[time_col] >= t_pred]
#                 if not post_data.empty:
#                     y_max, y_min = post_data[val_col].max(), post_data[val_col].min()
#                     t_end = sub[time_col].max()
#                     ax.hlines(y=[y_max, y_min], xmin=t_pred, xmax=t_end, colors=['green','red'], linestyles=':', linewidth=1, alpha=0.6)
#                     stats_list.append(f"Max: {y_max:.2f}")
#                     stats_list.append(f"Min: {y_min:.2f}")

#                 # แสดงกล่องข้อความสรุป
#                 box_text = "\n".join(stats_list)
#                 ax.text(0.95, 0.95, box_text, transform=ax.transAxes, fontsize=7, ha='right', va='top',
#                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="orange", alpha=0.8))
#             else:
#                 ax.text(0.5, 0.5, "No Pred", transform=ax.transAxes, ha='center', color='gray')

#             ax.grid(True, alpha=0.3)
#             if i >= (nrows - 1) * args.ncols: ax.set_xlabel("time (ms)", fontsize=8)
#             if i % args.ncols == 0: ax.set_ylabel("value", fontsize=8)

#         # ลบแกนที่ไม่ได้ใช้
#         for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
#         fig.tight_layout()
        
#         save_path = out_dir / f"{stem_name}_page{page + 1}.png"
#         fig.savefig(save_path, dpi=150)
#         plt.close(fig)
#         print(f"      ✅ Saved: {save_path}")

#     print("\n✨ Comparison plots saved successfully!")

# if __name__ == "__main__":
#     main()

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