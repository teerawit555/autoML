import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Plot waves with pagination and post-pred stats.")
    ap.add_argument("--raw", required=True, help="Path to raw CSV")
    ap.add_argument("--pred", required=True, help="Path to prediction CSV")
    # ปรับ default output ให้ไปลงที่ plots/waves ตามที่ต้องการ
    ap.add_argument("--out", default="plots/waves/result.png", help="Output filename (folders will be created automatically)")
    ap.add_argument("--ncols", type=int, default=5, help="Columns per row")
    ap.add_argument("--per_page", type=int, default=50, help="Number of waves per page")
    args = ap.parse_args()

    # 1. โหลดข้อมูล
    print(f"📂 Reading Raw: {args.raw}")
    print(f"📂 Reading Pred: {args.pred}")
    raw = pd.read_csv(args.raw)
    pred = pd.read_csv(args.pred)

    val_col = "value" if "value" in raw.columns else "current"
    
    # Map คำตอบ
    pred_map = dict(zip(pred["wave_id"], pred["pred_wait_time_ms"]))
    all_wave_ids = sorted(raw["wave_id"].unique())
    
    total_waves = len(all_wave_ids)
    per_page = args.per_page
    num_pages = math.ceil(total_waves / per_page)

    print(f"📊 Total Waves: {total_waves}")
    print(f"📄 Splitting into {num_pages} pages (Max {per_page} per page)")

    # --- [ส่วนที่ 1] สร้าง Folder อัตโนมัติ ---
    base_out_path = Path(args.out)
    out_dir = base_out_path.parent  # ดึง path โฟลเดอร์จากชื่อไฟล์
    
    # สร้าง folder (เช่น plots/waves/...) ถ้ายังไม่มี
    if not out_dir.exists():
        print(f"📁 Creating directory: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        
    stem_name = base_out_path.stem 

    # 2. วนลูปสร้างกราฟ
    for page in range(num_pages):
        start_idx = page * per_page
        end_idx = min(start_idx + per_page, total_waves)
        batch_ids = all_wave_ids[start_idx:end_idx]
        
        n_batch = len(batch_ids)
        ncols = args.ncols
        nrows = math.ceil(n_batch / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4.0 * ncols, 3.0 * nrows),
            sharex=True, sharey=True
        )
        
        if n_batch > 1: axes = axes.flatten()
        else: axes = [axes]

        print(f"   🎨 Drawing Page {page + 1}/{num_pages} (Waves {start_idx+1}-{end_idx})...")

        for i, wid in enumerate(batch_ids):
            ax = axes[i]
            sub = raw[raw["wave_id"] == wid].sort_values("time")
            
            # Plot เส้นสัญญาณหลัก
            ax.plot(sub["time"], sub[val_col], linewidth=1, label='Signal', color='steelblue', alpha=0.8)
            ax.set_title(f"ID={wid}", fontsize=10, fontweight='bold')
            
            # Plot เส้นทำนาย (Prediction Line)
            if wid in pred_map:
                t_pred = float(pred_map[wid])
                
                # วาดเส้นแนวตั้ง (เวลาที่ทำนาย)
                ax.axvline(t_pred, color='orange', linestyle="--", linewidth=1.5)
                
                # --- [ส่วนที่ 2] คำนวณหา Max/Min หลังจากเวลา Prediction ---
                # ตัดข้อมูลเอาเฉพาะช่วงเวลาหลังจาก t_pred เป็นต้นไป
                post_data = sub[sub["time"] >= t_pred]
                
                if not post_data.empty:
                    y_max = post_data[val_col].max()
                    y_min = post_data[val_col].min()
                    t_end = sub["time"].max()

                    # วาดเส้นประแนวนอนบอกขอบเขต Max (สีเขียว) / Min (สีแดง)
                    ax.hlines(y=y_max, xmin=t_pred, xmax=t_end, colors='green', linestyles=':', linewidth=1, alpha=0.7)
                    ax.hlines(y=y_min, xmin=t_pred, xmax=t_end, colors='red', linestyles=':', linewidth=1, alpha=0.7)

                    # ใส่ตัวเลข Max/Min โชว์ในกราฟ
                    stats_text = f"Max: {y_max:.2f}\nPred: {t_pred:.2f}ms\nMin: {y_min:.2f}"
                    
                    # วางกล่องข้อความ (วางมุมขวาบน หรือตำแหน่งที่เหมาะสม)
                    ax.text(0.95, 0.95, stats_text, 
                            transform=ax.transAxes, # ใช้พิกัดเทียบกับกรอบรูป (0-1)
                            fontsize=7, 
                            color='black',
                            ha='right', va='top',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="orange", alpha=0.8))
                else:
                    # กรณีทำนายเกินเวลากราฟ
                    ax.text(t_pred, sub[val_col].max(), "Pred > End", fontsize=7, color='red')

            else:
                ax.text(0.5, 0.5, "No Pred", transform=ax.transAxes, ha='center', color='gray')

            ax.grid(True, alpha=0.3)
            
            # Label แกน
            if i >= (nrows - 1) * ncols: ax.set_xlabel("time (ms)", fontsize=8)
            if i % ncols == 0: ax.set_ylabel("value", fontsize=8)

        # ลบแกนเปล่าทิ้ง
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        
        save_name = f"{stem_name}_page{page + 1}.png"
        save_path = out_dir / save_name
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"      ✅ Saved: {save_path}")

    print("\n✨ All plots saved successfully!")

if __name__ == "__main__":
    main()