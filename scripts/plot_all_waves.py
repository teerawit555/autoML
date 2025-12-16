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
    # แก้ไข: เปลี่ยน 'current' เป็น 'value'
    for c in ["wave_id", "time", "value"]:
        if c not in raw.columns:
            # ใช้ 'time' แทน 'time_ms' (ตามโครงสร้างข้อมูลดิบที่คุณให้มา)
            raise ValueError(f"raw file missing column: {c}")

    # ต้องมี wave_id + pred_wait_time_ms ใน pred
    if "pred_wait_time_ms" not in pred.columns:
        raise ValueError("pred file missing column: pred_wait_time_ms")

    if "wave_id" not in pred.columns:
        raise ValueError("pred file missing column: wave_id (ดูหมายเหตุด้านล่าง)")

    # map predicted wait time ต่อ wave
    # ใช้ค่าเฉลี่ยของ pred_wait_time_ms ต่อ wave_id หากมีหลายแถวต่อ wave
    # (กรณี pred มาจาก train_with_predictions ซึ่งมีหลายแถวต่อ wave_id)
    # แต่เนื่องจากไฟล์ทำนายมาจาก features.csv ซึ่งมี 1 แถวต่อ wave_id จึงใช้ .iloc[0]
    
    # คำเตือน: ไฟล์ทำนาย (predicted_wait_time_1000.csv) มี 1 แถวต่อ wave_id อยู่แล้ว
    # แต่ถ้าใช้ไฟล์ train_with_predictions.csv (จากโหมด train) จะมีหลายแถว
    # เพื่อความปลอดภัย เราจะใช้ .iloc[0] เพราะไฟล์ features มี 1 แถวต่อ wave_id
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
    # จัดการกรณีมีแค่ 1 plot หรือมีหลาย plot
    axes = axes.flatten() if n > 1 else [axes] if n == 1 else []

    for i, wid in enumerate(wave_ids):
        ax = axes[i]
        sub = raw[raw["wave_id"] == wid].sort_values("time") # ใช้ "time"
        
        # แก้ไข: ใช้ "value" แทน "current"
        ax.plot(sub["time"], sub["value"], linewidth=1)
        ax.set_title(f"wave_id={wid}")

        # เส้น predicted wait
        if wid in pred_map:
            t_pred = float(pred_map[wid])
            ax.axvline(t_pred, linestyle="--", linewidth=1.5, color='r')
            
            # แก้ไข: ใช้ sub["value"].max()
            ax.text(t_pred, sub["value"].max() * 0.95, f" pred={t_pred:.2f}ms",
                     rotation=90, va="top", ha="left", fontsize=9, color='r')
        else:
            ax.text(0.02, 0.90, "no pred", transform=ax.transAxes, fontsize=9)

        ax.grid(True, alpha=0.3)

    # ลบ subplot ที่เหลือ (ถ้ามี)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.supxlabel("time (ms)")
    fig.supylabel("value") # แก้ไข label
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"saved: {args.out}")

if __name__ == "__main__":
    main()