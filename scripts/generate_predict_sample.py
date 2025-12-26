import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def make_wave(t, final_value, low, high, seed=0, add_late_vibe=False):
    rng = np.random.default_rng(seed)

    # 1. Random Physics Parameters
    freq_hz = rng.uniform(100, 1200)
    w = 2 * np.pi * freq_hz
    band_half = (high - low) / 2.0
    overshoot_scale = rng.uniform(1.5, 20.0) 
    A0 = band_half * overshoot_scale * rng.choice([1, -1])

    sd = rng.uniform(0.001, 0.04)
    # สมมติเวลา settle หลอกๆ เพื่อใช้สร้างทรงกราฟ (แต่จะไม่บอก Model)
    internal_settle_s = rng.uniform(2.0, 8.0) / 1000.0
    tau = max(internal_settle_s / rng.uniform(2.5, 6.0), 1e-6)
    ts = max(internal_settle_s / rng.uniform(3.0, 10.0), 1e-6)

    # 2. Base Calculation
    base = final_value * (1 - np.exp(-t / ts))
    ringing = A0 * np.exp(-t / tau) * np.sin(w * t)
    noise = rng.normal(0.0, sd, size=len(t))
    y = base + ringing + noise

    # Force settle ช่วงท้ายจริงๆ (หลอกว่าระบบนิ่งแล้ว)
    settle_idx = np.searchsorted(t, internal_settle_s)
    y[settle_idx:] = final_value + rng.normal(0.0, sd * 0.5, size=len(t)-settle_idx)

    # --- เพิ่มการสั่นช่วงปลาย (Late Sine Wave) ---
    if add_late_vibe:
        late_freq = rng.uniform(30, 100)
        late_amp = (high - low) * rng.uniform(0.4, 0.9)
        # เริ่มสั่นตั้งแต่ 60% ของความยาวคลื่น
        vibe_start = int(len(t) * 0.6)
        y[vibe_start:] += late_amp * np.sin(2 * np.pi * late_freq * t[vibe_start:])

    return y, sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/data_for_predict_wild.csv")
    ap.add_argument("--n_waves", type=int, default=100)
    ap.add_argument("--dt_ms", type=float, default=0.01)
    ap.add_argument("--t_end_ms", type=float, default=9.9)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_ms = np.arange(0, args.t_end_ms + 1e-12, args.dt_ms)
    t_s = t_ms / 1000.0
    rng = np.random.default_rng(888) # ใช้ Seed ใหม่สำหรับไฟล์ Predict

    # สุ่มเลือกกลุ่มเป้าหมายที่จะโดน "ป่วน"
    all_ids = np.arange(1, args.n_waves + 1)
    late_vibe_ids = rng.choice(all_ids, size=10, replace=False)
    # กรองเอา id ที่ไม่ซ้ำกับกลุ่มแรกมาทำ incomplete
    remaining_ids = np.setdiff1d(all_ids, late_vibe_ids)
    incomplete_ids = rng.choice(remaining_ids, size=10, replace=False)

    rows = []
    print(f"🧪 Generating Wild Inference Dataset ({args.n_waves} waves)...")

    for wave_id in all_ids:
        final_value = rng.uniform(0.5, 3.5)
        band = rng.uniform(0.05, 0.20)
        low, high = final_value - band/2, final_value + band/2

        # สร้าง Wave (ส่ง add_late_vibe=True ถ้าอยู่ในกลุ่มที่สุ่มได้)
        y, used_sd = make_wave(
            t=t_s, final_value=final_value, 
            low=low, high=high, 
            seed=7000 + wave_id, 
            add_late_vibe=(wave_id in late_vibe_ids)
        )

        # ตัดข้อมูลให้ไม่เท่ากัน (Incomplete)
        current_y = y
        current_t = t_ms
        if wave_id in incomplete_ids:
            cut_point = int(len(y) * rng.uniform(0.5, 0.9)) # ตัดหายไป 10-50%
            current_y = y[:cut_point]
            current_t = t_ms[:cut_point]

        # เก็บข้อมูลเข้า List (ไม่มี wait_time_ms แล้ว)
        for i, (tm, val) in enumerate(zip(current_t, current_y)):
            rows.append({
                "wave_id": wave_id,
                "sample": i,
                "time": float(tm),
                "value": float(val),
                "low_limit": float(low),
                "high_limit": float(high)
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    
    print(f"✅ Created: {out_path}")
    print(f"📊 Summary:")
    print(f"   - Late Vibration IDs (10): {sorted(late_vibe_ids)}")
    print(f"   - Incomplete Sample IDs (10): {sorted(incomplete_ids)}")
    print(f"   - Other waves (80): Normal randomized physics")
    print(f"\n🚀 Ready for prediction! This file has NO ground truth.")

if __name__ == "__main__":
    main()