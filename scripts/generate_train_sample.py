# scripts/generate_raw_long.py
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def make_wave(t, final_value, settle_time_ms, sd, low, high, seed=0):
    """
    สร้างรูปคลื่นแบบ: step -> overshoot+ringing (damped sine) -> เข้า band แล้วนิ่ง
    """
    rng = np.random.default_rng(seed)

    # ตั้งค่าให้คลื่นสั่นแรงช่วงแรก แล้ว decay
    # amplitude เริ่มต้นให้สัมพันธ์กับ band width
    band_half = (high - low) / 2.0
    A0 = band_half * 8.0          # แรงสั่นช่วงแรก (ปรับได้)
    freq_hz = 450                 # ความถี่สั่น (ปรับได้)
    w = 2 * np.pi * freq_hz

    # decay ให้ประมาณ settle_time_ms แล้วเข้าช่วงนิ่ง
    # exp(-t/tau) -> ที่ t=settle_time ให้เหลือ ~2% ของ A0
    settle_s = settle_time_ms / 1000.0
    tau = max(settle_s / 4.0, 1e-6)

    # step response + ringing around final_value
    # ใช้ (1 - exp(-t/ts)) ทำให้ขึ้นเร็ว แล้วมี ringing ทับ
    ts = max(settle_s / 6.0, 1e-6)
    base = final_value * (1 - np.exp(-t / ts))

    ringing = A0 * np.exp(-t / tau) * np.sin(w * t)

    noise = rng.normal(0.0, sd, size=len(t))

    y = base + ringing + noise

    # หลังจาก settle_time_ms: บังคับให้อยู่ใน band + noise เล็กน้อย
    settled_mask = (t >= settle_s)
    y[settled_mask] = final_value + rng.normal(0.0, sd * 0.25, size=settled_mask.sum())

    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/raw_long.csv")
    ap.add_argument("--n_waves", type=int, default=20)
    ap.add_argument("--dt_ms", type=float, default=0.1)      # 0.1ms ตามตัวอย่างคุณ
    ap.add_argument("--t_end_ms", type=float, default=9.9)   # 0..9.9 => 100 samples
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # time axis
    t_ms = np.arange(0, args.t_end_ms + 1e-12, args.dt_ms)
    t_s = t_ms / 1000.0

    rows = []
    rng = np.random.default_rng(123)

    for wave_id in range(1, args.n_waves + 1):
        # สุ่ม "final value" และ band ให้ดูเหมือน tester
        # ถ้าคุณอยาก fix แบบเดิม: wave_id1 ~3.05, wave_id2 ~1.05, wave_id3 ~1.80 ...
        if wave_id == 1:
            final_value = 3.05
            low, high = 2.95, 3.15
        elif wave_id == 2:
            final_value = 1.05
            low, high = 0.95, 1.15
        elif wave_id == 3:
            final_value = 1.80
            low, high = 1.75, 1.85
        else:
            final_value = rng.uniform(0.8, 3.2)
            band = rng.uniform(0.10, 0.25)
            low, high = final_value - band/2, final_value + band/2

        sd = 0.01

        # กำหนด settle_time ให้ต่างกันตาม wave_id (ตัวอย่าง: 3ms..7ms)
        # คุณจะ map ตาม scenario จริงได้ทีหลัง
        settle_time_ms = float(rng.choice([3.0, 3.5, 4.0, 4.7, 5.0, 5.5, 6.0, 6.5]))
        if wave_id == 1:
            settle_time_ms = 3.0
        if wave_id == 2:
            settle_time_ms = 5.0

        y = make_wave(
            t=t_s,
            final_value=final_value,
            settle_time_ms=settle_time_ms,
            sd=sd,
            low=low,
            high=high,
            seed=1000 + wave_id
        )

        for i, (tm, val) in enumerate(zip(t_ms, y)):
            rows.append({
                "wave_id": wave_id,
                "sample": i,
                "time_ms": float(tm),
                "value": float(val),
                "sd": float(sd),
                "low_limit": float(low),
                "high_limit": float(high),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"✅ Wrote: {out_path}  (rows={len(df)})")

if __name__ == "__main__":
    main()
