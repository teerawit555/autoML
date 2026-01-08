# scripts/generate_predict_sample.py
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd

# Import ตัวปกติ
from generate_train_sample import (
    generate_step_response,
    generate_high_start_oscillation,
    generate_continuous_triangular_pulses,
    generate_low_swing_sine_wave,
    generate_overdamped_decay,
    generate_pulse_train,
    generate_overdamped_decay1, # Type 4.1 (Overshoot)
    apply_cosine_taper_settling,
    add_post_settle_noise,
    add_time_delay
)

# =============================================================================
# HARD Generators (ปรับ Shape/Timing ให้โหดขึ้น)
# =============================================================================

def generate_step_response_HARD(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """ Type 0 HARD: Overshoot สูงปรี๊ด + สั่นนาน (Underdamped) """
    freq_hz = float(rng.uniform(80, 1500))
    w1 = 2.0 * np.pi * freq_hz
    band_half = (limit_high - limit_low) / 2.0

    overshoot_scale = float(rng.uniform(5.0, 20.0))  # Scale โหด
    direction = float(rng.choice([1.0, -1.0]))
    amp0 = band_half * overshoot_scale * direction

    tau = max(settling_time_s / float(rng.uniform(1.5, 3.5)), 1e-6) # หายช้า
    time_constant_rise = max(settling_time_s / float(rng.uniform(3.0, 10.0)), 1e-6)
    t_eff, _ = add_time_delay(time_vector, rng, max_delay_s=0.0008)

    base = target_value * (1.0 - np.exp(-t_eff / time_constant_rise))
    ring = amp0 * np.exp(-t_eff / tau) * np.sin(w1 * t_eff)
    y = base + ring

    y = apply_cosine_taper_settling(y, time_vector, settling_time_s, target_value, strength=0.85)
    y, sd = add_post_settle_noise(y, time_vector, settling_time_s, target_value, rng, probability=0.9)
    return y, sd, "type0_HARD", 0.0, 0

def generate_high_start_oscillation_HARD(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """ Type 1 HARD: เริ่มสูงมาก + แกว่งลึก """
    band = float(limit_high - limit_low)
    num_cycles = int(rng.integers(3, 10))
    t_eff, _ = add_time_delay(time_vector, rng, max_delay_s=0.0005)

    freq = num_cycles / max(settling_time_s, 1e-6)
    w = 2.0 * np.pi * freq
    A = band * float(rng.uniform(4.0, 9.0)) # Amplitude โหด

    bias_amp = band * float(rng.uniform(-1.5, 1.5))
    bias_tau = max(settling_time_s / float(rng.uniform(1.0, 2.5)), 1e-6)
    bias = bias_amp * np.exp(-t_eff / bias_tau)
    env = np.exp(-t_eff / (settling_time_s * 1.5)) # หายช้ามาก

    osc = A * env * np.cos(w * t_eff)
    y = target_value + bias + osc

    y = apply_cosine_taper_settling(y, time_vector, settling_time_s, target_value, strength=0.9)
    y, sd = add_post_settle_noise(y, time_vector, settling_time_s, target_value, rng)
    return y, sd, "type1_HARD", 0.0, 0

def generate_continuous_triangular_pulses_HARD(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """ Type 2 HARD: สามเหลี่ยมเบี้ยว (Sawtooth) + Jitter ระยะห่างเยอะ """
    y = np.full_like(time_vector, target_value, dtype=float)
    avg_height = (limit_high - limit_low) * float(rng.uniform(1.0, 3.0)) # สูงกว่าปกติ
    avg_period = float(rng.uniform(settling_time_s / 4.0, settling_time_s / 1.2))
    
    current_time = float(rng.uniform(0.0, avg_period * 0.3))
    
    while current_time < float(time_vector[-1]):
        # Hard: ความสูงเปลี่ยนไปมาเยอะ
        height = avg_height * float(rng.uniform(0.3, 1.8))
        
        # Hard: Pulse Width ไม่คงที่
        pulse_width = avg_period * float(rng.uniform(0.1, 0.4))
        
        # Hard: Skew (เบี้ยวซ้าย/ขวา)
        skew = float(rng.uniform(0.1, 0.9)) # 0.5 = สมมาตร, <0.5 เบี้ยวซ้าย, >0.5 เบี้ยวขวา
        t_peak_rel = pulse_width * skew

        t_start = current_time
        t_peak = t_start + t_peak_rel
        t_end = t_start + pulse_width

        # Rise
        rise = (time_vector >= t_start) & (time_vector < t_peak)
        if np.any(rise):
            y[rise] += (height / max(t_peak_rel, 1e-9)) * (time_vector[rise] - t_start)
        
        # Fall
        fall = (time_vector >= t_peak) & (time_vector < t_end)
        if np.any(fall):
            y[fall] += height - (height / max(pulse_width - t_peak_rel, 1e-9)) * (time_vector[fall] - t_peak)

        # Hard: Period Jitter เยอะๆ
        period = avg_period * float(rng.uniform(0.5, 1.8))
        current_time += period

    sd = max(target_value * 0.0005, 1e-6)
    y += rng.normal(0.0, sd, size=len(y))
    return y, sd, "type2_HARD", 0.0, 1

def generate_overdamped_decay_HARD(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """ Type 4 HARD: ลงช้ามากๆ (Super Slow Tail) """
    # Hard: เริ่มสูงมาก
    start_amp = (limit_high - limit_low) * float(rng.uniform(3.0, 6.0))
    
    # Hard: Tau ยาวกว่า Settling time (ทำให้ลงไม่สุด หรือลงช้ามาก)
    tau = settling_time_s * float(rng.uniform(0.8, 1.5)) 

    y = target_value + start_amp * np.exp(-time_vector / max(tau, 1e-12))
    
    # Taper น้อยๆ เพื่อปล่อยให้หางยาวๆ โผล่มา
    y = apply_cosine_taper_settling(y, time_vector, settling_time_s, target_value, strength=0.5)
    y, sd = add_post_settle_noise(y, time_vector, settling_time_s, target_value, rng)
    return y, sd, "type4_HARD", float(settling_time_s * 1000.0), 0

def generate_pulse_train_HARD(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """ Type 5 HARD: Pulse ผอม/อ้วนสลับกัน + Missing Pulse """
    y = np.full_like(time_vector, target_value, dtype=float)
    band = float(limit_high - limit_low)
    base_amp = band * float(rng.uniform(0.8, 2.5))

    t_end = float(time_vector[-1])
    period = float(rng.uniform(t_end / 6.0, t_end / 3.0))
    current_time = float(rng.uniform(0.0, period * 0.5))

    while current_time < t_end:
        # Hard: Pulse Width Swing เยอะๆ (บางอันผอมเป็นเข็ม บางอันอ้วน)
        duty = float(rng.uniform(0.02, 0.40)) 
        width = max(period * duty, 1e-6)

        # Hard: โอกาสที่ Pulse จะหายไป (Missing Pulse)
        if rng.random() > 0.15: 
            t_start = current_time
            t_stop = min(current_time + width, t_end)
            mask = (time_vector >= t_start) & (time_vector < t_stop)
            
            # Hard: Amplitude เปลี่ยนทุก Pulse แบบกระชาก
            this_amp = base_amp * float(rng.uniform(0.5, 2.0))
            y[mask] += this_amp

        # Hard: ระยะห่างแต่ละ Pulse ไม่เท่ากัน
        current_time += period * float(rng.uniform(0.6, 1.6))

    y, sd = add_post_settle_noise(y, time_vector, 0.0, target_value, rng, probability=0.0)
    return y, sd, "type5_HARD", 0.0, 1

# =============================================================================
# Main Logic
# =============================================================================

def build_generation_plan(n_waves: int, ratios, rng: np.random.Generator):
    plan = []
    allocated = 0
    for func, r in ratios:
        cnt = int(n_waves * float(r))
        plan.extend([func] * cnt)
        allocated += cnt
    
    remainder = n_waves - allocated
    if remainder > 0:
        plan.extend([ratios[0][0]] * remainder)
        
    rng.shuffle(plan)
    return plan

def main():
    ap = argparse.ArgumentParser(description="Generate PREDICT data (Normal + HARD modes).")
    ap.add_argument("--out", default="data/raw/data_predict.csv")
    ap.add_argument("--n_waves", type=int, default=200)
    ap.add_argument("--dt_ms", type=float, default=0.01)
    ap.add_argument("--t_end_ms", type=float, default=9.9)
    ap.add_argument("--predict_noise_scale", type=float, default=1.0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_ms = np.arange(0.0, args.t_end_ms + 1e-12, args.dt_ms)
    t_s = t_ms / 1000.0

    # แบ่งสัดส่วน Normal vs Hard (รวมๆ กันให้ได้ 1.0)
    ratios = [
        # Type 0: Step
        (generate_step_response,        0.14),
        (generate_step_response_HARD,   0.14),
        
        # Type 1: Oscillation
        (generate_high_start_oscillation,      0.12),
        (generate_high_start_oscillation_HARD, 0.12),

        # Type 2: Triangle
        (generate_continuous_triangular_pulses,      0.05),
        (generate_continuous_triangular_pulses_HARD, 0.05),

        # Type 3: Sine
        (generate_low_swing_sine_wave,      0.04),

        # Type 4: Overdamped (รวม 4.1 ด้วย)
        (generate_overdamped_decay,       0.05),
        (generate_overdamped_decay_HARD,  0.05),
        (generate_overdamped_decay1,      0.10), # Type 4.1 ใช้ตัวเดิม (เพราะมัน overshoot อยู่แล้ว)

        # Type 5: Pulse Train
        (generate_pulse_train,       0.05),
        (generate_pulse_train_HARD,  0.05),
    ]

    master_rng = np.random.default_rng(20251111) 
    gen_sequence = build_generation_plan(args.n_waves, ratios, master_rng)

    rows = []
    print(f"Generating MIXED (Normal + HARD) PREDICT dataset ({args.n_waves} waves)...")

    for wave_id, gen_func in enumerate(gen_sequence, start=1):
        final_value = float(master_rng.uniform(0.5, 3.5))
        band_pct = float(master_rng.uniform(0.05, 0.15))
        band = final_value * band_pct
        low = final_value - band / 2.0
        high = final_value + band / 2.0
        
        settle_time_ms = float(master_rng.uniform(2.0, 8.0))
        settle_s = settle_time_ms / 1000.0

        wave_rng = np.random.default_rng(700000 + wave_id)

        y, used_sd, type_name, _, _ = gen_func(
           t_s, final_value, settle_s, low, high, wave_rng
        )

        # Add Noise
        extra_rng = np.random.default_rng(900000 + wave_id)
        extra_sd = max(float(used_sd) * args.predict_noise_scale, 1e-6)
        y = y + extra_rng.normal(0.0, extra_sd, size=len(y))

        for i, (tm, val) in enumerate(zip(t_ms, y)):
            rows.append({
                "wave_id": wave_id,
                "sample": i,
                "time_ms": float(tm),
                "value": float(val),
                "low_limit": float(low),
                "high_limit": float(high),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved MIXED PREDICT data to: {out_path}")

if __name__ == "__main__":
    main()