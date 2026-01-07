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


# scripts/generate_train_sample.py
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# 1. Helper Functions (Utilities)
# ==========================================

def apply_cosine_taper_settling(signal_array, time_vector, settling_time_s, target_value):
    """ Applies a Cosine Taper to force the signal to settle smoothly. """
    fade_mask = np.zeros_like(time_vector)
    active_idx = time_vector < settling_time_s
    
    if np.any(active_idx):
        t_ratio = time_vector[active_idx] / settling_time_s
        fade_mask[active_idx] = 0.5 * (1 + np.cos(np.pi * t_ratio))
    
    deviation = signal_array - target_value
    smoothed_signal = target_value + (deviation * fade_mask)
    return smoothed_signal

def add_post_settle_noise(signal_array, time_vector, settling_time_s, target_value, random_gen, probability=0.5, smoothness=10):
    """ 
    1. Adds Floor Noise (Always).
    2. Randomly decides whether to add extra continuous noise after settling.
       probability: Chance of adding extra noise (0.0 to 1.0). Default 0.5 (50%).
    """
    
    # --- 1. Floor Noise (มีตลอดเส้นเสมอ) ---
    base_floor_sd = max(target_value * random_gen.uniform(0.0001, 0.0003), 1e-6)
    signal_array += random_gen.normal(0.0, base_floor_sd, size=len(time_vector))

    # ค่าความแรง Noise เริ่มต้น (กรณีไม่เกิด Noise เพิ่ม)
    final_noise_intensity = base_floor_sd 

    # --- 2. ตัดสินใจว่าจะใส่ Noise หลัง Settle หรือไม่? ---
    # สุ่มตัวเลข 0.0 - 1.0 ถ้าค่าน้อยกว่า probability ให้ทำ (เช่น 0.5 คือโอกาส 50%)
    should_add_noise = random_gen.random() < probability

    if should_add_noise:
        settle_idx = np.searchsorted(time_vector, settling_time_s)
        remaining_len = len(time_vector) - settle_idx
        
        if remaining_len > 0:
            # ความแรง Noise
            post_settle_sd = max(target_value * random_gen.uniform(0.001, 0.00155), 0.0001)
            
            # อัปเดตค่าความแรงสูงสุดที่จะส่งคืน
            final_noise_intensity = post_settle_sd 

            raw_noise = random_gen.normal(0.0, post_settle_sd, size=remaining_len)
            
            if smoothness > 1 and remaining_len > smoothness:
                # 2. ใช้ Moving Average เกลี่ย Noise ให้เป็นเส้นยึกๆ (Low Pass Filter)
                kernel = np.ones(smoothness) / smoothness
                wiggly_noise = np.convolve(raw_noise, kernel, mode='same')
                
                # 3. *สำคัญ* คูณความสูงกลับเข้าไป (Gain Compensation)
                # เพราะการเกลี่ยจะทำให้ยอดคลื่นเตี้ยลง เราต้องดึงกลับให้ "สูงเท่าเดิม"
                # ตามทฤษฎีคือคูณด้วย sqrt(window_size)
                wiggly_noise *= np.sqrt(smoothness)
                
                signal_array[settle_idx:] += wiggly_noise
            else:
                # ถ้า Smoothness = 1 หรือข้อมูลสั้นเกิน ก็ใส่แบบดิบๆ (เส้นหนา)
                signal_array[settle_idx:] += raw_noise
    
    return signal_array, final_noise_intensity

# ==========================================
# 2. Signal Generator Functions
# ==========================================

def generate_step_response(time_vector, target_value, settling_time_s, limit_low, limit_high, random_gen):
    """ Type 0: Standard Step Response. """
    freq_hz = random_gen.uniform(100, 1200)
    angular_freq = 2 * np.pi * freq_hz
    band_half = (limit_high - limit_low) / 2.0
    
    if target_value < 1.0: overshoot_scale = random_gen.uniform(1.5, 8.0)
    else: overshoot_scale = random_gen.uniform(1.5, 15.0)

    direction = random_gen.choice([1, -1])
    amplitude_0 = band_half * overshoot_scale * direction

    decay_factor = random_gen.uniform(2.5, 6.0) 
    tau = max(settling_time_s / decay_factor, 1e-6)

    rise_factor = random_gen.uniform(3.0, 10.0)
    time_constant_rise = max(settling_time_s / rise_factor, 1e-6)

    base_response = target_value * (1 - np.exp(-time_vector / time_constant_rise))
    ringing = amplitude_0 * np.exp(-time_vector / tau) * np.sin(angular_freq * time_vector)
    raw_signal = base_response + ringing

    smooth_signal = apply_cosine_taper_settling(raw_signal, time_vector, settling_time_s, target_value)
    final_signal, max_noise = add_post_settle_noise(smooth_signal, time_vector, settling_time_s, target_value, random_gen)
    return final_signal, max_noise, "type0_step_response"

def generate_high_start_oscillation(time_vector, target_value, settling_time_s, limit_low, limit_high, random_gen):
    """ Type 1: High Start Oscillation. """
    num_cycles = random_gen.uniform(1.5, 3.5) 
    freq_hz = num_cycles / settling_time_s
    angular_freq = 2 * np.pi * freq_hz
    
    start_amplitude = (limit_high - limit_low) * random_gen.uniform(2.0, 4.0)
    oscillation = start_amplitude * np.cos(angular_freq * time_vector)
    raw_signal = target_value + oscillation

    smooth_signal = apply_cosine_taper_settling(raw_signal, time_vector, settling_time_s, target_value)
    final_signal, max_noise = add_post_settle_noise(smooth_signal, time_vector, settling_time_s, target_value, random_gen)
    return final_signal, max_noise, "type1_Damped_Osc"

def generate_continuous_triangular_pulses(time_vector, target_value, settling_time_s, limit_low, limit_high, random_gen):
    """ Type 2: Continuous Triangular Pulse Train. """
    signal_array = np.full_like(time_vector, target_value)
    is_height_const = random_gen.choice([True, False])
    is_period_const = random_gen.choice([True, False])

    avg_height = (limit_high - limit_low) * random_gen.uniform(0.5, 1.5)
    avg_period = random_gen.uniform(settling_time_s / 3.0, settling_time_s / 1.5)
    pulse_width = avg_period * random_gen.uniform(0.15, 0.3) 
    current_time = random_gen.uniform(0, avg_period * 0.3)

    while current_time < time_vector[-1]:
        if is_height_const: height = avg_height
        else: height = avg_height * random_gen.uniform(0.5, 1.5)

        t_start, t_peak, t_end = current_time, current_time + pulse_width / 2, current_time + pulse_width
        
        mask_rise = (time_vector >= t_start) & (time_vector < t_peak)
        if np.any(mask_rise): signal_array[mask_rise] += (height / (pulse_width/2)) * (time_vector[mask_rise] - t_start)
            
        mask_fall = (time_vector >= t_peak) & (time_vector < t_end)
        if np.any(mask_fall): signal_array[mask_fall] += height - (height / (pulse_width/2)) * (time_vector[mask_fall] - t_peak)

        if is_period_const: period = avg_period
        else: period = avg_period * random_gen.uniform(0.7, 1.3)
        current_time += period
    
    return signal_array, 0.0, "type2_Triangle_Wave"

def generate_low_swing_sine_wave(time_vector, target_value, settling_time_s, limit_low, limit_high, random_gen):
    """ Type 3: Low Swing Sine Wave (Start at Center). """
    signal_array = np.full_like(time_vector, target_value)
    freq_hz = random_gen.uniform(100, 300) 
    angular_freq = 2 * np.pi * freq_hz
    amplitude = (limit_high - limit_low) * random_gen.uniform(0.1, 0.25)
    
    # Phase shift: 0 or PI to start at center
    phase = random_gen.choice([0.0, np.pi])
    
    oscillation = amplitude * np.sin(angular_freq * time_vector + phase)
    signal_array += oscillation
    
    noise_level_pct = random_gen.uniform(0.0, 0.00002)
    sd_noise = max(target_value * noise_level_pct, 1e-6)
    signal_array += random_gen.normal(0.0, sd_noise, size=len(time_vector))
    
    return signal_array, sd_noise, "type3_sine_wave"

def generate_overdamped_decay(time_vector, target_value, settling_time_s, limit_low, limit_high, random_gen):
    """
    Type 4: Overdamped Decay (Signal 2 Style).
    - Starts High.
    - Smooth exponential decay to Final Value (No Ringing).
    - Adds small noise after settle.
    """
    # 1. Start Amplitude (Start High)
    start_amplitude = (limit_high - limit_low) * random_gen.uniform(1.5, 3.0)
    
    # 2. Decay Parameter (Overdamped - ลงแบบเนิบๆ)
    tau = settling_time_s / random_gen.uniform(3.0, 5.0)
    
    # 3. Math Model: Exponential Decay (No Oscillation)
    decay_curve = start_amplitude * np.exp(-time_vector / tau)
    raw_signal = target_value + decay_curve

    # 4. Post-processing (บังคับหางให้นิ่ง)
    smooth_signal = apply_cosine_taper_settling(raw_signal, time_vector, settling_time_s, target_value)
    
    # 5. [IMPORTANT] Add post-settle noise (เติม Noise นิดๆ ตามที่ขอ)
    # ฟังก์ชันนี้จะเติมทั้ง Floor Noise และ Burst Noise (ขยุกขยิกเล็กน้อย)
    final_signal, max_noise = add_post_settle_noise(smooth_signal, time_vector, settling_time_s, target_value, random_gen)
    
    return final_signal, max_noise, "type4_overdamped_decay"

# ==========================================
# 3. Main Script
# ==========================================

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic TRAINING waveform data.")
    ap.add_argument("--out", default="data/raw/data1000samples_train.csv", help="Output CSV path")
    ap.add_argument("--n_waves", type=int, default=1000, help="Number of waveforms to generate")
    ap.add_argument("--dt_ms", type=float, default=0.01, help="Time step in milliseconds")
    ap.add_argument("--t_end_ms", type=float, default=9.9, help="End time in milliseconds")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_ms = np.arange(0, args.t_end_ms + 1e-12, args.dt_ms)
    t_s = t_ms / 1000.0

    rows = []
    master_rng = np.random.default_rng(12345) 

    # [UPDATED] Added Type 4 to the list
    signal_generators = [
        generate_step_response,                 # Type 0
        generate_high_start_oscillation,        # Type 1
        generate_continuous_triangular_pulses,  # Type 2
        generate_low_swing_sine_wave,           # Type 3
        generate_overdamped_decay,              # Type 4 
    ]

    print(f"Generating TRAINING dataset ({args.n_waves} waves)...")
    
    # --- Balanced Distribution Logic ---
    n_types = len(signal_generators)
    count_per_type = args.n_waves // n_types
    remainder = args.n_waves % n_types


    gen_sequence = []
    for gen in signal_generators:
        gen_sequence.extend([gen] * count_per_type)
    
    if remainder > 0:
        extras = master_rng.choice(signal_generators, size=remainder)
        gen_sequence.extend(extras)
        
    master_rng.shuffle(gen_sequence)
    perm_indices = master_rng.permutation(len(gen_sequence))
    shuffled_gens = [gen_sequence[i] for i in perm_indices]

    for wave_id in range(1, args.n_waves + 1):
        final_value = master_rng.uniform(0.5, 3.5)
        band_pct = master_rng.uniform(0.05, 0.15)
        band = final_value * band_pct
        low, high = final_value - band/2, final_value + band/2
        
        settle_time_ms = master_rng.uniform(2.0, 8.0)
        settle_s = settle_time_ms / 1000.0

        gen_func = shuffled_gens[wave_id - 1]
        wave_rng = np.random.default_rng(100000 + wave_id)

        y, used_sd, type_name = gen_func(
            t_s, final_value, settle_s, low, high, wave_rng
        )

        for i, (tm, val) in enumerate(zip(t_ms, y)):
            rows.append({
                "wave_id": wave_id,
                "type": type_name,
                "sample": i,
                "time_ms": float(tm),
                "value": float(val),
                "sd": float(used_sd),
                "low_limit": float(low),
                "high_limit": float(high),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Successfully saved TRAINING data to: {out_path}")

if __name__ == "__main__":
    main()