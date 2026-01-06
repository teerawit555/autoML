# scripts/generate_train_sample.py
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
 
# ==========================================
# 1. Helper Functions (Utilities)
# ==========================================
 
def apply_cosine_taper_settling(signal_array, time_vector, settling_time_s, target_value):
    """
    Applies a Cosine Taper to force the signal to settle smoothly to the final value.
    (Previously: apply_smooth_settle)
    """
    fade_mask = np.zeros_like(time_vector)
    active_idx = time_vector < settling_time_s
    if np.any(active_idx):
        t_ratio = time_vector[active_idx] / settling_time_s
        # Cosine Taper: Starts at 1.0 -> Ends at 0.0 (S-Curve)
        fade_mask[active_idx] = 0.5 * (1 + np.cos(np.pi * t_ratio))
    deviation = signal_array - target_value
    smoothed_signal = target_value + (deviation * fade_mask)
    return smoothed_signal
 
def add_post_settle_noise(signal_array, time_vector, settling_time_s, target_value, random_gen):
    """
    Adds realistic background noise and intermittent burst noise after the settling time.
    (Previously: apply_post_settle_noise)
    """
    # Base floor noise (Very low level)
    base_floor_sd = max(target_value * random_gen.uniform(0.0001, 0.0003), 1e-6)
    signal_array += random_gen.normal(0.0, base_floor_sd, size=len(time_vector))
 
    # Burst noise logic
    settle_idx = np.searchsorted(time_vector, settling_time_s)
    remaining_len = len(time_vector) - settle_idx
    burst_max_intensity = base_floor_sd
 
    # Only add burst noise if there is enough space remaining
    if remaining_len > 50:
        num_bursts = random_gen.integers(1, 4)
        burst_intensity = max(target_value * random_gen.uniform(0.002, 0.006), 0.001)
        burst_max_intensity = burst_intensity
 
        for _ in range(num_bursts):
            burst_width = random_gen.integers(50, 100)
            if remaining_len > burst_width:
                offset = random_gen.integers(0, remaining_len - burst_width)
                start_abs = settle_idx + offset
                end_abs = start_abs + burst_width
                # Add burst
                signal_array[start_abs:end_abs] += random_gen.normal(0.0, burst_intensity, size=burst_width)
    return signal_array, burst_max_intensity
 
# ==========================================
# 2. Signal Generator Functions (Refactored Names)
# ==========================================
 
def generate_step_response(time_vector, target_value, settling_time_s, limit_low, limit_high, random_gen):
    """
    Type 0: Standard Step Response.
    Generates a step response with ringing/overshoot and smooth settling.
    """
    freq_hz = random_gen.uniform(100, 1200)
    angular_freq = 2 * np.pi * freq_hz
    band_half = (limit_high - limit_low) / 2.0
    # Determine overshoot scale
    if target_value < 1.0:
        overshoot_scale = random_gen.uniform(1.5, 8.0)
    else:
        overshoot_scale = random_gen.uniform(1.5, 15.0)
 
    direction = random_gen.choice([1, -1])
    amplitude_0 = band_half * overshoot_scale * direction
 
    decay_factor = random_gen.uniform(2.5, 6.0) 
    tau = max(settling_time_s / decay_factor, 1e-6)
 
    rise_factor = random_gen.uniform(3.0, 10.0)
    time_constant_rise = max(settling_time_s / rise_factor, 1e-6)
 
    # Physical Model
    base_response = target_value * (1 - np.exp(-time_vector / time_constant_rise))
    ringing = amplitude_0 * np.exp(-time_vector / tau) * np.sin(angular_freq * time_vector)
    raw_signal = base_response + ringing
 
    # Post-processing
    smooth_signal = apply_cosine_taper_settling(raw_signal, time_vector, settling_time_s, target_value)
    final_signal, max_noise = add_post_settle_noise(smooth_signal, time_vector, settling_time_s, target_value, random_gen)
    return final_signal, max_noise, "type0_step_response"
 
def generate_high_start_oscillation(time_vector, target_value, settling_time_s, limit_low, limit_high, random_gen):
    """
    Type 1: High Start Oscillation.
    Generates a signal starting with high amplitude oscillation that settles smoothly.
    """
    num_cycles = random_gen.uniform(1.5, 3.5) 
    freq_hz = num_cycles / settling_time_s
    angular_freq = 2 * np.pi * freq_hz
    start_amplitude = (limit_high - limit_low) * random_gen.uniform(2.0, 4.0)
    oscillation = start_amplitude * np.cos(angular_freq * time_vector)
    raw_signal = target_value + oscillation
            
    smooth_signal = apply_cosine_taper_settling(raw_signal, time_vector, settling_time_s, target_value)
    final_signal, max_noise = add_post_settle_noise(smooth_signal, time_vector, settling_time_s, target_value, random_gen)
    return final_signal, max_noise, "type1_high_start_osc"

def generate_continuous_triangular_pulses(time_vector, target_value, settling_time_s, limit_low, limit_high, random_gen):
    """
    Type 2: Continuous Triangular Pulse Train.
    Generates continuous triangular pulses with variable or constant Height/Period.
    (No noise, continuous signal).
    """
    signal_array = np.full_like(time_vector, target_value)
    # Configuration flags
    is_height_const = random_gen.choice([True, False])
    is_period_const = random_gen.choice([True, False])
    subtype_str = f"type2_tri_H{'const' if is_height_const else 'var'}_T{'const' if is_period_const else 'var'}"
 
    # Parameters
    avg_height = (limit_high - limit_low) * random_gen.uniform(0.5, 1.5)
    avg_period = random_gen.uniform(settling_time_s / 3.0, settling_time_s / 1.5)
    pulse_width = avg_period * random_gen.uniform(0.15, 0.3)
 
    current_time = random_gen.uniform(0, avg_period * 0.3)
 
    while current_time < time_vector[-1]:
        # Determine Height
        if is_height_const:
            height = avg_height
        else:
            height = avg_height * random_gen.uniform(0.5, 1.5)
 
        # Draw Triangle
        t_start = current_time
        t_peak = current_time + pulse_width / 2
        t_end = current_time + pulse_width
 
        # Rise phase
        mask_rise = (time_vector >= t_start) & (time_vector < t_peak)
        if np.any(mask_rise):
            signal_array[mask_rise] += (height / (pulse_width/2)) * (time_vector[mask_rise] - t_start)
        # Fall phase
        mask_fall = (time_vector >= t_peak) & (time_vector < t_end)
        if np.any(mask_fall):
            signal_array[mask_fall] += height - (height / (pulse_width/2)) * (time_vector[mask_fall] - t_peak)
 
        # Determine next Period
        if is_period_const:
            period = avg_period
        else:
            period = avg_period * random_gen.uniform(0.7, 1.3)
        current_time += period
    return signal_array, 0.0, subtype_str
 
def generate_low_swing_sine_wave(time_vector, target_value, settling_time_s, limit_low, limit_high, random_gen):
    """
    Type 3: Low Swing Sine Wave (Sea Wave).
    Generates a continuous sine wave with low amplitude and very low noise.
    """
    signal_array = np.full_like(time_vector, target_value)
    # Wave Parameters (Freq 200-500Hz)
    freq_hz = random_gen.uniform(100, 300) 
    angular_freq = 2 * np.pi * freq_hz
    # Amplitude: Low swing
    amplitude = (limit_high - limit_low) * random_gen.uniform(0.1, 0.25)
    # Phase shift
    phase = random_gen.uniform(0, 2 * np.pi)
    # Generate Base Wave
    oscillation = amplitude * np.sin(angular_freq * time_vector + phase)
    signal_array += oscillation
    # Add Noise (0-0.002%)
    noise_level_pct = random_gen.uniform(0.0, 0.00002)
    sd_noise = max(target_value * noise_level_pct, 1e-6)
    noise = random_gen.normal(0.0, sd_noise, size=len(time_vector))
    signal_array += noise
    return signal_array, sd_noise, "type3_sea_wave"
 
# ==========================================
# 3. Main Script (Standard Structure)
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
 
    # Update list with new formal function names
    signal_generators = [
        generate_step_response,
        generate_high_start_oscillation, 
        generate_continuous_triangular_pulses,
        generate_low_swing_sine_wave,
    ]
 
    print(f"Generating TRAINING dataset ({args.n_waves} waves)...")
    print(f"Active Types: {[func.__name__ for func in signal_generators]}")
 
    for wave_id in range(1, args.n_waves + 1):
        final_value = master_rng.uniform(0.5, 3.5)
        band_pct = master_rng.uniform(0.05, 0.15)
        band = final_value * band_pct
        low, high = final_value - band/2, final_value + band/2
        settle_time_ms = master_rng.uniform(2.0, 8.0)
        settle_s = settle_time_ms / 1000.0
 
        gen_func = master_rng.choice(signal_generators)
        wave_rng = np.random.default_rng(100000 + wave_id)
 
        y, used_sd, type_name = gen_func(
            t_s, final_value, settle_s, low, high, wave_rng
        )
 
        for i, (tm, val) in enumerate(zip(t_ms, y)):
            rows.append({
                "wave_id": wave_id,
                "sample": i,
                "time_ms": float(tm),
                "value": float(val),
                "sd": float(used_sd),
                "low_limit": float(low),
                "high_limit": float(high),
                "signal_type": type_name,
                # "wait_time_ms": float(settle_time_ms)
            })
 
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Successfully saved TRAINING data to: {out_path}")
    print(f"Total rows generated: {len(df)}")
    print("Distribution:", df.groupby('wave_id')['signal_type'].first().value_counts())
 
if __name__ == "__main__":
    main()



# scripts/generate_train_sample.py
# import argparse
# import numpy as np
# import pandas as pd
# from pathlib import Path
 
# # ==========================================
# # 1. Helper Functions (Utilities)
# # ==========================================
 
# def apply_cosine_taper_settling(signal_array, time_vector, settling_time_s, target_value):
#     """ Applies a Cosine Taper to force the signal to settle smoothly. """
#     fade_mask = np.zeros_like(time_vector)
#     active_idx = time_vector < settling_time_s
   
#     if np.any(active_idx):
#         t_ratio = time_vector[active_idx] / settling_time_s
#         fade_mask[active_idx] = 0.5 * (1 + np.cos(np.pi * t_ratio))
   
#     deviation = signal_array - target_value
#     smoothed_signal = target_value + (deviation * fade_mask)
#     return smoothed_signal
 
# def add_post_settle_noise(signal_array, time_vector, settling_time_s, target_value, random_gen):
#     """ Adds realistic background noise and intermittent burst noise. """
#     # 1. Floor Noise (มีตลอดเส้น นิดๆ)
#     base_floor_sd = max(target_value * random_gen.uniform(0.0001, 0.0003), 1e-6)
#     signal_array += random_gen.normal(0.0, base_floor_sd, size=len(time_vector))
 
#     # 2. Burst Noise (มีหลัง Settle)
#     settle_idx = np.searchsorted(time_vector, settling_time_s)
#     remaining_len = len(time_vector) - settle_idx
#     burst_max_intensity = base_floor_sd
 
#     if remaining_len > 50:
#         num_bursts = random_gen.integers(1, 4)
#         burst_intensity = max(target_value * random_gen.uniform(0.002, 0.006), 0.001)
#         burst_max_intensity = burst_intensity
 
#         for _ in range(num_bursts):
#             burst_width = random_gen.integers(50, 100)
#             if remaining_len > burst_width:
#                 offset = random_gen.integers(0, remaining_len - burst_width)
#                 start_abs = settle_idx + offset
#                 end_abs = start_abs + burst_width
#                 signal_array[start_abs:end_abs] += random_gen.normal(0.0, burst_intensity, size=burst_width)
   
#     return signal_array, burst_max_intensity
 
# # ==========================================
# # 2. Signal Generator Functions
# # ==========================================
 
# def generate_step_response(time_vector, target_value, settling_time_s, limit_low, limit_high, random_gen):
#     """ Type 0: Standard Step Response. """
#     freq_hz = random_gen.uniform(100, 1200)
#     angular_freq = 2 * np.pi * freq_hz
#     band_half = (limit_high - limit_low) / 2.0
   
#     if target_value < 1.0: overshoot_scale = random_gen.uniform(1.5, 8.0)
#     else: overshoot_scale = random_gen.uniform(1.5, 15.0)
 
#     direction = random_gen.choice([1, -1])
#     amplitude_0 = band_half * overshoot_scale * direction
 
#     decay_factor = random_gen.uniform(2.5, 6.0)
#     tau = max(settling_time_s / decay_factor, 1e-6)
 
#     rise_factor = random_gen.uniform(3.0, 10.0)
#     time_constant_rise = max(settling_time_s / rise_factor, 1e-6)
 
#     base_response = target_value * (1 - np.exp(-time_vector / time_constant_rise))
#     ringing = amplitude_0 * np.exp(-time_vector / tau) * np.sin(angular_freq * time_vector)
#     raw_signal = base_response + ringing
 
#     smooth_signal = apply_cosine_taper_settling(raw_signal, time_vector, settling_time_s, target_value)
#     final_signal, max_noise = add_post_settle_noise(smooth_signal, time_vector, settling_time_s, target_value, random_gen)
#     return final_signal, max_noise, "type0_step_response"
 
# def generate_high_start_oscillation(time_vector, target_value, settling_time_s, limit_low, limit_high, random_gen):
#     """ Type 1: High Start Oscillation. """
#     num_cycles = random_gen.uniform(1.5, 3.5)
#     freq_hz = num_cycles / settling_time_s
#     angular_freq = 2 * np.pi * freq_hz
   
#     start_amplitude = (limit_high - limit_low) * random_gen.uniform(2.0, 4.0)
#     oscillation = start_amplitude * np.cos(angular_freq * time_vector)
#     raw_signal = target_value + oscillation
 
#     smooth_signal = apply_cosine_taper_settling(raw_signal, time_vector, settling_time_s, target_value)
#     final_signal, max_noise = add_post_settle_noise(smooth_signal, time_vector, settling_time_s, target_value, random_gen)
#     return final_signal, max_noise, "type1_high_start_osc"
 
# def generate_continuous_triangular_pulses(time_vector, target_value, settling_time_s, limit_low, limit_high, random_gen):
#     """ Type 2: Continuous Triangular Pulse Train. """
#     signal_array = np.full_like(time_vector, target_value)
#     is_height_const = random_gen.choice([True, False])
#     is_period_const = random_gen.choice([True, False])
#     subtype_str = f"type2_tri_H{'const' if is_height_const else 'var'}_T{'const' if is_period_const else 'var'}"
 
#     avg_height = (limit_high - limit_low) * random_gen.uniform(0.5, 1.5)
#     avg_period = random_gen.uniform(settling_time_s / 3.0, settling_time_s / 1.5)
#     pulse_width = avg_period * random_gen.uniform(0.15, 0.3)
#     current_time = random_gen.uniform(0, avg_period * 0.3)
 
#     while current_time < time_vector[-1]:
#         if is_height_const: height = avg_height
#         else: height = avg_height * random_gen.uniform(0.5, 1.5)
 
#         t_start, t_peak, t_end = current_time, current_time + pulse_width / 2, current_time + pulse_width
       
#         mask_rise = (time_vector >= t_start) & (time_vector < t_peak)
#         if np.any(mask_rise): signal_array[mask_rise] += (height / (pulse_width/2)) * (time_vector[mask_rise] - t_start)
           
#         mask_fall = (time_vector >= t_peak) & (time_vector < t_end)
#         if np.any(mask_fall): signal_array[mask_fall] += height - (height / (pulse_width/2)) * (time_vector[mask_fall] - t_peak)
 
#         if is_period_const: period = avg_period
#         else: period = avg_period * random_gen.uniform(0.7, 1.3)
#         current_time += period
   
#     return signal_array, 0.0, subtype_str
 
# def generate_low_swing_sine_wave(time_vector, target_value, settling_time_s, limit_low, limit_high, random_gen):
#     """ Type 3: Low Swing Sine Wave (Start at Center). """
#     signal_array = np.full_like(time_vector, target_value)
#     freq_hz = random_gen.uniform(100, 300)
#     angular_freq = 2 * np.pi * freq_hz
#     amplitude = (limit_high - limit_low) * random_gen.uniform(0.1, 0.25)
   
#     # Phase shift: 0 or PI to start at center
#     phase = random_gen.choice([0.0, np.pi])
   
#     oscillation = amplitude * np.sin(angular_freq * time_vector + phase)
#     signal_array += oscillation
   
#     noise_level_pct = random_gen.uniform(0.0, 0.00002)
#     sd_noise = max(target_value * noise_level_pct, 1e-6)
#     signal_array += random_gen.normal(0.0, sd_noise, size=len(time_vector))
   
#     return signal_array, sd_noise, "type3_sea_wave"
 
# def generate_overdamped_decay(time_vector, target_value, settling_time_s, limit_low, limit_high, random_gen):
#     """
#     Type 4: Overdamped Decay (Signal 2 Style).
#     - Starts High.
#     - Smooth exponential decay to Final Value (No Ringing).
#     - Adds small noise after settle.
#     """
#     # 1. Start Amplitude (Start High)
#     start_amplitude = (limit_high - limit_low) * random_gen.uniform(1.5, 3.0)
   
#     # 2. Decay Parameter (Overdamped - ลงแบบเนิบๆ)
#     tau = settling_time_s / random_gen.uniform(3.0, 5.0)
   
#     # 3. Math Model: Exponential Decay (No Oscillation)
#     decay_curve = start_amplitude * np.exp(-time_vector / tau)
#     raw_signal = target_value + decay_curve
 
#     # 4. Post-processing (บังคับหางให้นิ่ง)
#     smooth_signal = apply_cosine_taper_settling(raw_signal, time_vector, settling_time_s, target_value)
   
#     # 5. [IMPORTANT] Add post-settle noise (เติม Noise นิดๆ ตามที่ขอ)
#     # ฟังก์ชันนี้จะเติมทั้ง Floor Noise และ Burst Noise (ขยุกขยิกเล็กน้อย)
#     final_signal, max_noise = add_post_settle_noise(smooth_signal, time_vector, settling_time_s, target_value, random_gen)
   
#     return final_signal, max_noise, "type4_overdamped_decay"
 
# # ==========================================
# # 3. Main Script
# # ==========================================
 
# def main():
#     ap = argparse.ArgumentParser(description="Generate synthetic TRAINING waveform data.")
#     ap.add_argument("--out", default="data/raw/data1000samples_train.csv", help="Output CSV path")
#     ap.add_argument("--n_waves", type=int, default=1000, help="Number of waveforms to generate")
#     ap.add_argument("--dt_ms", type=float, default=0.01, help="Time step in milliseconds")
#     ap.add_argument("--t_end_ms", type=float, default=9.9, help="End time in milliseconds")
#     args = ap.parse_args()
 
#     out_path = Path(args.out)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
 
#     t_ms = np.arange(0, args.t_end_ms + 1e-12, args.dt_ms)
#     t_s = t_ms / 1000.0
 
#     rows = []
#     master_rng = np.random.default_rng(12345)
 
#     # [UPDATED] Added Type 4 to the list
#     signal_generators = [
#         generate_step_response,                 # Type 0
#         generate_high_start_oscillation,        # Type 1
#         generate_continuous_triangular_pulses,  # Type 2
#         generate_low_swing_sine_wave,           # Type 3
#         generate_overdamped_decay,              # Type 4
#     ]
 
#     print(f"Generating TRAINING dataset ({args.n_waves} waves)...")
   
#     # --- Balanced Distribution Logic ---
#     n_types = len(signal_generators)
#     count_per_type = args.n_waves // n_types
#     remainder = args.n_waves % n_types
 
 
#     gen_sequence = []
#     for gen in signal_generators:
#         gen_sequence.extend([gen] * count_per_type)
   
#     if remainder > 0:
#         extras = master_rng.choice(signal_generators, size=remainder)
#         gen_sequence.extend(extras)
       
#     master_rng.shuffle(gen_sequence)
#     perm_indices = master_rng.permutation(len(gen_sequence))
#     shuffled_gens = [gen_sequence[i] for i in perm_indices]
 
#     for wave_id in range(1, args.n_waves + 1):
#         final_value = master_rng.uniform(0.5, 3.5)
#         band_pct = master_rng.uniform(0.05, 0.15)
#         band = final_value * band_pct
#         low, high = final_value - band/2, final_value + band/2
       
#         settle_time_ms = master_rng.uniform(2.0, 8.0)
#         settle_s = settle_time_ms / 1000.0
 
#         gen_func = shuffled_gens[wave_id - 1]
#         wave_rng = np.random.default_rng(100000 + wave_id)
 
#         y, used_sd, type_name = gen_func(
#             t_s, final_value, settle_s, low, high, wave_rng
#         )
 
#         for i, (tm, val) in enumerate(zip(t_ms, y)):
#             rows.append({
#                 "wave_id": wave_id,
#                 "sample": i,
#                 "time_ms": float(tm),
#                 "value": float(val),
#                 "sd": float(used_sd),
#                 "low_limit": float(low),
#                 "high_limit": float(high),
#                 "signal_type": type_name,
#             })
 
#     df = pd.DataFrame(rows)
#     df.to_csv(out_path, index=False)
#     print(f"Successfully saved TRAINING data to: {out_path}")
 
# if __name__ == "__main__":
#     main()
 
 
 
 
 
 
 
