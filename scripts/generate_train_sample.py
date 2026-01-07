# scripts/generate_train_sample.py
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
 
# ==========================================
# 1. Helper Functions (Utilities)
# ==========================================

def apply_cosine_taper_settling(signal_array, time_vector, settling_time_s, target_value, strength=1):
    """
    strength=1.0 -> บังคับนิ่งเต็มที่ (เหมือนเดิม)
    strength<1.0 -> ปล่อยให้ยังเหลือ deviation บ้าง (ดูเหมือนจริงขึ้น)
    """
    fade_mask = np.zeros_like(time_vector)
    active_idx = time_vector < settling_time_s

    if np.any(active_idx):
        t_ratio = time_vector[active_idx] / max(settling_time_s, 1e-12)
        fade_mask[active_idx] = 0.5 * (1 + np.cos(np.pi * t_ratio))

    deviation = signal_array - target_value
    smoothed_signal = target_value + (deviation * fade_mask)

    # ✅ key: blend กลับกับสัญญาณเดิม เพื่อลดความ “เป๊ะ”
    # strength=1 -> ใช้ smoothed ล้วน
    # strength=0 -> ไม่ taper เลย
    return strength * smoothed_signal + (1.0 - strength) * signal_array


def add_post_settle_noise(
    signal_array, time_vector, settling_time_s, target_value, rng,
    probability=0.9,
    post_sd_scale=(0.0008, 0.0014),
    smoothness_range=(20, 40),
    add_wobble_prob=0.18,
    wobble_scale=(0.00010, 0.00022),
    wobble_win_range=(35, 80),
):
    """
    เบากว่าเดิม:
    - floor noise (ตลอด)
    - post-settle wiggle (MA 1 ครั้ง)
    - optional wobble (MA ใหญ่ 1 ครั้ง แค่บางเส้น)
    """

    # 1) floor noise
    base_floor_sd = max(target_value * rng.uniform(0.0001, 0.00025), 1e-6)
    signal_array += rng.normal(0.0, base_floor_sd, size=len(time_vector))
    final_sd = base_floor_sd

    settle_idx = np.searchsorted(time_vector, settling_time_s)
    remaining_len = len(time_vector) - settle_idx
    if remaining_len <= 5:
        return signal_array, final_sd

    # 2) post-settle wiggle
    if rng.random() < probability:
        post_sd = max(target_value * rng.uniform(*post_sd_scale), 1e-6)
        final_sd = max(final_sd, post_sd)

        raw = rng.normal(0.0, post_sd, size=remaining_len)

        smoothness = int(rng.integers(smoothness_range[0], smoothness_range[1] + 1))
        smoothness = min(smoothness, remaining_len - 1)

        if smoothness > 2:
            k = np.ones(smoothness) / smoothness
            wig = np.convolve(raw, k, mode="same") * np.sqrt(smoothness)
            signal_array[settle_idx:] += wig
        else:
            signal_array[settle_idx:] += raw

    # 3) optional wobble (ลดโอกาส + ลดแรง)
    if rng.random() < add_wobble_prob:
        wob_sd = max(target_value * rng.uniform(*wobble_scale), 1e-6)
        final_sd = max(final_sd, wob_sd)

        wob = rng.normal(0.0, wob_sd, size=remaining_len)
        win = int(rng.integers(wobble_win_range[0], wobble_win_range[1] + 1))
        win = min(win, remaining_len - 1)

        if win > 3:
            k2 = np.ones(win) / win
            wob = np.convolve(wob, k2, mode="same") * np.sqrt(win)

        signal_array[settle_idx:] += wob

    return signal_array, final_sd

def add_dc_offset_and_drift(y, t, rng, offset_frac=(0.0, 0.01), drift_frac=(0.0, 0.01)):
    """เพิ่ม offset และ linear drift เล็กน้อย"""
    off = rng.uniform(*offset_frac)
    drift = rng.uniform(-drift_frac[1], drift_frac[1])
    return y + off + drift * (t / max(t[-1], 1e-12))

def add_time_delay(t, rng, max_delay_s=0.0006):
    """ทำ delay แบบสุ่ม (เลื่อนเวลา)"""
    d = rng.uniform(0.0, max_delay_s)
    return np.clip(t - d, 0.0, None), d

def add_quantization(y, rng, q_step_frac=0.0005):
    """จำลอง ADC quantization เล็กน้อย"""
    q = max(np.abs(np.mean(y)) * q_step_frac, 1e-6)
    return np.round(y / q) * q

# ==========================================
# 2. Signal Generator Functions (Refactored Names)
# ==========================================
 
def generate_step_response(time_vector, target_value, settling_time_s, limit_low, limit_high, random_gen):
    """ Type 0: Standard Step Response (more realistic post-settle) """
    freq_hz = random_gen.uniform(100, 1200)
    angular_freq = 2 * np.pi * freq_hz
    band_half = (limit_high - limit_low) / 2.0

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

    # (A) random delay
    t_eff, _delay_s = add_time_delay(time_vector, random_gen, max_delay_s=0.0006)

    # base rise: 1-pole or 2-pole
    if random_gen.random() < 0.5:
        base_response = target_value * (1 - np.exp(-t_eff / time_constant_rise))
    else:
        tau2 = max(time_constant_rise * random_gen.uniform(1.5, 4.0), 1e-6)
        base_response = target_value * (1 - 0.6*np.exp(-t_eff/time_constant_rise) - 0.4*np.exp(-t_eff/tau2))

    # ringing: 1-tone or 2-tone mix
    if random_gen.random() < 0.6:
        ringing = amplitude_0 * np.exp(-t_eff / tau) * np.sin(angular_freq * t_eff)
    else:
        freq2 = freq_hz * random_gen.uniform(0.75, 1.25)
        w2 = 2 * np.pi * freq2
        mix = random_gen.uniform(0.2, 0.6)
        ringing = amplitude_0 * np.exp(-t_eff / tau) * (
            (1 - mix) * np.sin(angular_freq * t_eff) + mix * np.sin(w2 * t_eff)
        )

    raw_signal = base_response + ringing

    # optional kick
    if random_gen.random() < 0.25:
        kick_amp = (limit_high - limit_low) * random_gen.uniform(0.15, 0.8)
        kick_amp *= random_gen.choice([1.0, -1.0])
        kick_tau = max(settling_time_s / random_gen.uniform(6.0, 14.0), 1e-6)
        raw_signal += kick_amp * np.exp(-t_eff / kick_tau)

    # ✅ (NEW) make taper less "perfect" like Type 1
    taper_strength = random_gen.uniform(0.78, 0.95)  # ยิ่งต่ำ ยิ่งไม่เป๊ะ
    smooth_signal = apply_cosine_taper_settling(
        raw_signal, time_vector, settling_time_s, target_value,
        strength=taper_strength
    )

    # ✅ (NEW) stronger post-settle wiggle (แต่ไม่ให้ sine-y เกิน)
    final_signal, max_noise = add_post_settle_noise(
        smooth_signal, time_vector, settling_time_s, target_value, random_gen,
        probability=0.95,        # ให้ติด post-noise เกือบทุกเส้น
        smoothness_range=(10, 28),   # กลางๆ: สั่นเป็นริ้ว ไม่เป็น sine ยาว
        post_sd_scale=(0.0009, 0.0016),
        add_wobble_prob=0.35,
        wobble_scale=(0.00012, 0.00030),
    )

    return final_signal, max_noise, "type0_step_response", float(settling_time_s*1000.0), 0



def generate_high_start_oscillation(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """
    Type 1 (optimized):
    - cycles เยอะขึ้นได้
    - valley ไม่ลงที่เดิม (transient bias)
    - valley ลึก -> peak2 มักสูง (coupling แบบเบาและคำนวณเฉพาะช่วง)
    - บางเส้น underdamped ยาว
    """

    band = float(limit_high - limit_low)

    # 0) cycles
    if rng.random() < 0.70:
        num_cycles = int(rng.integers(2, 4))   # 2-3
    else:
        num_cycles = int(rng.integers(4, 7))   # 4-6

    # 1) delay
    t_eff, _ = add_time_delay(time_vector, rng, max_delay_s=0.0004)

    # 2) omega
    freq = num_cycles / max(settling_time_s, 1e-6)
    w = 2 * np.pi * freq

    # jitter บางเส้น
    if rng.random() < 0.35:
        w *= rng.uniform(0.88, 1.12)

    # 3) amplitude + transient bias
    A = band * rng.uniform(2.0, 4.0)

    bias_amp = band * rng.uniform(-0.55, 0.55)
    bias_tau = max(settling_time_s / rng.uniform(0.9, 2.2), 1e-6)
    bias = bias_amp * np.exp(-t_eff / bias_tau)

    # 4) envelope (2-stage แต่เรียบขึ้น)
    tau_slow = max(settling_time_s * rng.uniform(0.65, 1.35), 1e-6)
    tau_fast = max(tau_slow / rng.uniform(2.2, 4.8), 1e-6)
    t_switch = settling_time_s * rng.uniform(0.22, 0.42)

    # underdamped long ringing บางเส้น
    if rng.random() < 0.22:
        tau_slow *= rng.uniform(1.5, 2.8)
        tau_fast *= rng.uniform(2.0, 4.5)
        t_switch = settling_time_s * rng.uniform(0.30, 0.55)

    env = np.where(
        t_eff < t_switch,
        np.exp(-t_eff / tau_slow),
        np.exp(-t_switch / tau_slow) * np.exp(-(t_eff - t_switch) / tau_fast)
    )

    # 5) coupling แบบ “เบา” (คำนวณเฉพาะ early window)
    # early window ~ 1.8 cycles
    T = (2*np.pi) / max(w, 1e-12)
    t_gate_end = 1.8 * T
    early_mask = (t_eff >= 0.0) & (t_eff <= t_gate_end)

    if np.any(early_mask):
        te = t_eff[early_mask]
        eve = env[early_mask]
        bse = bias[early_mask]

        y_early = target_value + bse + (A * eve * np.cos(w * te))
        valley_depth = float(target_value - np.min(y_early))  # >0 => ลงลึก

        depth_ratio = np.clip(valley_depth / max(band, 1e-12), 0.0, 1.8)

        # boost peak2 window ~ 1.0–2.2 cycles
        k_boost = rng.uniform(0.55, 1.05)
        boost_amp = 1.0 + k_boost * depth_ratio

        t1 = 1.0 * T
        t2 = 2.2 * T
        mid_mask = (t_eff >= t1) & (t_eff <= t2)
        if np.any(mid_mask):
            x = (t_eff[mid_mask] - t1) / max((t2 - t1), 1e-12)
            bump = 0.5 - 0.5*np.cos(2*np.pi*x)  # 0..1..0
            env[mid_mask] *= (1.0 + (boost_amp - 1.0) * bump)

    # 6) oscillation (ลด 2-tone เหลือแค่บาง %)
    if rng.random() < 0.18:
        w2 = w * rng.uniform(0.85, 1.15)
        mix = rng.uniform(0.25, 0.55)
        osc = A * env * ((1 - mix) * np.cos(w * t_eff) + mix * np.cos(w2 * t_eff))
    else:
        osc = A * env * np.cos(w * t_eff)

    raw = target_value + bias + osc

    # 7) taper (อย่าเป๊ะเกิน แต่ไม่ทำหลายทาง)
    taper_strength = rng.uniform(0.78, 0.95) if rng.random() < 0.45 else 1.0
    taper_settle = settling_time_s * rng.uniform(0.60, 0.90) if rng.random() < 0.25 else settling_time_s
    smooth = apply_cosine_taper_settling(raw, time_vector, taper_settle, target_value, strength=taper_strength)

    # 8) noise หลังนิ่ง
    final, sd = add_post_settle_noise(
        smooth, time_vector, settling_time_s, target_value, rng,
        probability=0.9,
        post_sd_scale=(0.0008, 0.0013),
        smoothness_range=(10, 20),
        add_wobble_prob=0.18,
        wobble_scale=(0.00010, 0.00022),
    )

    return final, sd, "type1_Damped_Osc", float(settling_time_s * 1000.0), 0


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
    
    noise_level_pct = random_gen.uniform(0.00005, 0.0002)
    sd_noise = max(target_value * noise_level_pct, 1e-6)
    signal_array += random_gen.normal(0.0, sd_noise, size=len(time_vector))
    
    return signal_array, sd_noise, "type2_Triangle_Wave", 0.0, 1


def generate_low_swing_sine_wave(time_vector, target_value, settling_time_s, limit_low, limit_high, random_gen):
    """
    Type 3: Low Swing Sine Wave (Sea Wave).
    Generates a continuous sine wave with low amplitude and very low noise.
    """
    signal_array = np.full_like(time_vector, target_value)
    freq_hz = random_gen.uniform(200, 500) 
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
    signal_array += random_gen.normal(0.0, sd_noise, size=len(time_vector))
    
    return signal_array, sd_noise, "type3_sine_wave", 0.0, 1


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
    
    return final_signal, max_noise, "type4_overdamped_decay", float(settling_time_s*1000.0), 0

def generate_pulse_train(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """
    Type 5: Steady baseline + POSITIVE square pulses (no settling).
    - Baseline stays near target_value for the whole record.
    - Pulses occur periodically; amplitude can be high or low.
    - Per-pulse amplitude varies, with chance to repeat previous amplitude.
    - Guarantees at least 1 pulse in the record.
    """
    y = np.full_like(time_vector, target_value, dtype=float)
    band = float(limit_high - limit_low)

    # --- choose "style" per waveform: high or low amplitude ---
    if rng.random() < 0.35:
        amp_scale = rng.uniform(1.2, 2.5)   # high pulses
    else:
        amp_scale = rng.uniform(0.25, 1.1)  # low/medium pulses

    base_amp = band * amp_scale  # ✅ base amplitude ต่อ waveform (positive)

    # --- period & duty ---
    t_end = float(time_vector[-1])
    period = rng.uniform(t_end / 5.0, t_end / 2.0)   # คาบยาวขึ้น
    duty = rng.uniform(0.08, 0.20)                  # pulse แคบ
    jitter_frac = rng.uniform(0.00, 0.08)

    # start offset
    current_time = rng.uniform(0.0, period * 0.6)

    # amplitude repeat probability
    p_same = 0.35
    prev_amp = None
    has_pulse = False

    while current_time < t_end:
        this_period = period * rng.uniform(1.0 - jitter_frac, 1.0 + jitter_frac)
        width = max(this_period * duty * rng.uniform(0.85, 1.15), 1e-6)

        t_start = current_time
        t_stop = min(current_time + width, t_end)

        mask = (time_vector >= t_start) & (time_vector < t_stop)
        if np.any(mask):
            has_pulse = True

            # decide this pulse amplitude
            if (prev_amp is not None) and (rng.random() < p_same):
                this_amp = prev_amp
            else:
                this_amp = base_amp * rng.uniform(0.4, 1.6)  # สุ่มใหม่ต่อ pulse
                prev_amp = this_amp

            # ✅ IMPORTANT: actually add pulse to signal
            y[mask] += this_amp

        current_time += this_period

    # Guarantee at least one pulse
    if not has_pulse:
        t_mid = 0.5 * t_end
        width = 0.05 * t_end
        mask = (time_vector >= t_mid) & (time_vector < t_mid + width)
        y[mask] += base_amp

    # add only "family floor noise" (no post-settle wiggle)
    y, sd = add_post_settle_noise(
        y, time_vector,
        settling_time_s=0.0,
        target_value=target_value,
        rng=rng,
        probability=0.0,
        add_wobble_prob=0.0,
    )

    return y, sd, "type5_Square_Pulse_Train", 0.0, 1


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
    generate_overdamped_decay,
    generate_pulse_train,   # NEW
    ]


    print(f"Generating TRAINING dataset ({args.n_waves} waves)...")
    
    # --- Balanced Distribution Logic ---
    ratios = [
    (generate_step_response,              0.28),  # Type 0
    (generate_high_start_oscillation,     0.22),  # Type 1
    (generate_overdamped_decay,           0.18),  # Type 4
    (generate_pulse_train,                0.14),  # Type 5
    (generate_continuous_triangular_pulses,0.09), # Type 2
    (generate_low_swing_sine_wave,        0.09),  # Type 3  
    ]   


    plan = []
    current = 0
    for func, r in ratios:
        cnt = int(args.n_waves * r)
        plan.append((func, cnt))
        current += cnt

    remainder = args.n_waves - current
    if remainder > 0:
        f0, c0 = plan[0]
        plan[0] = (f0, c0 + remainder)

    gen_sequence = []
    for f, cnt in plan:
        gen_sequence.extend([f] * cnt)

    master_rng.shuffle(gen_sequence)
    shuffled_gens = gen_sequence

    for wave_id in range(1, args.n_waves + 1):
        final_value = master_rng.uniform(0.5, 3.5)
        band_pct = master_rng.uniform(0.05, 0.15)
        band = final_value * band_pct
        low, high = final_value - band/2, final_value + band/2
        settle_time_ms = master_rng.uniform(2.0, 8.0)
        settle_s = settle_time_ms / 1000.0
 
        gen_func = master_rng.choice(signal_generators)
        wave_rng = np.random.default_rng(100000 + wave_id)

        y, used_sd, type_name, true_settle_ms, true_is_zero = gen_func(
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
            })
 
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Successfully saved TRAINING data to: {out_path}")
    print(f"Total rows generated: {len(df)}")
    print("Distribution:", df.groupby('wave_id')['signal_type'].first().value_counts())
 
if __name__ == "__main__":
    main()


 
 
 
 
 
