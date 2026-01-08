# scripts/generate_train_sample.py
"""
Generate synthetic training waveform data.

This script creates multiple waveform families (Type 0–5) and exports them
to a long-format CSV: one row per (wave_id, sample).

Columns:
- wave_id: waveform index
- type: waveform type label
- sample: sample index within the waveform
- time_ms: time in milliseconds
- value: signal value
- sd: estimated noise scale used in generation
- low_limit / high_limit: band limits around the target value
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1) Utility Functions
# =============================================================================

def apply_cosine_taper_settling(
    signal_array: np.ndarray,
    time_vector: np.ndarray,
    settling_time_s: float,
    target_value: float,
    strength: float = 1.0
) -> np.ndarray:
    """
    Smoothly forces the signal to approach the target value by the settling time
    using a cosine taper mask.

    Parameters
    ----------
    signal_array : np.ndarray
        Input signal array.
    time_vector : np.ndarray
        Time array (seconds), same length as signal_array.
    settling_time_s : float
        Settling time in seconds.
    target_value : float
        Final target value after settling.
    strength : float, default=1.0
        1.0 -> full taper (strong enforcement)
        0.0 -> no taper (original signal)

    Returns
    -------
    np.ndarray
        Tapered signal.
    """
    fade_mask = np.zeros_like(time_vector, dtype=float)
    active = time_vector < settling_time_s

    if np.any(active):
        # Cosine taper from 1 -> 0 across [0, settling_time_s)
        t_ratio = time_vector[active] / max(settling_time_s, 1e-12)
        fade_mask[active] = 0.5 * (1.0 + np.cos(np.pi * t_ratio))

    deviation = signal_array - target_value
    tapered = target_value + deviation * fade_mask

    # Blend with original to avoid an overly "perfect" settling behavior.
    return strength * tapered + (1.0 - strength) * signal_array


def add_post_settle_noise(
    signal_array: np.ndarray,
    time_vector: np.ndarray,
    settling_time_s: float,
    target_value: float,
    rng: np.random.Generator,
    probability: float = 0.9,
    post_sd_scale=(0.0008, 0.0014),
    smoothness_range=(20, 40),
    add_wobble_prob: float = 0.18,
    wobble_scale=(0.00010, 0.00022),
    wobble_win_range=(35, 80),
):
    """
    Adds noise in a measurement-like way:
    1) Floor noise across the entire record.
    2) Post-settle correlated wiggle (moving average once).
    3) Optional larger "wobble" (rare, longer window moving average).

    Returns
    -------
    (signal_array, final_sd)
        final_sd is the maximum noise scale used.
    """
    # 1) Floor noise (always present)
    base_floor_sd = max(target_value * rng.uniform(0.0001, 0.00025), 1e-6)
    signal_array = signal_array + rng.normal(0.0, base_floor_sd, size=len(time_vector))
    final_sd = base_floor_sd

    # Identify post-settle segment
    settle_idx = int(np.searchsorted(time_vector, settling_time_s))
    remaining_len = len(time_vector) - settle_idx
    if remaining_len <= 5:
        return signal_array, final_sd

    # 2) Post-settle correlated wiggle
    if rng.random() < probability:
        post_sd = max(target_value * rng.uniform(*post_sd_scale), 1e-6)
        final_sd = max(final_sd, post_sd)

        raw = rng.normal(0.0, post_sd, size=remaining_len)
        smoothness = int(rng.integers(smoothness_range[0], smoothness_range[1] + 1))
        smoothness = min(smoothness, remaining_len - 1)

        if smoothness > 2:
            k = np.ones(smoothness) / smoothness
            wig = np.convolve(raw, k, mode="same") * math.sqrt(smoothness)
            signal_array[settle_idx:] += wig
        else:
            signal_array[settle_idx:] += raw

    # 3) Optional wobble (rare, lower probability)
    if rng.random() < add_wobble_prob:
        wob_sd = max(target_value * rng.uniform(*wobble_scale), 1e-6)
        final_sd = max(final_sd, wob_sd)

        wob = rng.normal(0.0, wob_sd, size=remaining_len)
        win = int(rng.integers(wobble_win_range[0], wobble_win_range[1] + 1))
        win = min(win, remaining_len - 1)

        if win > 3:
            k2 = np.ones(win) / win
            wob = np.convolve(wob, k2, mode="same") * math.sqrt(win)

        signal_array[settle_idx:] += wob

    return signal_array, final_sd


def add_time_delay(t: np.ndarray, rng: np.random.Generator, max_delay_s: float = 0.0006):
    """
    Applies a random time delay (time shift).
    Returns the delayed time array and the delay value.
    """
    d = float(rng.uniform(0.0, max_delay_s))
    return np.clip(t - d, 0.0, None), d


# =============================================================================
# 2) Waveform Generators (Type 0–5)
# =============================================================================

def generate_step_response(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """
    Type 0: Step response with damped ringing + measurement-like post-settle noise.
    """
    freq_hz = float(rng.uniform(100, 1200))
    w1 = 2.0 * np.pi * freq_hz
    band_half = (limit_high - limit_low) / 2.0

    overshoot_scale = float(rng.uniform(1.5, 8.0 if target_value < 1.0 else 15.0))
    direction = float(rng.choice([1.0, -1.0]))
    amp0 = band_half * overshoot_scale * direction

    tau = max(settling_time_s / float(rng.uniform(2.5, 6.0)), 1e-6)
    time_constant_rise = max(settling_time_s / float(rng.uniform(3.0, 10.0)), 1e-6)

    # Random delay on time axis
    t_eff, _ = add_time_delay(time_vector, rng, max_delay_s=0.0006)

    # Rise (1-pole or 2-pole)
    if rng.random() < 0.5:
        base = target_value * (1.0 - np.exp(-t_eff / time_constant_rise))
    else:
        tau2 = max(time_constant_rise * float(rng.uniform(1.5, 4.0)), 1e-6)
        base = target_value * (1.0 - 0.6*np.exp(-t_eff/time_constant_rise) - 0.4*np.exp(-t_eff/tau2))

    # Ringing (single-tone or mixed-tone)
    if rng.random() < 0.6:
        ring = amp0 * np.exp(-t_eff / tau) * np.sin(w1 * t_eff)
    else:
        freq2 = freq_hz * float(rng.uniform(0.75, 1.25))
        w2 = 2.0 * np.pi * freq2
        mix = float(rng.uniform(0.2, 0.6))
        ring = amp0 * np.exp(-t_eff / tau) * ((1.0 - mix) * np.sin(w1 * t_eff) + mix * np.sin(w2 * t_eff))

    y = base + ring

    # Optional early kick transient
    if rng.random() < 0.25:
        kick_amp = (limit_high - limit_low) * float(rng.uniform(0.15, 0.8)) * float(rng.choice([1.0, -1.0]))
        kick_tau = max(settling_time_s / float(rng.uniform(6.0, 14.0)), 1e-6)
        y += kick_amp * np.exp(-t_eff / kick_tau)

    # Imperfect taper to avoid overly ideal settling
    taper_strength = float(rng.uniform(0.78, 0.95))
    y = apply_cosine_taper_settling(y, time_vector, settling_time_s, target_value, strength=taper_strength)

    # Post-settle noise texture
    y, sd = add_post_settle_noise(
        y, time_vector, settling_time_s, target_value, rng,
        probability=0.95,
        smoothness_range=(10, 28),
        post_sd_scale=(0.0009, 0.0016),
        add_wobble_prob=0.35,
        wobble_scale=(0.00012, 0.00030),
    )

    return y, sd, "type0_Step_Response", float(settling_time_s * 1000.0), 0


def generate_high_start_oscillation(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """
    Type 1: High-start underdamped oscillation with envelope shaping and post-settle noise.
    """
    band = float(limit_high - limit_low)

    # Number of cycles within settling window
    num_cycles = int(rng.integers(2, 4)) if rng.random() < 0.70 else int(rng.integers(4, 7))

    # Random delay on time axis
    t_eff, _ = add_time_delay(time_vector, rng, max_delay_s=0.0004)

    freq = num_cycles / max(settling_time_s, 1e-6)
    w = 2.0 * np.pi * freq
    if rng.random() < 0.35:
        w *= float(rng.uniform(0.88, 1.12))

    # Oscillation amplitude and transient bias
    A = band * float(rng.uniform(2.0, 4.0))
    bias_amp = band * float(rng.uniform(-0.55, 0.55))
    bias_tau = max(settling_time_s / float(rng.uniform(0.9, 2.2)), 1e-6)
    bias = bias_amp * np.exp(-t_eff / bias_tau)

    # Envelope shaping
    tau_slow = max(settling_time_s * float(rng.uniform(0.65, 1.35)), 1e-6)
    tau_fast = max(tau_slow / float(rng.uniform(2.2, 4.8)), 1e-6)
    t_switch = settling_time_s * float(rng.uniform(0.22, 0.42))

    if rng.random() < 0.22:
        tau_slow *= float(rng.uniform(1.5, 2.8))
        tau_fast *= float(rng.uniform(2.0, 4.5))
        t_switch = settling_time_s * float(rng.uniform(0.30, 0.55))

    env = np.where(
        t_eff < t_switch,
        np.exp(-t_eff / tau_slow),
        np.exp(-t_switch / tau_slow) * np.exp(-(t_eff - t_switch) / tau_fast),
    )

    # Mild coupling: deeper valley -> slightly boosted mid-window amplitude
    T = (2.0*np.pi) / max(w, 1e-12)
    early_mask = (t_eff >= 0.0) & (t_eff <= 1.8 * T)
    if np.any(early_mask):
        y_early = target_value + bias[early_mask] + (A * env[early_mask] * np.cos(w * t_eff[early_mask]))
        valley_depth = float(target_value - np.min(y_early))
        depth_ratio = np.clip(valley_depth / max(band, 1e-12), 0.0, 1.8)

        boost_amp = 1.0 + float(rng.uniform(0.55, 1.05)) * depth_ratio
        mid_mask = (t_eff >= 1.0 * T) & (t_eff <= 2.2 * T)
        if np.any(mid_mask):
            x = (t_eff[mid_mask] - 1.0 * T) / max((2.2 * T - 1.0 * T), 1e-12)
            bump = 0.5 - 0.5*np.cos(2.0*np.pi*x)  # 0..1..0
            env[mid_mask] *= (1.0 + (boost_amp - 1.0) * bump)

    # Optional 2-tone mix
    if rng.random() < 0.18:
        w2 = w * float(rng.uniform(0.85, 1.15))
        mix = float(rng.uniform(0.25, 0.55))
        osc = A * env * ((1.0 - mix) * np.cos(w * t_eff) + mix * np.cos(w2 * t_eff))
    else:
        osc = A * env * np.cos(w * t_eff)

    y = target_value + bias + osc

    taper_strength = float(rng.uniform(0.78, 0.95)) if rng.random() < 0.45 else 1.0
    taper_settle = settling_time_s * float(rng.uniform(0.60, 0.90)) if rng.random() < 0.25 else settling_time_s
    y = apply_cosine_taper_settling(y, time_vector, taper_settle, target_value, strength=taper_strength)

    y, sd = add_post_settle_noise(
        y, time_vector, settling_time_s, target_value, rng,
        probability=0.9,
        post_sd_scale=(0.0008, 0.0013),
        smoothness_range=(10, 20),
        add_wobble_prob=0.18,
        wobble_scale=(0.00010, 0.00022),
    )

    return y, sd, "type1_Damped_Osc", float(settling_time_s * 1000.0), 0


def generate_continuous_triangular_pulses(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """
    Type 2: Triangular pulse train (no explicit settling in current implementation).
    Note: This remains as-is from your current behavior: continuous pulses + small noise.
    """
    y = np.full_like(time_vector, target_value, dtype=float)

    is_height_const = bool(rng.choice([True, False]))
    is_period_const = bool(rng.choice([True, False]))

    avg_height = (limit_high - limit_low) * float(rng.uniform(0.5, 1.5))
    avg_period = float(rng.uniform(settling_time_s / 3.0, settling_time_s / 1.5))
    pulse_width = avg_period * float(rng.uniform(0.15, 0.30))
    current_time = float(rng.uniform(0.0, avg_period * 0.3))

    while current_time < float(time_vector[-1]):
        height = avg_height if is_height_const else avg_height * float(rng.uniform(0.5, 1.5))

        t_start = current_time
        t_peak = current_time + pulse_width / 2.0
        t_end = current_time + pulse_width

        rise = (time_vector >= t_start) & (time_vector < t_peak)
        if np.any(rise):
            y[rise] += (height / (pulse_width / 2.0)) * (time_vector[rise] - t_start)

        fall = (time_vector >= t_peak) & (time_vector < t_end)
        if np.any(fall):
            y[fall] += height - (height / (pulse_width / 2.0)) * (time_vector[fall] - t_peak)

        period = avg_period if is_period_const else avg_period * float(rng.uniform(0.7, 1.3))
        current_time += period

    noise_level_pct = float(rng.uniform(0.00005, 0.0002))
    sd = max(target_value * noise_level_pct, 1e-6)
    early = time_vector < (0.2 * time_vector[-1])
    y[early] = np.maximum(y[early], target_value)


    return y, sd, "type2_Triangle_Wave", 0.0, 1


def generate_low_swing_sine_wave(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """
    Type 3: Low-amplitude sine oscillation around the target value.
    """
    y = np.full_like(time_vector, target_value, dtype=float)

    freq_hz = float(rng.uniform(200, 500))
    w = 2.0 * np.pi * freq_hz
    amplitude = (limit_high - limit_low) * float(rng.uniform(0.1, 0.25))

    phase = float(rng.choice([0.0, np.pi]))
    y += amplitude * np.sin(w * time_vector + phase)

    noise_level_pct = float(rng.uniform(0.0, 0.00002))
    sd = max(target_value * noise_level_pct, 1e-6)
    y += rng.normal(0.0, sd, size=len(time_vector))

    return y, sd, "type3_Sine_Wave", 0.0, 1


def generate_overdamped_decay(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """
    Type 4: Overdamped exponential decay toward the target value (no ringing).
    """
    start_amp = (limit_high - limit_low) * float(rng.uniform(1.5, 3.0))
    tau = settling_time_s / float(rng.uniform(3.0, 5.0))

    y = target_value + start_amp * np.exp(-time_vector / max(tau, 1e-12))
    y = apply_cosine_taper_settling(y, time_vector, settling_time_s, target_value, strength=1.0)

    y, sd = add_post_settle_noise(y, time_vector, settling_time_s, target_value, rng)
    return y, sd, "type4_overdamped_no_overshoot", float(settling_time_s * 1000.0), 0

def generate_overdamped_decay1(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """
    Type 4 (modified): Overdamped bi-exponential with a single undershoot ("overshoot before settling")
    - No sinusoid, so still not 'ringing' in the oscillatory sense.
    - Shape: start high -> dip below target once -> recover to target.
    """
    band = float(limit_high - limit_low)

    # --- time constants (fast then slow) ---
    # fast decays quickly, slow decays more slowly (controls recovery tail)
    tau_fast = max(settling_time_s / float(rng.uniform(6.0, 12.0)), 1e-6)
    tau_slow = max(settling_time_s / float(rng.uniform(1.8, 3.5)), 1e-6)

    # --- amplitudes ---
    # A_pos sets initial height above target
    A_pos = band * float(rng.uniform(1.5, 3.0))

    # A_neg controls how deep the undershoot is
    # keep A_neg a bit smaller than A_pos so y(0) still above target
    A_neg = A_pos * float(rng.uniform(0.35, 0.75))

    # bi-exponential: +fast -slow  -> creates one dip below target then returns
    y = target_value + A_pos * np.exp(-time_vector / tau_fast) - A_neg * np.exp(-time_vector / tau_slow)

    # Optional: make the dip occur "ก่อนถึงเส้นนิ่ง" มากขึ้นด้วย time delay เล็ก ๆ
    if rng.random() < 0.35:
        t_eff, _ = add_time_delay(time_vector, rng, max_delay_s=0.0004)
        y = target_value + A_pos * np.exp(-t_eff / tau_fast) - A_neg * np.exp(-t_eff / tau_slow)

    # taper ให้เข้าหา target ช่วงใกล้ settling_time (กันปลายแกว่ง/เพี้ยนเกิน)
    taper_strength = float(rng.uniform(0.85, 1.0))
    y = apply_cosine_taper_settling(y, time_vector, settling_time_s, target_value, strength=taper_strength)

    # post-settle noise
    y, sd = add_post_settle_noise(y, time_vector, settling_time_s, target_value, rng)

    return y, sd, "type4_overdamped_decay_overshoot", float(settling_time_s * 1000.0), 0


def generate_pulse_train(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """
    Type 5: Steady baseline + POSITIVE square pulses (no settling).
    - Baseline stays at the target value for the whole record.
    - Pulses are periodic with per-pulse amplitude variation.
    - There is a controlled probability (p_same) that consecutive pulses share the same amplitude.
    - Guarantees at least one pulse is present.
    """
    y = np.full_like(time_vector, target_value, dtype=float)
    band = float(limit_high - limit_low)

    # Waveform-level amplitude family (high vs low)
    amp_scale = float(rng.uniform(1.2, 2.5)) if rng.random() < 0.35 else float(rng.uniform(0.25, 1.1))
    base_amp = band * amp_scale  # positive baseline pulse amplitude scale

    # Period and width control (long period, short width)
    t_end = float(time_vector[-1])
    period = float(rng.uniform(t_end / 5.0, t_end / 2.0))
    duty = float(rng.uniform(0.08, 0.20))
    jitter_frac = float(rng.uniform(0.00, 0.08))

    current_time = float(rng.uniform(0.0, period * 0.6))

    # Amplitude repeat behavior
    p_same = 0.35
    prev_amp = None
    has_pulse = False

    while current_time < t_end:
        this_period = period * float(rng.uniform(1.0 - jitter_frac, 1.0 + jitter_frac))
        width = max(this_period * duty * float(rng.uniform(0.85, 1.15)), 1e-6)

        t_start = current_time
        t_stop = min(current_time + width, t_end)

        mask = (time_vector >= t_start) & (time_vector < t_stop)
        if np.any(mask):
            has_pulse = True

            if (prev_amp is not None) and (rng.random() < p_same):
                this_amp = prev_amp
            else:
                this_amp = base_amp * float(rng.uniform(0.4, 1.6))
                prev_amp = this_amp

            y[mask] += this_amp

        current_time += this_period

    # Ensure at least one pulse exists
    if not has_pulse:
        t_mid = 0.5 * t_end
        width = 0.05 * t_end
        mask = (time_vector >= t_mid) & (time_vector < t_mid + width)
        y[mask] += base_amp

    # Add only floor noise to stay within the same family as other generators
    y, sd = add_post_settle_noise(
        y, time_vector,
        settling_time_s=0.0,
        target_value=target_value,
        rng=rng,
        probability=0.0,
        add_wobble_prob=0.0,
    )

    return y, sd, "type5_Square_Pulse_Wave", 0.0, 1


def generate_no_undershoot_ringing(
    time_vector, target_value, settling_time_s, limit_low, limit_high, rng
):
    """
    Ringing decay แต่ "ไม่ลงต่ำกว่า target" (no undershoot)
    ได้ทรงแบบเส้นแดง: ลงมา + แกว่ง แต่ minimum อยู่เหนือ target แล้วค่อยนิ่ง
    """
    t = time_vector.astype(float)
    t_end = float(t[-1]) if len(t) else 0.0
    band = float(limit_high - limit_low)

    # 1) baseline decay จากค่าเริ่มต้น (เริ่มสูงกว่า target)
    start_offset = band * float(rng.uniform(0.6, 1.4))   # สูงกว่า target
    tau_base = float(rng.uniform(max(t_end/8, 1e-6), max(t_end/3, 1e-6)))
    base = target_value + start_offset * np.exp(-t / tau_base)

    # 2) ringing แบบ "บวกเท่านั้น" (rectified) -> ไม่ undershoot
    # เลือกความถี่ให้มี 2-5 รอบในช่วงก่อนนิ่ง
    n_cycles = float(rng.uniform(2.0, 5.0))
    f = n_cycles / max(settling_time_s, t_end, 1e-6)  # cycles per second
    phi = float(rng.uniform(0, 2*np.pi))

    ring_amp0 = band * float(rng.uniform(0.15, 0.45))
    tau_ring = float(rng.uniform(max(t_end/10, 1e-6), max(t_end/2.5, 1e-6)))
    envelope = np.exp(-t / tau_ring)

    # rectified sine: (sin + 1)/2 อยู่ใน [0,1]
    ring = ring_amp0 * envelope * (0.5 * (np.sin(2*np.pi*f*t + phi) + 1.0))

    y = base + ring

    # 3) กันหลุดต่ำกว่า target แบบชัวร์ (ถ้าคุณต้องการ 100% no-undershoot)
    y = np.maximum(y, target_value)

    # 4) noise เล็ก ๆ
    noise_sd = max(abs(target_value) * float(rng.uniform(5e-5, 2e-4)), 1e-6)
    y += rng.normal(0.0, noise_sd, size=len(t))

    # กัน noise ทำหลุดต่ำกว่า target (ถ้าต้อง no-undershoot แบบสุด)
    y = np.maximum(y, target_value)

    return y, noise_sd, "typeX_NoUndershoot_Ringing", 0.0, 1


# =============================================================================
# 3) Main: Dataset Generation
# =============================================================================

def build_generation_plan(n_waves: int, ratios, rng: np.random.Generator):
    """
    Converts ratio specification into a shuffled list of generator functions.

    Parameters
    ----------
    n_waves : int
        Total number of waveforms.
    ratios : list[(callable, float)]
        Each entry is (generator_function, ratio).
    rng : np.random.Generator
        RNG for shuffling.

    Returns
    -------
    list[callable]
        Shuffled generator list of length n_waves.
    """
    plan = []
    allocated = 0

    for func, r in ratios:
        cnt = int(n_waves * float(r))
        plan.extend([func] * cnt)
        allocated += cnt

    # Assign any remainder to the first generator to keep total length == n_waves
    remainder = n_waves - allocated
    if remainder > 0:
        plan.extend([ratios[0][0]] * remainder)

    rng.shuffle(plan)
    return plan


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic TRAINING waveform data.")
    ap.add_argument("--out", default="data/raw/data1000samples_train.csv", help="Output CSV path")
    ap.add_argument("--n_waves", type=int, default=1000, help="Number of waveforms to generate")
    ap.add_argument("--dt_ms", type=float, default=0.01, help="Time step in milliseconds")
    ap.add_argument("--t_end_ms", type=float, default=9.9, help="End time in milliseconds")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Time axis (ms for export, seconds for generation math)
    t_ms = np.arange(0.0, args.t_end_ms + 1e-12, args.dt_ms)
    t_s = t_ms / 1000.0

    # Ratio plan: include all waveform families (Type 0–5)
    ratios = [
        (generate_step_response,               0.26),  # Type 0
        (generate_high_start_oscillation,      0.20),  # Type 1

        (generate_overdamped_decay,            0.18),  # Type 4 (เรียบ)
        (generate_overdamped_decay1,           0.10),  # Type 4.1 (เรียบ + overshoot)

        (generate_pulse_train,                 0.12),  # Type 5
        (generate_continuous_triangular_pulses,0.07),  # Type 2
        (generate_low_swing_sine_wave,         0.07),  # Type 3
    ]
   


    master_rng = np.random.default_rng(12345)
    gen_sequence = build_generation_plan(args.n_waves, ratios, master_rng)

    rows = []
    print(f"Generating TRAINING dataset ({args.n_waves} waves)...")

    for wave_id, gen_func in enumerate(gen_sequence, start=1):
        # Target value and band (limits) per waveform
        final_value = float(master_rng.uniform(0.5, 3.5))
        band_pct = float(master_rng.uniform(0.05, 0.15))
        band = final_value * band_pct
        low = final_value - band / 2.0
        high = final_value + band / 2.0

        # Settling time is used only by settling-based waveform types
        settle_time_ms = float(master_rng.uniform(2.0, 8.0))
        settle_s = settle_time_ms / 1000.0

        # Per-wave RNG ensures reproducibility per waveform id
        wave_rng = np.random.default_rng(100000 + wave_id)

        y, used_sd, type_name, true_settle_ms, true_is_zero = gen_func(
            t_s, final_value, settle_s, low, high, wave_rng
        )

        # Export rows in long format
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
                # Optional labels (uncomment if you need them in training):
                # "true_settle_ms": float(true_settle_ms),
                # "true_is_zero": int(true_is_zero),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Successfully saved TRAINING data to: {out_path}")


if __name__ == "__main__":
    main()
