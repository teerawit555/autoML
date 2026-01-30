# # scripts/generate_train_sample.py
# """
# Generate synthetic training waveform data.

# This script creates multiple waveform families (Type 0–5) and exports them
# to a long-format CSV: one row per (wave_id, sample).

# Columns:
# - wave_id: waveform index
# - type: waveform type label
# - sample: sample index within the waveform
# - time_ms: time in milliseconds
# - value: signal value
# - sd: estimated noise scale used in generation
# - low_limit / high_limit: band limits around the target value
# """

# import argparse
# import math
# from pathlib import Path

# import numpy as np
# import pandas as pd


# # =============================================================================
# # 1) Utility Functions
# # =============================================================================

# def apply_cosine_taper_settling(
#     signal_array: np.ndarray,
#     time_vector: np.ndarray,
#     settling_time_s: float,
#     target_value: float,
#     strength: float = 1.0
# ) -> np.ndarray:
#     """
#     Smoothly forces the signal to approach the target value by the settling time
#     using a cosine taper mask.

#     Parameters
#     ----------
#     signal_array : np.ndarray
#         Input signal array.
#     time_vector : np.ndarray
#         Time array (seconds), same length as signal_array.
#     settling_time_s : float
#         Settling time in seconds.
#     target_value : float
#         Final target value after settling.
#     strength : float, default=1.0
#         1.0 -> full taper (strong enforcement)
#         0.0 -> no taper (original signal)

#     Returns
#     -------
#     np.ndarray
#         Tapered signal.
#     """
#     fade_mask = np.zeros_like(time_vector, dtype=float)
#     active = time_vector < settling_time_s

#     if np.any(active):
#         # Cosine taper from 1 -> 0 across [0, settling_time_s)
#         t_ratio = time_vector[active] / max(settling_time_s, 1e-12)
#         fade_mask[active] = 0.5 * (1.0 + np.cos(np.pi * t_ratio))

#     deviation = signal_array - target_value
#     tapered = target_value + deviation * fade_mask

#     # Blend with original to avoid an overly "perfect" settling behavior.
#     return strength * tapered + (1.0 - strength) * signal_array


# def add_post_settle_noise(
#     signal_array: np.ndarray,
#     time_vector: np.ndarray,
#     settling_time_s: float,
#     target_value: float,
#     rng: np.random.Generator,
#     probability: float = 0.9,
#     post_sd_scale=(0.0008, 0.0014),
#     smoothness_range=(20, 40),
#     add_wobble_prob: float = 0.18,
#     wobble_scale=(0.00010, 0.00022),
#     wobble_win_range=(35, 80),
# ):
#     """
#     Adds noise in a measurement-like way:
#     1) Floor noise across the entire record.
#     2) Post-settle correlated wiggle (moving average once).
#     3) Optional larger "wobble" (rare, longer window moving average).

#     Returns
#     -------
#     (signal_array, final_sd)
#         final_sd is the maximum noise scale used.
#     """
#     # 1) Floor noise (always present)
#     base_floor_sd = max(target_value * rng.uniform(0.0001, 0.00025), 1e-6)
#     signal_array = signal_array + rng.normal(0.0, base_floor_sd, size=len(time_vector))
#     final_sd = base_floor_sd

#     # Identify post-settle segment
#     settle_idx = int(np.searchsorted(time_vector, settling_time_s))
#     remaining_len = len(time_vector) - settle_idx
#     if remaining_len <= 5:
#         return signal_array, final_sd

#     # 2) Post-settle correlated wiggle
#     if rng.random() < probability:
#         post_sd = max(target_value * rng.uniform(*post_sd_scale), 1e-6)
#         final_sd = max(final_sd, post_sd)

#         raw = rng.normal(0.0, post_sd, size=remaining_len)
#         smoothness = int(rng.integers(smoothness_range[0], smoothness_range[1] + 1))
#         smoothness = min(smoothness, remaining_len - 1)

#         if smoothness > 2:
#             k = np.ones(smoothness) / smoothness
#             wig = np.convolve(raw, k, mode="same") * math.sqrt(smoothness)
#             signal_array[settle_idx:] += wig
#         else:
#             signal_array[settle_idx:] += raw

#     # 3) Optional wobble (rare, lower probability)
#     if rng.random() < add_wobble_prob:
#         wob_sd = max(target_value * rng.uniform(*wobble_scale), 1e-6)
#         final_sd = max(final_sd, wob_sd)

#         wob = rng.normal(0.0, wob_sd, size=remaining_len)
#         win = int(rng.integers(wobble_win_range[0], wobble_win_range[1] + 1))
#         win = min(win, remaining_len - 1)

#         if win > 3:
#             k2 = np.ones(win) / win
#             wob = np.convolve(wob, k2, mode="same") * math.sqrt(win)

#         signal_array[settle_idx:] += wob

#     return signal_array, final_sd


# def add_time_delay(t: np.ndarray, rng: np.random.Generator, max_delay_s: float = 0.0006):
#     """
#     Applies a random time delay (time shift).
#     Returns the delayed time array and the delay value.
#     """
#     d = float(rng.uniform(0.0, max_delay_s))
#     t_eff = np.zeros_like(t)
#     mask = t >= d
#     t_eff[mask] = t[mask] - d
#     return np.clip(t - d, 0.0, None), d

# def _pick_gain_to_reach(max_abs_value: float, limit_low: float, limit_high: float, rng) -> float:
#     """
#     เลือก gain >= 1 เพื่อขยายสเกลให้ |value| มีโอกาสไปใกล้ max_abs_value
#     โดยดูจาก limit ที่ส่งมาเป็นฐาน (กันคูณจนหลุดโลก)
#     """
#     hi = float(limit_high)
#     lo = float(limit_low)

#     base = max(abs(hi), abs(lo), 1e-6)  # ขนาดสเกลเดิม
#     gain_max = max_abs_value / base

#     # ถ้าเดิมใหญ่พอแล้ว -> ไม่ต้องขยาย
#     if gain_max <= 1.05:
#         return 1.0

#     # สุ่ม gain ให้กระจาย (บางอันไม่ขยายมาก บางอันขยายเยอะ)
#     # ปรับช่วงนี้ได้ตามใจ
#     g = float(rng.uniform(1.0, gain_max))
#     return max(1.0, g)


# def _scale_params(target_value, settling_time_s, limit_low, limit_high, rng, max_abs_value=50.0):
#     g = _pick_gain_to_reach(max_abs_value, limit_low, limit_high, rng)
#     return float(target_value) * g, float(limit_low) * g, float(limit_high) * g, g

# def sample_target_0_50(rng, max_v=50.0, n_bins=10):
#     # แบ่งช่วง 0..50 เป็น bin แล้วสุ่ม bin ก่อน -> กระจายครบ
#     edges = np.linspace(0.0, max_v, n_bins + 1)
#     b = int(rng.integers(0, n_bins))
#     lo, hi = float(edges[b]), float(edges[b + 1])
#     return float(rng.uniform(lo, hi))



# # =============================================================================
# # 2) Waveform Generators (Type 0–5)
# # =============================================================================

# def generate_step_response(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
# #     target_value, limit_low, limit_high, g = _scale_params(
# #         target_value, settling_time_s, limit_low, limit_high, rng, max_abs_value=50.0
# # )

    
#     """
#     Type 0: Step response with damped ringing + measurement-like post-settle noise.
#     """
#     freq_hz = float(rng.uniform(100, 1200))
#     w1 = 2.0 * np.pi * freq_hz
#     band_half = (limit_high - limit_low) / 2.0

#     overshoot_scale = float(rng.uniform(1.5, 8.0 if target_value < 1.0 else 15.0))
#     direction = float(rng.choice([1.0, -1.0]))
#     amp0 = band_half * overshoot_scale * direction

#     tau = max(settling_time_s / float(rng.uniform(2.5, 6.0)), 1e-6)
#     time_constant_rise = max(settling_time_s / float(rng.uniform(3.0, 10.0)), 1e-6)

#     # Random delay on time axis
#     t_eff, _ = add_time_delay(time_vector, rng, max_delay_s=0.0006)

#     # Rise (1-pole or 2-pole)
#     if rng.random() < 0.5:
#         base = target_value * (1.0 - np.exp(-t_eff / time_constant_rise))
#     else:
#         tau2 = max(time_constant_rise * float(rng.uniform(1.5, 4.0)), 1e-6)
#         base = target_value * (1.0 - 0.6*np.exp(-t_eff/time_constant_rise) - 0.4*np.exp(-t_eff/tau2))

#     # Ringing (single-tone or mixed-tone)
#     if rng.random() < 0.6:
#         ring = amp0 * np.exp(-t_eff / tau) * np.sin(w1 * t_eff)
#     else:
#         freq2 = freq_hz * float(rng.uniform(0.75, 1.25))
#         w2 = 2.0 * np.pi * freq2
#         mix = float(rng.uniform(0.2, 0.6))
#         ring = amp0 * np.exp(-t_eff / tau) * ((1.0 - mix) * np.sin(w1 * t_eff) + mix * np.sin(w2 * t_eff))

#     y = base + ring

#     # Optional early kick transient
#     if rng.random() < 0.25:
#         kick_amp = (limit_high - limit_low) * float(rng.uniform(0.15, 0.8)) * float(rng.choice([1.0, -1.0]))
#         kick_tau = max(settling_time_s / float(rng.uniform(6.0, 14.0)), 1e-6)
#         y += kick_amp * np.exp(-t_eff / kick_tau)

#     # Imperfect taper to avoid overly ideal settling
#     taper_strength = float(rng.uniform(0.78, 0.95))
#     y = apply_cosine_taper_settling(y, time_vector, settling_time_s, target_value, strength=taper_strength)

#     # Post-settle noise texture
#     y, sd = add_post_settle_noise(
#         y, time_vector, settling_time_s, target_value, rng,
#         probability=0.95,
#         smoothness_range=(10, 28),
#         post_sd_scale=(0.0009, 0.0016),
#         add_wobble_prob=0.35,
#         wobble_scale=(0.00012, 0.00030),
#     )

#     return y, sd, "type0_Step_Response", float(settling_time_s * 1000.0), 0

# def generate_high_start_oscillation(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
#     # target_value, limit_low, limit_high, g = _scale_params(
#     # target_value, settling_time_s, limit_low, limit_high, rng, max_abs_value=50.0
#     # )
#     """
#     Type 1 (SAFE): High-start underdamped oscillation
#     - บังคับให้ "settle จริง" ภายใน settling_time_s (เช่น 10ms)
#     - กันเคสที่ env ยังเหลือเยอะตอนท้ายจน label เพี้ยน
#     """
#     band = float(limit_high - limit_low)
#     t = time_vector.astype(float)

#     # --------------------------
#     # 1) cycles: จำกัดไม่ให้ยาวเกินจนปลายยังแกว่ง
#     # --------------------------
#     # เดิมมี 4-7 cycles เยอะไปสำหรับ 10ms -> มักไม่ settle
#     num_cycles = int(rng.integers(2, 5))  # 2..4 cycles

#     # Random delay on time axis
#     t_eff, _ = add_time_delay(t, rng, max_delay_s=0.0004)

#     # freq ให้ครบ num_cycles ภายใน settling_time_s
#     freq = num_cycles / max(settling_time_s, 1e-6)
#     w = 2.0 * np.pi * freq
#     if rng.random() < 0.25:
#         w *= float(rng.uniform(0.92, 1.08))

#     # --------------------------
#     # 2) amplitude + bias
#     # --------------------------
#     A = band * float(rng.uniform(1.6, 3.0))  # ลดนิดเพื่อไม่ให้ tail ยังเห็นชัด
#     bias_amp = band * float(rng.uniform(-0.45, 0.45))
#     bias_tau = max(settling_time_s / float(rng.uniform(1.4, 3.5)), 1e-6)
#     bias = bias_amp * np.exp(-t_eff / bias_tau)

#     # --------------------------
#     # 3) Envelope: "บังคับให้ปลายเหลือ < eps"
#     # --------------------------
#     # อยากให้ตอน t = settling_time_s: env_end <= eps
#     eps = float(rng.uniform(0.010, 0.025))  # 1.0%..2.5% ของ amplitude

#     # single-exponential ที่คุมปลายได้ตรง
#     # exp(-T/tau) = eps  => tau = -T / ln(eps)
#     tau_main = max(-settling_time_s / np.log(max(eps, 1e-9)), 1e-6)

#     # ทำให้ช่วงต้น decay ช้ากว่าเล็กน้อย แล้วสับไปเร็วตอนกลางๆ (ดูธรรมชาติ)
#     tau_slow = tau_main * float(rng.uniform(1.10, 1.35))
#     tau_fast = tau_main * float(rng.uniform(0.55, 0.85))
#     t_switch = settling_time_s * float(rng.uniform(0.35, 0.55))

#     env = np.where(
#         t_eff < t_switch,
#         np.exp(-t_eff / max(tau_slow, 1e-12)),
#         np.exp(-t_switch / max(tau_slow, 1e-12)) * np.exp(-(t_eff - t_switch) / max(tau_fast, 1e-12)),
#     )

#     # --------------------------
#     # 4) Oscillation (optionally 2-tone)
#     # --------------------------
#     if rng.random() < 0.15:
#         w2 = w * float(rng.uniform(0.90, 1.10))
#         mix = float(rng.uniform(0.25, 0.50))
#         osc = A * env * ((1.0 - mix) * np.cos(w * t_eff) + mix * np.cos(w2 * t_eff))
#     else:
#         osc = A * env * np.cos(w * t_eff)

#     y = target_value + bias + osc

#     # --------------------------
#     # 5) Hard settle enforcement (สำคัญ): ให้ปลาย settle จริง
#     # --------------------------
#     # สำหรับ Type นี้ แนะนำให้ taper แรงเสมอ
#     y = apply_cosine_taper_settling(
#         y, t, settling_time_s, target_value,
#         strength=float(rng.uniform(0.92, 1.00))
#     )

#     # --------------------------
#     # 6) Noise (อย่าให้ noise ทำเหมือนยังแกว่ง)
#     # --------------------------
#     y, sd = add_post_settle_noise(
#         y, t, settling_time_s, target_value, rng,
#         probability=0.90,
#         post_sd_scale=(0.0006, 0.0011),   # ลดนิด กัน tail ดูเหมือนยัง oscillate
#         smoothness_range=(10, 18),
#         add_wobble_prob=0.12,
#         wobble_scale=(0.00008, 0.00018),
#     )

#     return y, sd, "type1_Damped_Osc", float(settling_time_s * 1000.0), 0


# # def generate_high_start_oscillation(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
# #     """
# #     Type 1: High-start underdamped oscillation with envelope shaping and post-settle noise.
# #     """
# #     band = float(limit_high - limit_low)

# #     # Number of cycles within settling window
# #     num_cycles = int(rng.integers(2, 4)) if rng.random() < 0.70 else int(rng.integers(4, 7))

# #     # Random delay on time axis
# #     t_eff, _ = add_time_delay(time_vector, rng, max_delay_s=0.0004)

# #     freq = num_cycles / max(settling_time_s, 1e-6)
# #     w = 2.0 * np.pi * freq
# #     if rng.random() < 0.35:
# #         w *= float(rng.uniform(0.88, 1.12))

# #     # Oscillation amplitude and transient bias
# #     A = band * float(rng.uniform(2.0, 4.0))
# #     bias_amp = band * float(rng.uniform(-0.55, 0.55))
# #     bias_tau = max(settling_time_s / float(rng.uniform(0.9, 2.2)), 1e-6)
# #     bias = bias_amp * np.exp(-t_eff / bias_tau)

# #     # Envelope shaping
# #     tau_slow = max(settling_time_s * float(rng.uniform(0.65, 1.35)), 1e-6)
# #     tau_fast = max(tau_slow / float(rng.uniform(2.2, 4.8)), 1e-6)
# #     t_switch = settling_time_s * float(rng.uniform(0.22, 0.42))

# #     if rng.random() < 0.22:
# #         tau_slow *= float(rng.uniform(1.5, 2.8))
# #         tau_fast *= float(rng.uniform(2.0, 4.5))
# #         t_switch = settling_time_s * float(rng.uniform(0.30, 0.55))

# #     env = np.where(
# #         t_eff < t_switch,
# #         np.exp(-t_eff / tau_slow),
# #         np.exp(-t_switch / tau_slow) * np.exp(-(t_eff - t_switch) / tau_fast),
# #     )

# #     # Mild coupling: deeper valley -> slightly boosted mid-window amplitude
# #     T = (2.0*np.pi) / max(w, 1e-12)
# #     early_mask = (t_eff >= 0.0) & (t_eff <= 1.8 * T)
# #     if np.any(early_mask):
# #         y_early = target_value + bias[early_mask] + (A * env[early_mask] * np.cos(w * t_eff[early_mask]))
# #         valley_depth = float(target_value - np.min(y_early))
# #         depth_ratio = np.clip(valley_depth / max(band, 1e-12), 0.0, 1.8)

# #         boost_amp = 1.0 + float(rng.uniform(0.55, 1.05)) * depth_ratio
# #         mid_mask = (t_eff >= 1.0 * T) & (t_eff <= 2.2 * T)
# #         if np.any(mid_mask):
# #             x = (t_eff[mid_mask] - 1.0 * T) / max((2.2 * T - 1.0 * T), 1e-12)
# #             bump = 0.5 - 0.5*np.cos(2.0*np.pi*x)  # 0..1..0
# #             env[mid_mask] *= (1.0 + (boost_amp - 1.0) * bump)

# #     # Optional 2-tone mix
# #     if rng.random() < 0.18:
# #         w2 = w * float(rng.uniform(0.85, 1.15))
# #         mix = float(rng.uniform(0.25, 0.55))
# #         osc = A * env * ((1.0 - mix) * np.cos(w * t_eff) + mix * np.cos(w2 * t_eff))
# #     else:
# #         osc = A * env * np.cos(w * t_eff)

# #     y = target_value + bias + osc

# #     taper_strength = float(rng.uniform(0.78, 0.95)) if rng.random() < 0.45 else 1.0
# #     taper_settle = settling_time_s * float(rng.uniform(0.60, 0.90)) if rng.random() < 0.25 else settling_time_s
# #     y = apply_cosine_taper_settling(y, time_vector, taper_settle, target_value, strength=taper_strength)

# #     y, sd = add_post_settle_noise(
# #         y, time_vector, settling_time_s, target_value, rng,
# #         probability=0.9,
# #         post_sd_scale=(0.0008, 0.0013),
# #         smoothness_range=(10, 20),
# #         add_wobble_prob=0.18,
# #         wobble_scale=(0.00010, 0.00022),
# #     )

# #     return y, sd, "type1_Damped_Osc", float(settling_time_s * 1000.0), 0


# def generate_continuous_triangular_pulses(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
#     """
#     Type 2: Triangular pulse train (no explicit settling in current implementation).
#     Note: This remains as-is from your current behavior: continuous pulses + small noise.
#     """
#     y = np.full_like(time_vector, target_value, dtype=float)

#     is_height_const = bool(rng.choice([True, False]))
#     is_period_const = bool(rng.choice([True, False]))

#     avg_height = (limit_high - limit_low) * float(rng.uniform(0.5, 1.5))
#     avg_period = float(rng.uniform(settling_time_s / 3.0, settling_time_s / 1.5))
#     pulse_width = avg_period * float(rng.uniform(0.15, 0.30))
#     current_time = float(rng.uniform(0.0, avg_period * 0.3))

#     while current_time < float(time_vector[-1]):
#         height = avg_height if is_height_const else avg_height * float(rng.uniform(0.5, 1.5))

#         t_start = current_time
#         t_peak = current_time + pulse_width / 2.0
#         t_end = current_time + pulse_width

#         rise = (time_vector >= t_start) & (time_vector < t_peak)
#         if np.any(rise):
#             y[rise] += (height / (pulse_width / 2.0)) * (time_vector[rise] - t_start)

#         fall = (time_vector >= t_peak) & (time_vector < t_end)
#         if np.any(fall):
#             y[fall] += height - (height / (pulse_width / 2.0)) * (time_vector[fall] - t_peak)

#         period = avg_period if is_period_const else avg_period * float(rng.uniform(0.7, 1.3))
#         current_time += period

#     noise_level_pct = float(rng.uniform(0.00005, 0.0002))
#     sd = max(target_value * noise_level_pct, 1e-6)
#     early = time_vector < (0.2 * time_vector[-1])
#     y[early] = np.maximum(y[early], target_value)


#     return y, sd, "type2_Triangle_Wave", 0.0, 1


# def generate_low_swing_sine_wave(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
#     """
#     Type 3: Low-amplitude sine oscillation around the target value.
#     """
#     y = np.full_like(time_vector, target_value, dtype=float)

#     freq_hz = float(rng.uniform(200, 500))
#     w = 2.0 * np.pi * freq_hz
#     amplitude = (limit_high - limit_low) * float(rng.uniform(0.1, 0.25))

#     phase = float(rng.choice([0.0, np.pi]))
#     y += amplitude * np.sin(w * time_vector + phase)

#     noise_level_pct = float(rng.uniform(0.0, 0.00002))
#     sd = max(target_value * noise_level_pct, 1e-6)
#     y += rng.normal(0.0, sd, size=len(time_vector))

#     return y, sd, "type3_Sine_Wave", 0.0, 1


# def generate_overdamped_decay(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
#     # target_value, limit_low, limit_high, g = _scale_params(
#     # target_value, settling_time_s, limit_low, limit_high, rng, max_abs_value=50.0
#     # )
#     """
#     Type 4: Overdamped exponential decay toward the target value (no ringing).
#     """
#     start_amp = (limit_high - limit_low) * float(rng.uniform(1.5, 3.0))
#     tau = settling_time_s / float(rng.uniform(3.0, 5.0))

#     y = target_value + start_amp * np.exp(-time_vector / max(tau, 1e-12))
#     y = apply_cosine_taper_settling(y, time_vector, settling_time_s, target_value, strength=1.0)

#     y, sd = add_post_settle_noise(y, time_vector, settling_time_s, target_value, rng)
#     return y, sd, "type4_overdamped_no_overshoot", float(settling_time_s * 1000.0), 0

# def generate_overdamped_decay1(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
#     """
#     Type 4 (modified): Overdamped bi-exponential with a single undershoot ("overshoot before settling")
#     - No sinusoid, so still not 'ringing' in the oscillatory sense.
#     - Shape: start high -> dip below target once -> recover to target.
#     """
#     band = float(limit_high - limit_low)

#     # --- time constants (fast then slow) ---
#     # fast decays quickly, slow decays more slowly (controls recovery tail)
#     tau_fast = max(settling_time_s / float(rng.uniform(6.0, 12.0)), 1e-6)
#     tau_slow = max(settling_time_s / float(rng.uniform(1.8, 3.5)), 1e-6)

#     # --- amplitudes ---
#     # A_pos sets initial height above target
#     A_pos = band * float(rng.uniform(1.5, 3.0))

#     # A_neg controls how deep the undershoot is
#     # keep A_neg a bit smaller than A_pos so y(0) still above target
#     A_neg = A_pos * float(rng.uniform(0.35, 0.75))

#     # bi-exponential: +fast -slow  -> creates one dip below target then returns
#     y = target_value + A_pos * np.exp(-time_vector / tau_fast) - A_neg * np.exp(-time_vector / tau_slow)

#     # Optional: make the dip occur "ก่อนถึงเส้นนิ่ง" มากขึ้นด้วย time delay เล็ก ๆ
#     if rng.random() < 0.35:
#         t_eff, _ = add_time_delay(time_vector, rng, max_delay_s=0.0004)
#         y = target_value + A_pos * np.exp(-t_eff / tau_fast) - A_neg * np.exp(-t_eff / tau_slow)

#     # taper ให้เข้าหา target ช่วงใกล้ settling_time (กันปลายแกว่ง/เพี้ยนเกิน)
#     taper_strength = float(rng.uniform(0.85, 1.0))
#     y = apply_cosine_taper_settling(y, time_vector, settling_time_s, target_value, strength=taper_strength)

#     # post-settle noise
#     y, sd = add_post_settle_noise(y, time_vector, settling_time_s, target_value, rng)

#     return y, sd, "type4_overdamped_decay_overshoot", float(settling_time_s * 1000.0), 0


# def generate_pulse_train(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
#     """
#     Type 5: Steady baseline + POSITIVE square pulses (no settling).
#     - Baseline stays at the target value for the whole record.
#     - Pulses are periodic with per-pulse amplitude variation.
#     - There is a controlled probability (p_same) that consecutive pulses share the same amplitude.
#     - Guarantees at least one pulse is present.
#     """
#     y = np.full_like(time_vector, target_value, dtype=float)
#     band = float(limit_high - limit_low)

#     # Waveform-level amplitude family (high vs low)
#     amp_scale = float(rng.uniform(1.2, 2.5)) if rng.random() < 0.35 else float(rng.uniform(0.25, 1.1))
#     base_amp = band * amp_scale  # positive baseline pulse amplitude scale

#     # Period and width control (long period, short width)
#     t_end = float(time_vector[-1])
#     period = float(rng.uniform(t_end / 5.0, t_end / 2.0))
#     duty = float(rng.uniform(0.08, 0.20))
#     jitter_frac = float(rng.uniform(0.00, 0.08))

#     current_time = float(rng.uniform(0.0, period * 0.6))

#     # Amplitude repeat behavior
#     p_same = 0.35
#     prev_amp = None
#     has_pulse = False

#     while current_time < t_end:
#         this_period = period * float(rng.uniform(1.0 - jitter_frac, 1.0 + jitter_frac))
#         width = max(this_period * duty * float(rng.uniform(0.85, 1.15)), 1e-6)

#         t_start = current_time
#         t_stop = min(current_time + width, t_end)

#         mask = (time_vector >= t_start) & (time_vector < t_stop)
#         if np.any(mask):
#             has_pulse = True

#             if (prev_amp is not None) and (rng.random() < p_same):
#                 this_amp = prev_amp
#             else:
#                 this_amp = base_amp * float(rng.uniform(0.4, 1.6))
#                 prev_amp = this_amp

#             y[mask] += this_amp

#         current_time += this_period

#     # Ensure at least one pulse exists
#     if not has_pulse:
#         t_mid = 0.5 * t_end
#         width = 0.05 * t_end
#         mask = (time_vector >= t_mid) & (time_vector < t_mid + width)
#         y[mask] += base_amp

#     # Add only floor noise to stay within the same family as other generators
#     y, sd = add_post_settle_noise(
#         y, time_vector,
#         settling_time_s=0.0,
#         target_value=target_value,
#         rng=rng,
#         probability=0.0,
#         add_wobble_prob=0.0,
#     )

#     return y, sd, "type5_Square_Pulse_Wave", 0.0, 1


# def generate_no_undershoot_ringing(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
#     # target_value, limit_low, limit_high, g = _scale_params(
#     #     target_value, settling_time_s, limit_low, limit_high, rng, max_abs_value=50.0
#     # )
#     """
#     Ringing decay แต่ "ไม่ลงต่ำกว่า target" (no undershoot)
#     ได้ทรงแบบเส้นแดง: ลงมา + แกว่ง แต่ minimum อยู่เหนือ target แล้วค่อยนิ่ง
#     """
#     t = time_vector.astype(float)
#     t_end = float(t[-1]) if len(t) else 0.0
#     band = float(limit_high - limit_low)

#     # 1) baseline decay จากค่าเริ่มต้น (เริ่มสูงกว่า target)
#     start_offset = band * float(rng.uniform(0.6, 1.4))   # สูงกว่า target
#     tau_base = float(rng.uniform(max(t_end/8, 1e-6), max(t_end/3, 1e-6)))
#     base = target_value + start_offset * np.exp(-t / tau_base)

#     # 2) ringing แบบ "บวกเท่านั้น" (rectified) -> ไม่ undershoot
#     # เลือกความถี่ให้มี 2-5 รอบในช่วงก่อนนิ่ง
#     n_cycles = float(rng.uniform(2.0, 5.0))
#     f = n_cycles / max(settling_time_s, t_end, 1e-6)  # cycles per second
#     phi = float(rng.uniform(0, 2*np.pi))

#     ring_amp0 = band * float(rng.uniform(0.15, 0.45))
#     tau_ring = float(rng.uniform(max(t_end/10, 1e-6), max(t_end/2.5, 1e-6)))
#     envelope = np.exp(-t / tau_ring)

#     # rectified sine: (sin + 1)/2 อยู่ใน [0,1]
#     ring = ring_amp0 * envelope * (0.5 * (np.sin(2*np.pi*f*t + phi) + 1.0))

#     y = base + ring

#     # 3) กันหลุดต่ำกว่า target แบบชัวร์ (ถ้าคุณต้องการ 100% no-undershoot)
#     y = np.maximum(y, target_value)

#     # 4) noise เล็ก ๆ
#     noise_sd = max(abs(target_value) * float(rng.uniform(5e-5, 2e-4)), 1e-6)
#     y += rng.normal(0.0, noise_sd, size=len(t))

#     # กัน noise ทำหลุดต่ำกว่า target (ถ้าต้อง no-undershoot แบบสุด)
#     y = np.maximum(y, target_value)

#     return y, noise_sd, "typeX_NoUndershoot_Ringing", 0.0, 1

# # =============================================================================
# # 3) Main: Dataset Generation
# # =============================================================================
# def iter_generation_plan(n_waves: int, ratios, rng: np.random.Generator):
#     counts = []
#     allocated = 0
#     for func, r in ratios:
#         cnt = int(n_waves * float(r))
#         counts.append([func, cnt])
#         allocated += cnt

#     remainder = n_waves - allocated
#     if remainder > 0:
#         counts[0][1] += remainder

#     order = []
#     for func, cnt in counts:
#         order.extend([func] * cnt)

#     rng.shuffle(order)
#     yield from order


# def main():
#     ap = argparse.ArgumentParser("Generate synthetic TRAINING waveform data (FAST streaming)")
#     ap.add_argument("--out", default="data/raw/data_for_train.csv")
#     ap.add_argument("--n_waves", type=int, default=1000)
#     ap.add_argument("--dt_ms", type=float, default=0.01)
#     ap.add_argument("--t_end_ms", type=float, default=9.9)
#     ap.add_argument("--waves_per_flush", type=int, default=10, help="flush every N waves (controls RAM & speed)")
#     ap.add_argument("--seed", type=int, default=12345)
#     args = ap.parse_args()

#     out_path = Path(args.out)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     if out_path.exists():
#         out_path.unlink()

#     # Time axis (shared)
#     t_ms = np.arange(0.0, args.t_end_ms + 1e-12, args.dt_ms)
#     t_s = t_ms / 1000.0
#     n_samples = len(t_ms)

#     ratios = [
#         (generate_step_response,              0.26),
#         (generate_high_start_oscillation,     0.20),
#         (generate_overdamped_decay,           0.18),
#         (generate_overdamped_decay1,          0.12),
#         (generate_pulse_train,                0.12),
#         (generate_continuous_triangular_pulses, 0.06),
#         (generate_low_swing_sine_wave,        0.06),
#     ]


#     master_rng = np.random.default_rng(args.seed)
#     gen_sequence = iter_generation_plan(args.n_waves, ratios, master_rng)

#     print(f"Generating: waves={args.n_waves}, samples/wave={n_samples} -> total_rows={args.n_waves*n_samples:,}")
#     print(f"Output: {out_path}")

#     wrote_header = False
#     batch_frames = []

#     for wave_id, gen_func in enumerate(gen_sequence, start=1):
#         final_value = sample_target_0_50(master_rng, max_v=50.0, n_bins=10)  # 0..50 ครบช่วง
#         band_pct = float(master_rng.uniform(0.05, 0.15))
#         band = max(final_value * band_pct, 0.2)   # กัน band เล็กเกิน (ปรับ 0.2 ได้)
#         low = final_value - band / 2.0
#         high = final_value + band / 2.0

#         t_end_ms = float(args.t_end_ms)

#         max_settle_ms = 0.90 * t_end_ms  # กันชน: ไม่ให้ไปชนท้ายพอดี

#         p = master_rng.random()
#         if p < 0.78:
#             settle_time_ms = float(master_rng.uniform(2.0, min(12.0, max_settle_ms)))
#         elif p < 0.96:
#             lo = min(12.0, max_settle_ms)
#             hi = min(30.0, max_settle_ms)
#             settle_time_ms = float(master_rng.uniform(lo, hi)) if hi > lo else float(lo)
#         else:
#             lo = min(30.0, max_settle_ms)
#             hi = min(60.0, max_settle_ms)
#             settle_time_ms = float(master_rng.uniform(lo, hi)) if hi > lo else float(lo)

#         settle_s = settle_time_ms / 1000.0

#         wave_rng = np.random.default_rng(100000 + wave_id)
#         y, used_sd, type_name, true_settle_ms, true_is_zero = gen_func(
#             t_s, final_value, settle_s, low, high, wave_rng
#         )

#         if len(y) != n_samples:
#             raise RuntimeError(f"len(y)={len(y)} expected={n_samples} (wave_id={wave_id}, type={type_name})")

#         # ✅ vectorized per-wave dataframe (เร็วกว่า dict ต่อ sample มาก)
#         dfw = pd.DataFrame({
#             "wave_id": np.full(n_samples, wave_id, dtype=np.int32),
#             "type":   np.full(n_samples, type_name, dtype=object),
#             "sample": np.arange(n_samples, dtype=np.int32),
#             "time_ms": t_ms.astype(np.float32),
#             "value":  np.asarray(y, dtype=np.float32),
#             "sd":     np.full(n_samples, float(used_sd), dtype=np.float32),
#             "low_limit":  np.full(n_samples, float(low), dtype=np.float32),
#             "high_limit": np.full(n_samples, float(high), dtype=np.float32),
#                 # ✅ ground truth
#             "true_wait_time_ms": np.full(n_samples, float(true_settle_ms), dtype=np.float32),
#             "true_is_zero":      np.full(n_samples, int(true_is_zero), dtype=np.int8),

#             # ✅ debug: settle ที่ส่งเข้า generator
#             "settle_in_ms":      np.full(n_samples, float(settle_time_ms), dtype=np.float32),
#         })

#         batch_frames.append(dfw)

#         # flush every N waves
#         if (wave_id % args.waves_per_flush) == 0:
#             out_df = pd.concat(batch_frames, ignore_index=True)
#             out_df.to_csv(out_path, mode="a", index=False, header=(not wrote_header))
#             wrote_header = True
#             batch_frames.clear()

#             #print(f"  ... flushed wave {wave_id}/{args.n_waves} | wrote_rows={wave_id*n_samples:,}")

#     # flush last batch
#     if batch_frames:
#         out_df = pd.concat(batch_frames, ignore_index=True)
#         out_df.to_csv(out_path, mode="a", index=False, header=(not wrote_header))

#     print(f"✅ Done. Saved: {out_path}")


# if __name__ == "__main__":
#     main()

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
    strength: float = 1.0,
    gamma: float = 0.8,          # ✅ เพิ่มตัวนี้ (เร่งให้ settle ไวขึ้น)
) -> np.ndarray:
    """
    Smoothly forces the signal to approach the target value by the settling time
    using a cosine taper mask.

    gamma:
        >1.0  -> ลด deviation เร็วขึ้น (settle ไวขึ้น)
        =1.0  -> เดิม
        <1.0  -> ช้าลง
    """
    fade_mask = np.zeros_like(time_vector, dtype=float)
    active = time_vector < settling_time_s

    if np.any(active):
        t_ratio = time_vector[active] / max(settling_time_s, 1e-12)  # 0..1
        base = 0.5 * (1.0 + np.cos(np.pi * t_ratio))                 # 1..0
        fade_mask[active] = np.power(base, gamma)                    # ✅ เร่ง

    deviation = signal_array - target_value
    tapered = target_value + deviation * fade_mask

    return strength * tapered + (1.0 - strength) * signal_array



def add_post_settle_noise(
    signal_array: np.ndarray,
    time_vector: np.ndarray,
    settling_time_s: float,
    target_value: float,
    rng: np.random.Generator,
    probability: float = 0.65,
    post_sd_scale=(0.0004, 0.0009),
    smoothness_range=(18, 35),
    add_wobble_prob: float = 0.00,
    wobble_scale=(0.00001, 0.00005),
    wobble_win_range=(60, 130),
):
    """
    Noise scales with the magnitude of target_value.
    For very small targets (ns/us), floors are also very small (not 1e-6).
    """

    mag = max(abs(float(target_value)), 1e-12)

    # ✅ floor แบบสัมพันธ์กับ magnitude (ปรับได้)
    floor_abs = max(mag * 1e-4, 1e-15)     # 0.01% ของค่า

    # 1) base floor noise
    base_floor_sd = max(mag * rng.uniform(1e-4, 2.5e-4), floor_abs)
    signal_array = signal_array + rng.normal(0.0, base_floor_sd, size=len(time_vector))
    final_sd = base_floor_sd

    settle_idx = int(np.searchsorted(time_vector, settling_time_s))
    remaining_len = len(time_vector) - settle_idx
    if remaining_len <= 5:
        return signal_array, final_sd

    # 2) correlated wiggle
    if rng.random() < probability:
        post_sd = max(mag * rng.uniform(*post_sd_scale), floor_abs)
        final_sd = max(final_sd, post_sd)

        raw = rng.normal(0.0, post_sd, size=remaining_len)
        smoothness = int(rng.integers(smoothness_range[0], smoothness_range[1] + 1))
        smoothness = min(smoothness, remaining_len - 1)

        if smoothness > 2:
            k = np.ones(smoothness) / smoothness
            wig = np.convolve(raw, k, mode="same") * (math.sqrt(smoothness) * 0.35)
            signal_array[settle_idx:] += wig
        else:
            signal_array[settle_idx:] += raw

    # 3) optional wobble
    if rng.random() < add_wobble_prob:
        wob_sd = max(mag * rng.uniform(*wobble_scale), floor_abs)
        final_sd = max(final_sd, wob_sd)

        wob = rng.normal(0.0, wob_sd, size=remaining_len)
        win = int(rng.integers(wobble_win_range[0], wobble_win_range[1] + 1))
        win = min(win, remaining_len - 1)

        if win > 3:
            k2 = np.ones(win) / win
            wob = np.convolve(wob, k2, mode="same") * math.sqrt(win)
        wob = wob - np.mean(wob)
        wob = wob * 0.25
        signal_array[settle_idx:] += wob

    return signal_array, final_sd


def add_time_delay(t: np.ndarray, rng: np.random.Generator, max_delay_s: float = 0.0004):
    """Applies a random time delay (time shift). Returns (delayed_time, delay_value)."""
    d = float(rng.uniform(0.0, max_delay_s))
    return np.clip(t - d, 0.0, None), d


def sample_target_0_50(rng, max_v=50.0, n_bins=10):
    """Stratified sampling over 0..max_v to keep coverage across the range."""
    edges = np.linspace(0.0, max_v, n_bins + 1)
    b = int(rng.integers(0, n_bins))
    lo, hi = float(edges[b]), float(edges[b + 1])
    return float(rng.uniform(lo, hi))

def sample_target_mixed_units(rng: np.random.Generator):
    """
    Return target_value that can be:
      - ns scale: ~1e-9 .. 1e-8
      - us scale: ~1e-6 .. 1e-5
      - small:    ~1e-3 .. 1
      - normal:   ~1 .. 50
    Uses log-uniform inside each bucket so you get good coverage.
    """
    p = rng.random()

    # weights (ปรับได้ตามที่อยากให้ dataset เอนเอียง)
    if p < 0.18:  # 18% ns
        lo, hi = 1e-9, 1e-8
        return float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))
    elif p < 0.36:  # 18% us
        lo, hi = 1e-6, 1e-5
        return float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))
    elif p < 0.56:  # 20% small (milli..1)
        lo, hi = 1e-3, 1.0
        return float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))
    else:          # 44% original-ish range
        return float(rng.uniform(1.0, 50.0))

def soft_flatten_after_settle(y, t, settling_time_s, target, blend_window_s=0.0003):
    """Blend เข้า target ก่อน settle และ hold target หลัง settle."""
    if settling_time_s <= 0:
        return y

    N = len(y)
    si = int(np.searchsorted(t, settling_time_s))
    if si <= 1 or si >= N:
        return y

    t0 = max(settling_time_s - blend_window_s, 0.0)
    i0 = int(np.searchsorted(t, t0))
    i0 = max(0, min(i0, si))

    w = np.linspace(0.0, 1.0, max(si - i0, 1))
    y2 = y.copy()
    y2[i0:si] = (1 - w) * y2[i0:si] + w * target
    y2[si:] = target
    return y2

# =============================================================================
# 2) Waveform Generators (Type 0–5)
# =============================================================================

def generate_step_response(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """Type 0: Step response with damped ringing + measurement-like post-settle noise."""
    freq_hz = float(rng.uniform(100, 1200))
    w1 = 2.0 * np.pi * freq_hz
    band_half = (limit_high - limit_low) / 2.0

    overshoot_scale = float(rng.uniform(1.5, 8.0 if target_value < 1.0 else 15.0))
    direction = float(rng.choice([1.0, -1.0]))
    amp0 = band_half * overshoot_scale * direction

    tau = max(settling_time_s / float(rng.uniform(2.5, 6.0)), 1e-6)
    time_constant_rise = max(settling_time_s / float(rng.uniform(3.0, 10.0)), 1e-6)

    t_eff, _ = add_time_delay(time_vector, rng, max_delay_s=0.00004)

    if rng.random() < 0.5:
        base = target_value * (1.0 - np.exp(-t_eff / time_constant_rise))
    else:
        tau2 = max(time_constant_rise * float(rng.uniform(1.5, 4.0)), 1e-6)
        base = target_value * (1.0 - 0.6*np.exp(-t_eff/time_constant_rise) - 0.4*np.exp(-t_eff/tau2))

    if rng.random() < 0.6:
        ring = amp0 * np.exp(-t_eff / tau) * np.sin(w1 * t_eff)
    else:
        freq2 = freq_hz * float(rng.uniform(0.75, 1.25))
        w2 = 2.0 * np.pi * freq2
        mix = float(rng.uniform(0.2, 0.6))
        ring = amp0 * np.exp(-t_eff / tau) * ((1.0 - mix) * np.sin(w1 * t_eff) + mix * np.sin(w2 * t_eff))

    y = base + ring

    if rng.random() < 0.25:
        kick_amp = (limit_high - limit_low) * float(rng.uniform(0.15, 0.8)) * float(rng.choice([1.0, -1.0]))
        kick_tau = max(settling_time_s / float(rng.uniform(6.0, 14.0)), 1e-6)
        y += kick_amp * np.exp(-t_eff / kick_tau)

    taper_strength = float(rng.uniform(0.65, 0.70))
    y = apply_cosine_taper_settling(y, time_vector, settling_time_s, target_value, strength=taper_strength)

    # ------------------------------------------------------------------
    # ✅ เพิ่มตรงนี้: หลัง settle ให้ "นิ่งจริง" (ก่อนใส่ noise)
    # ------------------------------------------------------------------
    si = int(np.searchsorted(time_vector, settling_time_s))
    if 2 < si < len(y):
        dt = float(time_vector[1] - time_vector[0]) if len(time_vector) > 1 else 1e-6

        # ใช้ค่าเฉลี่ยก่อน settle เล็กน้อยเป็นค่าที่จะ hold (ดูสมจริง)
        win_s = 0.00015   # 0.15ms (ปรับได้)
        win_n = max(3, int(win_s / max(dt, 1e-12)))
        i0 = max(0, si - win_n)
        hold_val = float(np.mean(y[i0:si]))

        # blend ก่อน settle ให้เนียน กันเกิด step
        blend_s = 0.00025  # 0.25ms (ปรับได้)
        blend_n = max(1, int(blend_s / max(dt, 1e-12)))
        b0 = max(0, si - blend_n)

        if si > b0:
            w = np.linspace(0.0, 1.0, si - b0, endpoint=False)
            y[b0:si] = (1.0 - w) * y[b0:si] + w * hold_val

        # หลัง settle แบนจริง
        y[si:] = hold_val
    # ------------------------------------------------------------------

    # เติม noise ทีหลัง -> หลัง settle จะเห็นแค่ noise
    y, sd = add_post_settle_noise(
        y, time_vector, settling_time_s, target_value, rng,
        probability=0.95,
        smoothness_range=(10, 28),
        post_sd_scale=(0.0009, 0.0016),
        add_wobble_prob=0.35,
        wobble_scale=(0.00012, 0.00030),
    )

    return y, sd, "type0_Step_Response"

def generate_step_response_HARD(t, target, settle_s, low, high, rng):
    """Step response but with more realistic zeta/bw + optional slew limiting."""
    band = high - low
    t_eff = tiny_time_delay(t, rng)

    zeta = rng.uniform(0.15, 0.6)
    n_cycles = rng.uniform(1.5, 4.5)
    wd = 2 * np.pi * n_cycles / max(settle_s, 1e-6)
    wn = wd / math.sqrt(max(1 - zeta**2, 1e-6))

    sqrt_term = math.sqrt(max(1 - zeta**2, 1e-6))
    y = target * (
        1
        - np.exp(-zeta * wn * t_eff)
        * (np.cos(wd * t_eff) + (zeta / sqrt_term) * np.sin(wd * t_eff))
    )

    # initial condition offset
    ic = band * rng.uniform(-0.7, 0.7)
    y += ic * np.exp(-t_eff / (settle_s / rng.uniform(2.0, 6.0)))

    # optional slew-rate limit
    if rng.random() < SLEW_LIMIT_PROB:
        max_slope = abs(band) * rng.uniform(3000, 12000)
        dt = t[1] - t[0] if len(t) > 1 else 0.0
        max_step = max_slope * dt
        max_idx = min(len(y), int(0.001 / max(dt, 1e-12)))
        for i in range(1, max_idx):
            dy = y[i] - y[i - 1]
            y[i] = y[i - 1] + np.clip(dy, -max_step, max_step)

    y = apply_cosine_taper_settling(y, t, settle_s, target, rng.uniform(0.92, 0.99))
    y = soft_flatten_after_settle(y, t, settle_s, target)

    y, sd = add_post_settle_noise(y, t, settle_s, target, rng)
    return y, sd, "type0_HARD", 0.0, 0

def generate_high_start_oscillation(time_vector,target_value,settling_time_s,limit_low,limit_high,rng):
    """
    Type 1: Damped / underdamped oscillation with controlled first peak
    - ลด first peak ไม่ให้สูงโดดเกิน peak ถัดไป
    - ยังมี oscillation 2–4 รอบ
    - settle เนียน และหลัง settle เหลือแค่ noise
    """

    t = time_vector.astype(float)
    band = float(limit_high - limit_low)

    # -------------------------------------------------
    # 1) time delay เล็กน้อย (ไม่เริ่มพร้อมกันเป๊ะ)
    # -------------------------------------------------
    t_eff, _ = add_time_delay(t, rng, max_delay_s=0.00004)

    # -------------------------------------------------
    # 2) เลือกจำนวนรอบที่อยากเห็นก่อน settle
    # -------------------------------------------------
    num_cycles = float(rng.uniform(2.0, 4.0))
    freq = num_cycles / max(settling_time_s, 1e-6)
    w = 2.0 * np.pi * freq

    # -------------------------------------------------
    # 3) envelope decay (คุมให้หมดจริงก่อน settle)
    # -------------------------------------------------
    eps = float(rng.uniform(0.010, 0.025))   # ปลายเหลือ 1–2.5%
    tau_env = max(-settling_time_s / np.log(max(eps, 1e-9)), 1e-6)
    env = np.exp(-t_eff / tau_env)

    # -------------------------------------------------
    # 4) STARTUP RAMP  ⭐ จุดสำคัญ ⭐
    # ลด amplitude ตอนต้น → first peak ไม่โดด
    # ramp(t) = 1 - exp(-t/tau_ramp)
    # -------------------------------------------------
    tau_ramp = max(settling_time_s / float(rng.uniform(10.0, 20.0)), 1e-6)
    ramp = 1.0 - np.exp(-t_eff / tau_ramp)

    # -------------------------------------------------
    # 5) amplitude + bias
    # -------------------------------------------------
    A = band * float(rng.uniform(1.4, 2.6))   # ลดจากเดิมให้สุภาพขึ้น
    bias_amp = band * float(rng.uniform(-0.25, 0.25))
    bias_tau = max(settling_time_s / float(rng.uniform(1.6, 3.0)), 1e-6)
    bias = bias_amp * np.exp(-t_eff / bias_tau)

    # -------------------------------------------------
    # 6) oscillation (single-tone / optional 2-tone)
    # -------------------------------------------------
    if rng.random() < 0.15:
        w2 = w * float(rng.uniform(0.90, 1.10))
        mix = float(rng.uniform(0.25, 0.50))
        osc = (
            A
            * ramp
            * env
            * ((1.0 - mix) * np.cos(w * t_eff) + mix * np.cos(w2 * t_eff))
        )
    else:
        osc = A * ramp * env * np.cos(w * t_eff)

    y = target_value + bias + osc

    # -------------------------------------------------
    # 7) taper ให้เข้า target แบบเนียน
    # -------------------------------------------------
    y = apply_cosine_taper_settling(
        y,
        t,
        settling_time_s,
        target_value,
        strength=float(rng.uniform(0.92, 0.99))
    )

    # -------------------------------------------------
    # 9) post-settle noise (wiggle / wobble แบบสุภาพ)
    # -------------------------------------------------
    y, sd = add_post_settle_noise(
        y, t, settling_time_s, target_value, rng,
        probability=0.70,
        post_sd_scale=(0.00035, 0.00075),
        smoothness_range=(20, 38),
        add_wobble_prob=0.04,
        wobble_scale=(0.00005, 0.00012),
        wobble_win_range=(55, 110),
    )

    return y, sd, "type1_Damped_Osc"



def generate_continuous_triangular_pulses(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """Type 2: Triangular pulse train (continuous pulses + small noise).
    FIX: ไม่ผูก period กับ settling_time_s แล้ว → ผูกกับ t_end แทน
    """
    y = np.full_like(time_vector, target_value, dtype=float)

    is_height_const = bool(rng.choice([True, False]))
    is_period_const = bool(rng.choice([True, False]))

    band = float(limit_high - limit_low)
    avg_height = band * float(rng.uniform(0.5, 1.5))

    # -------------------------------
    # ✅ FIX: period อิงจากความยาวสัญญาณทั้ง record (t_end) ไม่ใช่ settling_time_s
    # -------------------------------
    t_end = float(time_vector[-1])

    # เลือก period ให้เห็นหลายพัลส์พอดีในช่วง record (ปรับช่วงได้)
    avg_period = float(rng.uniform(t_end / 20.0, t_end / 6.0))

    # ความกว้างพัลส์เป็นสัดส่วนของ period
    pulse_width = avg_period * float(rng.uniform(0.15, 0.30))

    start_after_s = 0.0005  # 0.5 ms
    current_time = max(float(settling_time_s), start_after_s) + float(rng.uniform(0.0, avg_period * 0.3))



    while current_time < t_end:
        height = avg_height if is_height_const else avg_height * float(rng.uniform(0.5, 1.5))

        t_start = current_time
        t_peak = current_time + pulse_width / 2.0
        t_end_pulse = current_time + pulse_width

        rise = (time_vector >= t_start) & (time_vector < t_peak)
        if np.any(rise):
            y[rise] += (height / (pulse_width / 2.0)) * (time_vector[rise] - t_start)

        fall = (time_vector >= t_peak) & (time_vector < t_end_pulse)
        if np.any(fall):
            y[fall] += height - (height / (pulse_width / 2.0)) * (time_vector[fall] - t_peak)

        # period jitter/variation
        period = avg_period if is_period_const else avg_period * float(rng.uniform(0.7, 1.3))
        current_time += period

    # noise เบาๆ
    noise_level_pct = float(rng.uniform(0.00005, 0.0002))
    mag = max(abs(float(target_value)), 1e-12)
    floor_abs = max(mag * 1e-4, 1e-15)   # หรือ mag*1e-5 ก็ได้
    sd = max(mag * noise_level_pct, floor_abs)
    y += rng.normal(0.0, sd, size=len(time_vector))

    # กัน early dip ต่ำกว่า baseline (ตามของเดิมคุณ)
    early = time_vector < (0.2 * time_vector[-1])
    y[early] = np.maximum(y[early], target_value)
    pre = time_vector < max(float(settling_time_s), start_after_s)
    y[pre] = target_value + rng.normal(0.0, sd, size=int(np.sum(pre)))

    return y, sd, "type2_Triangle_Wave"


# def generate_low_swing_sine_wave(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
#     """Type 3: Low-amplitude sine oscillation around the target value."""
#     y = np.full_like(time_vector, target_value, dtype=float)

#     freq_hz = float(rng.uniform(200, 500))
#     w = 2.0 * np.pi * freq_hz
#     amplitude = (limit_high - limit_low) * float(rng.uniform(0.1, 0.25))

#     phase = float(rng.choice([0.0, np.pi]))
#     y += amplitude * np.sin(w * time_vector + phase)

#     noise_level_pct = float(rng.uniform(0.0, 0.00002))
#     mag = max(abs(float(target_value)), 1e-12)
#     floor_abs = max(mag * 1e-4, 1e-15)
#     sd = max(mag * noise_level_pct, floor_abs)
#     y += rng.normal(0.0, sd, size=len(time_vector))

#     return y, sd, "type3_Sine_Wave"


def generate_overdamped_decay(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """Type 4: Overdamped exponential decay toward the target value (no ringing)."""
    start_amp = (limit_high - limit_low) * float(rng.uniform(1.5, 3.0))
    tau = settling_time_s / float(rng.uniform(3.0, 5.0))

    y = target_value + start_amp * np.exp(-time_vector / max(tau, 1e-12))
    y = apply_cosine_taper_settling(y, time_vector, settling_time_s, target_value, strength=1.0)


    y, sd = add_post_settle_noise(y, time_vector, settling_time_s, target_value, rng)
    return y, sd, "type4_overdamped_no_overshoot"


def generate_overdamped_decay1(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """
    Type 4 (modified): Overdamped bi-exponential with a single undershoot
    - Shape: start high -> dip below target once -> recover to target.
    - FIX: หลัง settling_time_s "แบนจริง" (ไม่นับ noise ที่เติมทีหลัง)
    """
    band = float(limit_high - limit_low)

    tau_fast = max(settling_time_s / float(rng.uniform(6.0, 12.0)), 1e-6)
    tau_slow = max(settling_time_s / float(rng.uniform(1.8, 3.5)), 1e-6)

    A_pos = band * float(rng.uniform(1.5, 3.0))
    A_neg = A_pos * float(rng.uniform(0.35, 0.75))

    y = target_value + A_pos * np.exp(-time_vector / tau_fast) - A_neg * np.exp(-time_vector / tau_slow)

    if rng.random() < 0.35:
        t_eff, _ = add_time_delay(time_vector, rng, max_delay_s=0.00004)
        y = target_value + A_pos * np.exp(-t_eff / tau_fast) - A_neg * np.exp(-t_eff / tau_slow)

    taper_strength = float(rng.uniform(0.55, 0.8))
    y = apply_cosine_taper_settling(y, time_vector, settling_time_s, target_value, strength=taper_strength)

    # ------------------------------------------------------------------
    # ✅ เพิ่มส่วนนี้: "ล็อกหลัง settle" ให้แบนจริง (ก่อนใส่ noise)
    # ------------------------------------------------------------------
    si = int(np.searchsorted(time_vector, settling_time_s))
    if 2 < si < len(y):
        # เลือกค่า hold จากค่าเฉลี่ยช่วงท้ายก่อน settle เพื่อให้ดูสมจริง
        # window 0.15ms (ปรับได้)
        dt = float(time_vector[1] - time_vector[0]) if len(time_vector) > 1 else 1e-6
        win_s = 0.00015
        win_n = max(3, int(win_s / max(dt, 1e-12)))
        i0 = max(0, si - win_n)
        hold_val = float(np.mean(y[i0:si]))

        # blend 0.25ms ให้เนียน ไม่เกิด step
        blend_s = 0.00025
        blend_n = max(1, int(blend_s / max(dt, 1e-12)))
        b0 = max(0, si - blend_n)

        if si > b0:
            w = np.linspace(0.0, 1.0, si - b0, endpoint=False)
            y[b0:si] = (1.0 - w) * y[b0:si] + w * hold_val

        # หลัง settle แบนจริง
        y[si:] = hold_val
    # ------------------------------------------------------------------

    # เติม noise ทีหลัง -> หลัง settle จะเห็นแค่ noise
    y, sd = add_post_settle_noise(y, time_vector, settling_time_s, target_value, rng)

    return y, sd, "type4_overdamped_decay_overshoot"


def generate_pulse_train(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """
    Type 5: Steady baseline + POSITIVE square pulses (no settling).
    - Baseline stays at the target value for the whole record.
    - Pulses are periodic with per-pulse amplitude variation.
    - Guarantees at least one pulse is present.
    """
    y = np.full_like(time_vector, target_value, dtype=float)
    band = float(limit_high - limit_low)

    amp_scale = float(rng.uniform(1.2, 2.5)) if rng.random() < 0.35 else float(rng.uniform(0.25, 1.1))
    base_amp = band * amp_scale

    t_end = float(time_vector[-1])
    period = float(rng.uniform(t_end / 5.0, t_end / 2.0))
    duty = float(rng.uniform(0.08, 0.20))
    jitter_frac = float(rng.uniform(0.00, 0.08))

    start_after_s = 0.0005  # 0.5 ms
    current_time = max(float(settling_time_s), start_after_s) + float(rng.uniform(0.0, period * 0.3))

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

    if not has_pulse:
        t_mid = 0.5 * t_end
        width = 0.05 * t_end
        mask = (time_vector >= t_mid) & (time_vector < t_mid + width)
        y[mask] += base_amp

    y, sd = add_post_settle_noise(
        y, time_vector,
        settling_time_s=0.0,
        target_value=target_value,
        rng=rng,
        probability=0.0,
        add_wobble_prob=0.0,
    )
    pre = time_vector < max(float(settling_time_s), start_after_s)
    y[pre] = target_value

    return y, sd, "type5_Square_Pulse_Wave"

def generate_pulse_train_HARD(time_vector, target_value, settling_time_s, limit_low, limit_high, rng):
    """
    Pulse train with duty / period wander + polarity flip
    """
    y = np.full_like(time_vector, target_value)
    band = limit_high - limit_low
    t_end = time_vector[-1]

    period = rng.uniform(t_end / 6, t_end / 2.2)
    cur = rng.uniform(0, period * 0.5)

    while cur < t_end:
        p = period * rng.uniform(0.7, 1.4)
        w = p * rng.uniform(0.08, 0.25)
        amp = band * rng.uniform(0.3, 2.0) * (1 if rng.random() > 0.15 else -1)

        mask = (time_vector >= cur) & (time_vector < cur + w)
        y[mask] += amp
        cur += p

    y, sd = add_post_settle_noise(
        y, time_vector, 0.0, target_value, rng, probability=0.0, add_wobble_prob=0.0
    )
    return y, sd, "type5_Square_Pulse_HARD"


# =============================================================================
# 3) Main: Dataset Generation
# =============================================================================

def iter_generation_plan(n_waves: int, ratios, rng: np.random.Generator):
    counts = []
    allocated = 0
    for func, r in ratios:
        cnt = int(n_waves * float(r))
        counts.append([func, cnt])
        allocated += cnt

    remainder = n_waves - allocated
    if remainder > 0:
        counts[0][1] += remainder

    order = []
    for func, cnt in counts:
        order.extend([func] * cnt)

    rng.shuffle(order)
    yield from order


def main():
    ap = argparse.ArgumentParser("Generate synthetic TRAINING waveform data (FAST streaming)")
    ap.add_argument("--out", default="data/raw/data_for_train.csv")
    ap.add_argument("--n_waves", type=int, default=1000)
    ap.add_argument("--dt_ms", type=float, default=0.01)
    ap.add_argument("--t_end_ms", type=float, default=9.9)
    ap.add_argument("--waves_per_flush", type=int, default=10, help="flush every N waves (controls RAM & speed)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    t_ms = np.arange(0.0, args.t_end_ms + 1e-12, args.dt_ms)
    t_s = t_ms / 1000.0
    n_samples = len(t_ms)

    ratios = [
        (generate_step_response,                 0.22),
        (generate_high_start_oscillation,        0.20),
        (generate_overdamped_decay,              0.20),
        (generate_overdamped_decay1,             0.12),
        (generate_pulse_train,                   0.12),
        (generate_continuous_triangular_pulses,  0.07),
        #(generate_low_swing_sine_wave,           0.06),
        (generate_pulse_train_HARD,              0.07)
    ]

    # ฟังก์ชันที่ "ไม่ควรมี settling" → บังคับ settle = 0.1ms
    no_settle_funcs = {
        generate_continuous_triangular_pulses,
        #generate_low_swing_sine_wave,
        generate_pulse_train,
        generate_pulse_train_HARD,
    }

    master_rng = np.random.default_rng(np.random.SeedSequence())
    gen_sequence = iter_generation_plan(args.n_waves, ratios, master_rng)

    print(f"Generating: waves={args.n_waves}, samples/wave={n_samples} -> total_rows={args.n_waves*n_samples:,}")
    print(f"Output: {out_path}")

    wrote_header = False
    batch_frames = []

    for wave_id, gen_func in enumerate(gen_sequence, start=1):
        final_value = sample_target_mixed_units(master_rng)
        band_pct = float(master_rng.uniform(0.05, 0.15))
        mag = max(abs(final_value), 1e-12)  # กัน log/ศูนย์
        band_pct = float(master_rng.uniform(0.05, 0.15))

        # floor เป็นสัดส่วนของ magnitude (เช่น 2% ของค่า) ไม่ใช่ 0.2 แบบตายตัว
        band_floor = mag * 0.02
        band = max(mag * band_pct, band_floor)

        low = final_value - band / 2.0
        high = final_value + band / 2.0

        t_end_ms = float(args.t_end_ms)
        max_settle_ms = 0.75 * t_end_ms  # กันไม่ให้ชิดท้าย (มีช่วงนิ่ง ~25%)

        # สุ่ม settle_time_ms (สำหรับ type ที่ต้องมี settling)
        p = master_rng.random()

        fast_lo = 1.5
        fast_hi = 0.55 * max_settle_ms
        mid_lo  = 0.55 * max_settle_ms
        mid_hi  = 0.85 * max_settle_ms
        slow_lo = 0.85 * max_settle_ms
        slow_hi = max_settle_ms

        fast_hi = max(fast_hi, fast_lo + 0.2)

        if p < 0.78:
            settle_time_ms = float(master_rng.uniform(fast_lo, fast_hi))
        elif p < 0.96:
            settle_time_ms = float(master_rng.uniform(mid_lo, mid_hi))
        else:
            settle_time_ms = float(master_rng.uniform(slow_lo, slow_hi))

        # บังคับ settle สำหรับชนิดที่ไม่ควรมี settling จริง
        if gen_func in no_settle_funcs:
            settle_time_ms = 0.1

        settle_s = settle_time_ms / 1000.0

        wave_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))

        y, used_sd, type_name = gen_func(t_s, final_value, settle_s, low, high, wave_rng)
        # -----------------------------
        # ✅ NEW: low/high limit = min/max "หลัง settle"
        # -----------------------------
        si = int(np.searchsorted(t_s, settle_s))

        # กรณี si เลยท้าย หรือ settle_s=0 แล้ว si=0 ก็ยังใช้ได้
        post = np.asarray(y[si:], dtype=float) if si < len(y) else np.asarray(y, dtype=float)

        low_settle  = float(np.min(post))
        high_settle = float(np.max(post))

        if len(y) != n_samples:
            raise RuntimeError(f"len(y)={len(y)} expected={n_samples} (wave_id={wave_id}, type={type_name})")

        dfw = pd.DataFrame({
            "wave_id": np.full(n_samples, wave_id, dtype=np.int32),
            "type":   np.full(n_samples, type_name, dtype=object),
            "sample": np.arange(n_samples, dtype=np.int32),
            "time_ms": t_ms.astype(np.float32),
            "value":  np.asarray(y, dtype=np.float64),
            "sd":     np.full(n_samples, float(used_sd), dtype=np.float64),
            "low_limit":  np.full(n_samples, low_settle, dtype=np.float64),
            "high_limit": np.full(n_samples, high_settle, dtype=np.float64),

            "wait_time_ms": np.full(n_samples, float(settle_time_ms), dtype=np.float32),
        })

        batch_frames.append(dfw)

        if (wave_id % args.waves_per_flush) == 0:
            out_df = pd.concat(batch_frames, ignore_index=True)
            out_df.to_csv(out_path, mode="a", index=False, header=(not wrote_header))
            wrote_header = True
            batch_frames.clear()

    if batch_frames:
        out_df = pd.concat(batch_frames, ignore_index=True)
        out_df.to_csv(out_path, mode="a", index=False, header=(not wrote_header))

    print(f"✅ Done. Saved: {out_path}")



if __name__ == "__main__":
    main()
