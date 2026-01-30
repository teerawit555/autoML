# # scripts/generate_predict_sample.py
# import argparse
# import math
# from pathlib import Path
# import numpy as np
# import pandas as pd

# # ===== import ของ train (ใช้ noise / taper เหมือนเดิม) =====
# from generate_train_sample import (
#     generate_step_response,
#     generate_high_start_oscillation,
#     generate_continuous_triangular_pulses,
#     generate_low_swing_sine_wave,
#     generate_overdamped_decay,
#     generate_pulse_train,
#     generate_overdamped_decay1,
#     apply_cosine_taper_settling,
#     add_post_settle_noise,
# )

# # =============================================================================
# # Utility (predict only)
# # =============================================================================

# def soft_flatten_after_settle(
#     y, t, settling_time_s, target, blend_window_s=0.0003
# ):
#     if settling_time_s <= 0:
#         return y

#     N = len(y)
#     si = int(np.searchsorted(t, settling_time_s))
#     if si <= 1 or si >= N:
#         return y

#     t0 = max(settling_time_s - blend_window_s, 0.0)
#     i0 = int(np.searchsorted(t, t0))
#     i0 = max(0, min(i0, si))

#     w = np.linspace(0.0, 1.0, max(si - i0, 1))
#     y2 = y.copy()
#     y2[i0:si] = (1 - w) * y2[i0:si] + w * target
#     y2[si:] = target
#     return y2


# def tiny_time_delay(t, rng, max_samples=2):
#     """
#     Apply a very small time shift (measurement trigger jitter).
#     This MUST NOT modify y.
#     """
#     if len(t) < 2:
#         return t

#     dt = t[1] - t[0]
#     d = rng.uniform(0.0, max_samples * dt)
#     return np.clip(t - d, 0.0, None)

# def apply_slew_limit(y, dt, max_slope, t_limit_s):
#     """
#     Apply slew-rate limit for the first t_limit_s seconds.
#     """
#     if dt <= 0:
#         return y

#     max_step = max_slope * dt
#     max_idx = min(len(y), int(t_limit_s / dt))
#     y2 = y.copy()

#     for i in range(1, max_idx):
#         dy = y2[i] - y2[i - 1]
#         y2[i] = y2[i - 1] + np.clip(dy, -max_step, max_step)

#     return y2



# # =============================================================================
# # REALISTIC HARD generators
# # =============================================================================

# def generate_step_response_HARD(t, target, settle_s, low, high, rng):
#     """
#     Step response but with:
#     - different damping
#     - different bandwidth
#     - optional slew-rate limit
#     """
#     band = high - low
#     t_eff = tiny_time_delay(t, rng)

#     # --- 2nd-order underdamped step (real system) ---
#     zeta = rng.uniform(0.15, 0.6)
#     n_cycles = rng.uniform(1.5, 4.5)
#     wd = 2 * np.pi * n_cycles / max(settle_s, 1e-6)
#     wn = wd / math.sqrt(max(1 - zeta**2, 1e-6))

#     sqrt_term = math.sqrt(max(1 - zeta**2, 1e-6))
#     y = target * (
#         1
#         - np.exp(-zeta * wn * t_eff)
#         * (np.cos(wd * t_eff) + (zeta / sqrt_term) * np.sin(wd * t_eff))
#     )

#     # initial condition offset (pre-charge)
#     ic = band * rng.uniform(-0.7, 0.7)
#     y += ic * np.exp(-t_eff / (settle_s / rng.uniform(2.0, 6.0)))

#     # optional slew-rate limit (very realistic)
#     if rng.random() < 0.5:
#         max_slope = abs(band) * rng.uniform(3000, 12000)
#         dt = t[1] - t[0] if len(t) > 1 else 0.0
#         max_step = max_slope * dt
#         for i in range(1, min(len(y), int(0.001 / dt))):
#             dy = y[i] - y[i-1]
#             y[i] = y[i-1] + np.clip(dy, -max_step, max_step)

#     # enforce same settle behavior
#     y = apply_cosine_taper_settling(y, t, settle_s, target, rng.uniform(0.92, 0.99))
#     y = soft_flatten_after_settle(y, t, settle_s, target)

#     y, sd = add_post_settle_noise(y, t, settle_s, target, rng)
#     return y, sd, "type0_HARD", 0.0, 0


# def generate_high_start_oscillation_HARD(t, target, settle_s, low, high, rng):
#     """
#     Underdamped oscillation with:
#     - wider zeta / freq
#     - realistic IC + slew
#     """
#     band = high - low
#     t_eff = tiny_time_delay(t, rng)

#     zeta = rng.uniform(0.12, 0.55)
#     n_cycles = rng.uniform(2.0, 5.0)
#     wd = 2 * np.pi * n_cycles / max(settle_s, 1e-6)
#     wn = wd / math.sqrt(max(1 - zeta**2, 1e-6))

#     sqrt_term = math.sqrt(max(1 - zeta**2, 1e-6))
#     y = target * (
#         1
#         - np.exp(-zeta * wn * t_eff)
#         * (np.cos(wd * t_eff) + (zeta / sqrt_term) * np.sin(wd * t_eff))
#     )

#     ic = band * rng.uniform(-1.0, 1.0)
#     y += ic * np.exp(-t_eff / (settle_s / rng.uniform(1.8, 4.5)))

#     # mild slew
#     if rng.random() < 0.6:
#         max_slope = abs(band) * rng.uniform(2500, 10000)
#         if rng.random() < 0.6:
#             dt = t[1] - t[0]
#             max_slope = abs(band) * rng.uniform(2500, 10000)
#             y = apply_slew_limit(y, dt, max_slope, t_limit_s=0.0015)

#     y = apply_cosine_taper_settling(y, t, settle_s, target, rng.uniform(0.92, 0.99))
#     y = soft_flatten_after_settle(y, t, settle_s, target)

#     y, sd = add_post_settle_noise(y, t, settle_s, target, rng)
#     return y, sd, "type1_HARD", 0.0, 0


# def generate_overdamped_decay_HARD(t, target, settle_s, low, high, rng):
#     """
#     Overdamped but multi-time-constant (real RC ladder / bias network)
#     """
#     band = high - low
#     t_eff = tiny_time_delay(t, rng)

#     a1 = band * rng.uniform(-2.0, 2.0)
#     a2 = band * rng.uniform(-1.5, 1.5)
#     tau1 = rng.uniform(0.0002, 0.0015)
#     tau2 = rng.uniform(0.001, 0.008)

#     y = target + a1 * np.exp(-t_eff / tau1) + a2 * np.exp(-t_eff / tau2)

#     y = apply_cosine_taper_settling(y, t, settle_s, target, rng.uniform(0.92, 0.99))
#     y = soft_flatten_after_settle(y, t, settle_s, target)

#     y, sd = add_post_settle_noise(y, t, settle_s, target, rng)
#     return y, sd, "type4_HARD", settle_s * 1000.0, 0


# def generate_continuous_triangular_pulses_HARD(t, target, settle_s, low, high, rng):
#     """
#     Same physics as triangle but with period / height wander
#     """
#     y = np.full_like(t, target)
#     band = high - low
#     t_end = t[-1]

#     period = rng.uniform(t_end / 6, t_end / 2.5)
#     cur = rng.uniform(0, period * 0.4)

#     while cur < t_end:
#         p = period * rng.uniform(0.8, 1.3)
#         h = band * rng.uniform(0.5, 1.6)
#         w = p * rng.uniform(0.15, 0.35)

#         r = (t >= cur) & (t < cur + w / 2)
#         f = (t >= cur + w / 2) & (t < cur + w)

#         y[r] += h * (t[r] - cur) / (w / 2)
#         y[f] += h * (1 - (t[f] - (cur + w / 2)) / (w / 2))

#         cur += p

#     y, sd = add_post_settle_noise(
#         y, t, 0.0, target, rng, probability=0.0, add_wobble_prob=0.0
#     )
#     return y, sd, "type2_HARD", 0.0, 1


# def generate_pulse_train_HARD(t, target, settle_s, low, high, rng):
#     """
#     Pulse train with duty / period wander + polarity flip
#     """
#     y = np.full_like(t, target)
#     band = high - low
#     t_end = t[-1]

#     period = rng.uniform(t_end / 6, t_end / 2.2)
#     cur = rng.uniform(0, period * 0.5)

#     while cur < t_end:
#         p = period * rng.uniform(0.7, 1.4)
#         w = p * rng.uniform(0.08, 0.25)
#         amp = band * rng.uniform(0.3, 2.0) * (1 if rng.random() > 0.15 else -1)

#         mask = (t >= cur) & (t < cur + w)
#         y[mask] += amp
#         cur += p

#     y, sd = add_post_settle_noise(
#         y, t, 0.0, target, rng, probability=0.0, add_wobble_prob=0.0
#     )
#     return y, sd, "type5_HARD", 0.0, 1


# # =============================================================================
# # main
# # =============================================================================

# def build_generation_plan(n, ratios, rng):
#     plan = []
#     for f, r in ratios:
#         plan += [f] * int(n * r)
#     while len(plan) < n:
#         plan.append(ratios[0][0])
#     rng.shuffle(plan)
#     return plan


# def sample_settle_time_caseA(rng, t_end_ms):
#     max_s = 0.75 * t_end_ms
#     p = rng.random()
#     if p < 0.78:
#         return rng.uniform(1.5, 0.55 * max_s)
#     elif p < 0.96:
#         return rng.uniform(0.55 * max_s, 0.85 * max_s)
#     else:
#         return rng.uniform(0.85 * max_s, max_s)


# def call_gen(func, *args):
#     out = func(*args)
#     if len(out) == 5:
#         return out
#     if len(out) == 3:
#         y, sd, name = out
#         return y, sd, name, 0.0, 0
#     raise RuntimeError(f"{func.__name__} bad return")


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--out", default="data/raw/data_predict.csv")
#     ap.add_argument("--n_waves", type=int, default=200)
#     ap.add_argument("--dt_ms", type=float, default=0.01)
#     ap.add_argument("--t_end_ms", type=float, default=9.9)
#     ap.add_argument("--seed", type=int, default=20251111)
#     args = ap.parse_args()

#     t_ms = np.arange(0, args.t_end_ms + 1e-12, args.dt_ms)
#     t = t_ms / 1000.0

#     ratios = [
#         (generate_step_response, 0.12),
#         (generate_step_response_HARD, 0.12),
#         (generate_high_start_oscillation, 0.10),
#         (generate_high_start_oscillation_HARD, 0.10),
#         (generate_continuous_triangular_pulses, 0.05),
#         (generate_continuous_triangular_pulses_HARD, 0.05),
#         (generate_low_swing_sine_wave, 0.04),
#         (generate_overdamped_decay, 0.05),
#         (generate_overdamped_decay_HARD, 0.05),
#         (generate_overdamped_decay1, 0.12),
#         (generate_pulse_train, 0.06),
#         (generate_pulse_train_HARD, 0.06),
#     ]

#     rng = np.random.default_rng(args.seed)
#     plan = build_generation_plan(args.n_waves, ratios, rng)

#     rows = []
#     for wid, gen in enumerate(plan, 1):
#         target = rng.uniform(0.5, 50.0)
#         band = max(target * rng.uniform(0.05, 0.15), 0.2)
#         low, high = target - band / 2, target + band / 2

#         settle_ms = sample_settle_time_caseA(rng, args.t_end_ms)
#         settle_s = settle_ms / 1000.0

#         wave_rng = np.random.default_rng(700000 + wid)
#         y, sd, name, _, _ = call_gen(gen, t, target, settle_s, low, high, wave_rng)

#         for i in range(len(t)):
#             rows.append({
#                 "wave_id": wid,
#                 "type": name,
#                 "sample": i,
#                 "time_ms": float(t_ms[i]),
#                 "value": float(y[i]),
#                 "sd": float(sd),
#                 "low_limit": float(low),
#                 "high_limit": float(high),
#             })

#     df = pd.DataFrame(rows)
#     Path(args.out).parent.mkdir(parents=True, exist_ok=True)
#     df.to_csv(args.out, index=False)
#     print(f"Saved realistic PREDICT data to {args.out}")


# if __name__ == "__main__":
#     main()


# scripts/generate_predict_sample.py
# scripts/generate_predict_sample.py
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ===== import ของ train (ใช้ noise / taper เหมือนเดิม) =====
from generate_train_sample import (
    generate_step_response,
    generate_high_start_oscillation,
    generate_continuous_triangular_pulses,
    #generate_low_swing_sine_wave,
    generate_overdamped_decay,
    generate_pulse_train,
    generate_overdamped_decay1,
    apply_cosine_taper_settling,
    add_post_settle_noise,
    sample_target_mixed_units,  
)

# =============================================================================
# CONFIG (รวมพารามิเตอร์ให้เรียบร้อย)
# =============================================================================

# --- time base ---
DEFAULT_DT_MS = 0.01
DEFAULT_T_END_MS = 9.9

# --- settle policy for pulse-like waves ---
PULSE_SETTLE_S = 0.0001        # 0.1 ms (เหมือน train ที่บังคับ no_settle funcs)
PULSE_START_AFTER_S = 0.0005   # 0.5 ms (ช่วงต้นยังไม่มี pulse)
MAX_FIRST_PULSE_S = 0.002      # 2 ms (กันเว้นหน้าเยอะ)

# --- time delay jitter ---
TINY_DELAY_MAX_SAMPLES = 2

# --- HARD slew realism ---
SLEW_LIMIT_PROB = 0.5
SLEW_T_LIMIT_S = 0.0015


# =============================================================================
# Utility (predict only)
# =============================================================================

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


def tiny_time_delay(t, rng, max_samples=TINY_DELAY_MAX_SAMPLES):
    """
    Apply a very small time shift (measurement trigger jitter).
    This MUST NOT modify y.
    """
    if len(t) < 2:
        return t
    dt = t[1] - t[0]
    d = rng.uniform(0.0, max_samples * dt)
    return np.clip(t - d, 0.0, None)


def apply_slew_limit(y, dt, max_slope, t_limit_s):
    """Apply slew-rate limit for the first t_limit_s seconds."""
    if dt <= 0:
        return y
    max_step = max_slope * dt
    max_idx = min(len(y), int(t_limit_s / dt))
    y2 = y.copy()
    for i in range(1, max_idx):
        dy = y2[i] - y2[i - 1]
        y2[i] = y2[i - 1] + np.clip(dy, -max_step, max_step)
    return y2


def pick_first_pulse_time(rng, t_end):
    """
    ให้ pulse แรกเกิดหลัง PULSE_START_AFTER_S
    และไม่ช้ากว่า MAX_FIRST_PULSE_S
    """
    hi = min(MAX_FIRST_PULSE_S, t_end * 0.9)
    lo = min(PULSE_START_AFTER_S, hi)
    return float(rng.uniform(lo, hi))


# =============================================================================
# REALISTIC HARD generators
# =============================================================================

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


def generate_high_start_oscillation_HARD(t, target, settle_s, low, high, rng):
    """Underdamped oscillation with realistic IC + optional slew."""
    band = high - low
    t_eff = tiny_time_delay(t, rng)

    zeta = rng.uniform(0.12, 0.55)
    n_cycles = rng.uniform(2.0, 5.0)
    wd = 2 * np.pi * n_cycles / max(settle_s, 1e-6)
    wn = wd / math.sqrt(max(1 - zeta**2, 1e-6))

    sqrt_term = math.sqrt(max(1 - zeta**2, 1e-6))
    y = target * (
        1
        - np.exp(-zeta * wn * t_eff)
        * (np.cos(wd * t_eff) + (zeta / sqrt_term) * np.sin(wd * t_eff))
    )

    ic = band * rng.uniform(-1.0, 1.0)
    y += ic * np.exp(-t_eff / (settle_s / rng.uniform(1.8, 4.5)))

    if rng.random() < 0.6 and len(t) > 1:
        dt = t[1] - t[0]
        max_slope = abs(band) * rng.uniform(2500, 10000)
        y = apply_slew_limit(y, dt, max_slope, t_limit_s=SLEW_T_LIMIT_S)

    y = apply_cosine_taper_settling(y, t, settle_s, target, rng.uniform(0.92, 0.99))
    y = soft_flatten_after_settle(y, t, settle_s, target)

    y, sd = add_post_settle_noise(y, t, settle_s, target, rng)
    return y, sd, "type1_HARD", 0.0, 0


def generate_overdamped_decay_HARD(t, target, settle_s, low, high, rng):
    """Overdamped but multi-time-constant (RC ladder / bias network)."""
    band = high - low
    t_eff = tiny_time_delay(t, rng)

    a1 = band * rng.uniform(-2.0, 2.0)
    a2 = band * rng.uniform(-1.5, 1.5)
    tau1 = rng.uniform(0.0002, 0.0015)
    tau2 = rng.uniform(0.001, 0.008)

    y = target + a1 * np.exp(-t_eff / tau1) + a2 * np.exp(-t_eff / tau2)

    y = apply_cosine_taper_settling(y, t, settle_s, target, rng.uniform(0.92, 0.99))
    y = soft_flatten_after_settle(y, t, settle_s, target)

    y, sd = add_post_settle_noise(y, t, settle_s, target, rng)
    return y, sd, "type4_HARD", settle_s * 1000.0, 0


def generate_continuous_triangular_pulses_HARD(t, target, settle_s, low, high, rng):
    """
    Triangle pulse train (hard)
    ✅ บังคับให้ช่วงต้นยังไม่มี pulse: first pulse >= 0.5ms
    ✅ กันเว้นหน้าเยอะ: first pulse <= 2ms
    """
    y = np.full_like(t, target)
    band = high - low
    t_end = float(t[-1])

    period = float(rng.uniform(t_end / 6, t_end / 2.5))
    cur = pick_first_pulse_time(rng, t_end)  # ✅

    while cur < t_end:
        p = period * float(rng.uniform(0.8, 1.3))
        h = band * float(rng.uniform(0.5, 1.6))
        w = p * float(rng.uniform(0.15, 0.35))

        r = (t >= cur) & (t < cur + w / 2)
        f = (t >= cur + w / 2) & (t < cur + w)

        if np.any(r):
            y[r] += h * (t[r] - cur) / (w / 2)
        if np.any(f):
            y[f] += h * (1 - (t[f] - (cur + w / 2)) / (w / 2))

        cur += p

    y, sd = add_post_settle_noise(y, t, 0.0, target, rng, probability=0.0, add_wobble_prob=0.0)
    return y, sd, "type2_HARD", 0.0, 1


def generate_pulse_train_HARD(t, target, settle_s, low, high, rng):
    """
    Square pulse train (hard)
    ✅ บังคับให้ช่วงต้นยังไม่มี pulse: first pulse >= 0.5ms
    ✅ กันเว้นหน้าเยอะ: first pulse <= 2ms
    """
    y = np.full_like(t, target)
    band = high - low
    t_end = float(t[-1])

    period = float(rng.uniform(t_end / 6, t_end / 2.2))
    cur = pick_first_pulse_time(rng, t_end)  # ✅

    while cur < t_end:
        p = period * float(rng.uniform(0.7, 1.4))
        w = p * float(rng.uniform(0.08, 0.25))
        amp = band * float(rng.uniform(0.3, 2.0)) * (1 if rng.random() > 0.15 else -1)

        mask = (t >= cur) & (t < cur + w)
        if np.any(mask):
            y[mask] += amp
        cur += p

    y, sd = add_post_settle_noise(y, t, 0.0, target, rng, probability=0.0, add_wobble_prob=0.0)
    return y, sd, "type5_HARD", 0.0, 1


# =============================================================================
# main helpers
# =============================================================================

def build_generation_plan(n, ratios, rng):
    plan = []
    for f, r in ratios:
        plan += [f] * int(n * r)
    while len(plan) < n:
        plan.append(ratios[0][0])
    rng.shuffle(plan)
    return plan


def sample_settle_time_caseA(rng, t_end_ms):
    """สุ่ม settle time แบบเดิมสำหรับพวกที่ต้อง settle จริง ๆ"""
    max_s = 0.75 * t_end_ms
    p = rng.random()
    if p < 0.78:
        return rng.uniform(1.5, 0.55 * max_s)
    elif p < 0.96:
        return rng.uniform(0.55 * max_s, 0.85 * max_s)
    else:
        return rng.uniform(0.85 * max_s, max_s)


def call_gen(func, *args):
    out = func(*args)
    if len(out) == 5:
        return out
    if len(out) == 3:
        y, sd, name = out
        return y, sd, name, 0.0, 0
    raise RuntimeError(f"{func.__name__} bad return")


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/data_predict.csv")
    ap.add_argument("--n_waves", type=int, default=200)
    ap.add_argument("--dt_ms", type=float, default=DEFAULT_DT_MS)
    ap.add_argument("--t_end_ms", type=float, default=DEFAULT_T_END_MS)
    ap.add_argument("--seed", type=int, default=20251111)
    args = ap.parse_args()

    t_ms = np.arange(0.0, args.t_end_ms + 1e-12, args.dt_ms)
    t = t_ms / 1000.0

    ratios = [
        (generate_step_response, 0.14),
        (generate_step_response_HARD, 0.14),
        (generate_high_start_oscillation, 0.10),
        (generate_high_start_oscillation_HARD, 0.10),
        (generate_continuous_triangular_pulses, 0.05),
        (generate_continuous_triangular_pulses_HARD, 0.05),
        #(generate_low_swing_sine_wave, 0.04),
        (generate_overdamped_decay, 0.05),
        (generate_overdamped_decay_HARD, 0.05),
        (generate_overdamped_decay1, 0.12),
        (generate_pulse_train, 0.06),
        (generate_pulse_train_HARD, 0.06),
    ]

    pulse_funcs = {
        generate_continuous_triangular_pulses,
        generate_continuous_triangular_pulses_HARD,
        generate_pulse_train,
        generate_pulse_train_HARD,
    }

    rng = np.random.default_rng(args.seed)
    plan = build_generation_plan(args.n_waves, ratios, rng)

    rows = []
    for wid, gen in enumerate(plan, start=1):

        # ✅ target มี ns/us ตาม train
        target = float(sample_target_mixed_units(rng))

        # ✅ band ไม่มี floor 0.2 (ใช้ floor ตาม magnitude)
        mag = max(abs(target), 1e-12)
        band_pct = float(rng.uniform(0.05, 0.15))
        band_floor = mag * 0.02
        band = float(max(mag * band_pct, band_floor))
        low, high = target - band / 2.0, target + band / 2.0

        # ✅ settle: พวกที่ต้อง settle สุ่มตามเดิม
        settle_ms = float(sample_settle_time_caseA(rng, args.t_end_ms))
        settle_s = settle_ms / 1000.0

        # ✅ pulse types: ไม่มี settle จริง
        if gen in pulse_funcs:
            settle_s = PULSE_SETTLE_S

        wave_rng = np.random.default_rng(700000 + wid)
        y, sd, name, _, _ = call_gen(gen, t, target, settle_s, low, high, wave_rng)

        # ----------------------------
        # low/high จากช่วง "หลัง settle" จริง
        # ----------------------------
        si = int(np.searchsorted(t, settle_s)) if settle_s > 0 else 0
        si = max(0, min(si, len(y) - 1))

        post = y[si:] if si < len(y) else y[-1:]
        if len(post) < 5:
            post = y

        low_settle = float(np.min(post))
        high_settle = float(np.max(post))

        for i in range(len(t)):
            rows.append({
                "wave_id": wid,
                "sample": int(i),
                "time_ms": float(t_ms[i]),
                "value": float(y[i]),
                "sd": float(sd),
                "low_limit": low_settle,
                "high_limit": high_settle,
                "type": str(name),
            })

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"✅ Saved realistic PREDICT data to: {out_path}")
    print(f"Config: PULSE_SETTLE_S={PULSE_SETTLE_S*1000:.3f}ms, "
          f"PULSE_START_AFTER_S={PULSE_START_AFTER_S*1000:.3f}ms, "
          f"MAX_FIRST_PULSE_S={MAX_FIRST_PULSE_S*1000:.3f}ms")



if __name__ == "__main__":
    main()
