# V25 - fix scale value 0 - 50 , 10ms 
# fix glitch ,  sine wave
# Change label (rule based) to true_wait_time_ms from generate_train_sample.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import medfilt

def _infer_dt_ms(times: np.ndarray, default=0.01) -> float:
    t = np.asarray(times, float)
    if len(t) < 3: return default
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if len(d) else default

def periodic_score(x_tail: np.ndarray) -> float:
    x = x_tail - np.median(x_tail)
    n = len(x)
    if n < 30:
        return 0.0

    denom = np.dot(x, x) + 1e-12
    best = 0.0
    for lag in range(3, min(n // 2, 150)):
        c = np.dot(x[:-lag], x[lag:]) / denom
        if c > best:
            best = c
    return float(best)

def autocorr_peak_features(x: np.ndarray, dt_ms: float) -> dict:
    """
    Leak-safe periodicity features from tail autocorrelation:
    - ac_best: best normalized autocorr in tail
    - ac_lag_ms: lag at best corr (ms)
    """
    x = np.asarray(x, float)
    N = len(x)
    if N < 120:
        return {}

    # ใช้ tail เป็นหลัก (กัน step ช่วงหัวหลอก)
    tail = x[-min(N, 400):].copy()
    tail = tail - float(np.median(tail))
    n = len(tail)
    if n < 60:
        return {}

    denom = float(np.dot(tail, tail) + 1e-12)

    best = -1.0
    best_lag = 0
    max_lag = min(n // 2, 220)
    min_lag = 3

    for lag in range(min_lag, max_lag):
        c = float(np.dot(tail[:-lag], tail[lag:]) / denom)
        if c > best:
            best = c
            best_lag = lag

    return {
        "tail_ac_best": float(best),
        "tail_ac_lag_ms": float(best_lag * dt_ms),
    }

def fft_periodic_features(x: np.ndarray, dt_ms: float) -> dict:
    
    out0= {
        "fft_peak_freq_hz": 0.0,
        "fft_peak_power_ratio": 0.0,
        "fft_lowfreq_power_ratio": 0.0,
        "fft_peak_to_2nd_ratio": 1.0,
        "fft_spectral_entropy": 1.0,  # entropy สูง = ไม่ tonal
        "tail_crest_factor": 0.0,
    }

    x = np.asarray(x, float)
    N = len(x)
    if N < 120:
        return out0

    tail = x[-min(N, 400):].copy()
    tail = tail - float(np.median(tail))
    n = len(tail)
    if n < 80:
        return out0

    dt_s = float(dt_ms) * 1e-3
    if not np.isfinite(dt_s) or dt_s <= 0:
        return out0

    w = np.hanning(n)
    y = tail * w

    spec = np.fft.rfft(y)
    pwr = (spec.real ** 2 + spec.imag ** 2).astype(float)
    freqs = np.fft.rfftfreq(n, d=dt_s)

    if len(pwr) <= 3:
        return out0

    pwr[0] = 0.0
    total = float(np.sum(pwr) + 1e-12)

    k = int(np.argmax(pwr))
    peak_freq = float(freqs[k])
    peak_ratio = float(pwr[k] / total)

    low_mask = freqs <= 50.0
    low_ratio = float(np.sum(pwr[low_mask]) / total) if np.any(low_mask) else 0.0

    pwr2 = pwr.copy()
    pwr2[k] = 0.0
    k2 = int(np.argmax(pwr2))
    peak2 = float(pwr2[k2])
    peak1 = float(pwr[k])
    peak_to_2nd = float(peak1 / (peak2 + 1e-12)) if peak1 > 0 else 1.0

    prob = np.clip(pwr / total, 1e-15, 1.0)
    H = -float(np.sum(prob * np.log(prob)))
    Hn = float(H / (np.log(len(prob)) + 1e-12))

    rms = float(np.sqrt(np.mean(tail ** 2)) + 1e-12)
    crest = float(np.max(np.abs(tail)) / rms)

    return {
        "fft_peak_freq_hz": peak_freq,
        "fft_peak_power_ratio": peak_ratio,
        "fft_lowfreq_power_ratio": low_ratio,
        "fft_peak_to_2nd_ratio": peak_to_2nd,
        "fft_spectral_entropy": Hn,
        "tail_crest_factor": crest,
    }

def tail_zcr_features(x: np.ndarray) -> dict:
    """
    Zero-crossing-ish rate around tail median (sine จะสูง)
    """
    x = np.asarray(x, float)
    N = len(x)
    if N < 120:
        return {}

    tail = x[-min(N, 400):]
    m = float(np.median(tail))
    s = np.sign(tail - m)
    zc = int(np.sum(np.diff(s) != 0))
    rate = float(zc / max(len(tail) - 1, 1))
    return {
        "tail_zc_rate": rate,
    }


def tail_envelope_flatness(x: np.ndarray) -> dict:
    """
    Sine steady oscillation: std_mid ~ std_tail (ratio ~1)
    Damped ringing: std_tail << std_mid (ratio > 1)
    """
    x = np.asarray(x, float)
    N = len(x)
    if N < 240:
        return {}

    # ใช้ 2 ช่วงใน tail
    a = x[-400:-200] if N >= 400 else x[max(N-240, 0):max(N-120, 0)]
    b = x[-200:]      if N >= 200 else x[-120:]

    if len(a) < 50 or len(b) < 50:
        return {}

    std_a = float(np.std(a)) + 1e-12
    std_b = float(np.std(b)) + 1e-12

    return {
        "tail_std_ratio_mid_to_last": float(std_a / std_b),
    }


def _best_stable_tail_level(values: np.ndarray, times: np.ndarray, *, tail_ms: float, win_ms: float) -> tuple[float, float]:
    """
    หา end_ref ที่ไม่โดน late pulse หลอก:
    - มองเฉพาะ tail_ms สุดท้าย
    - สไลด์ window ขนาด win_ms แล้วเลือก window ที่ std ต่ำสุด
    return: (end_ref, end_noise_std)
    """
    x = np.asarray(values, float)
    t = np.asarray(times, float)
    N = len(x)
    if N < 20:
        m = float(np.median(x)) if N else 0.0
        return m, float(np.std(x)) if N else 0.0

    t_end = float(t[-1])
    t0 = t_end - float(tail_ms)
    idx_tail = np.where(t >= t0)[0]
    if len(idx_tail) < 20:
        idx_tail = np.arange(max(0, N - 50), N)

    xt = x[idx_tail]
    tt = t[idx_tail]

    # window size in samples
    #dt = np.median(np.diff(tt[tt.size > 1])) if len(tt) > 2 else (np.median(np.diff(t)) if len(t) > 2 else 0.01)
    if len(tt) > 2:
        dtt = np.diff(tt)
        dtt = dtt[np.isfinite(dtt) & (dtt > 0)]
        dt = float(np.median(dtt)) if len(dtt) else 0.01
    else:
        dt = 0.01
    #dt = float(dt) if np.isfinite(dt) and dt > 0 else 0.01
    W = int(max(10, round(win_ms / dt)))

    if len(xt) <= W:
        m = float(np.median(xt))
        mad = float(np.median(np.abs(xt - m))) + 1e-12
        return m, 1.4826 * mad

    best_std = float("inf")
    best_med = float(np.median(xt[-W:]))
    best_mad = float(np.median(np.abs(xt[-W:] - best_med))) + 1e-12

    for i in range(0, len(xt) - W + 1):
        w = xt[i:i+W]
        # robust std via MAD
        med = float(np.median(w))
        mad = float(np.median(np.abs(w - med))) + 1e-12
        rstd = 1.4826 * mad
        if rstd < best_std:
            best_std = rstd
            best_med = med
            best_mad = mad

    return best_med, 1.4826 * best_mad


def normalize_wave(values: np.ndarray, times: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    x = np.asarray(values, dtype=float)
    N = len(x)

    if times is None or len(times) != N:
        t = np.arange(N, dtype=float)
        has_time = False
    else:
        t = np.asarray(times, dtype=float)
        has_time = True

    # ---- start ref ----
    if has_time:
        t_start = float(t[0])
        idx_start = np.where(t <= t_start + 0.4)[0]  # 0.4ms
        if len(idx_start) < 10:
            idx_start = np.arange(min(30, N))
    else:
        idx_start = np.arange(min(30, N))
    start_ref = float(np.median(x[idx_start])) if len(idx_start) else float(np.median(x))

    # ---- end ref ----
    if has_time:
        end_ref, end_noise = _best_stable_tail_level(x, t, tail_ms=2.0, win_ms=0.6)
    else:
        seg = x[-min(50, N):]
        end_ref = float(np.median(seg))
        mad = float(np.median(np.abs(seg - end_ref))) + 1e-12
        end_noise = 1.4826 * mad

    #end_noise = float(max(end_noise, 1e-6))
    # scale reference (raw domain)
    # span_raw = float(max(p95 - p05, 0.0))
    # sig_scale = max(abs(end_ref), abs(start_ref), span_raw, 1e-15)

    # # floors ที่ scale ตามสัญญาณ (ไม่ fix 1e-6)
    # noise_floor = max(sig_scale * 1e-6, 1e-12)   # 1 ppm ของสเกล หรืออย่างต่ำ 1e-12
    # span_floor  = max(sig_scale * 1e-5, 1e-12)   # 10 ppm ของสเกล หรืออย่างต่ำ 1e-12

    # end_noise = float(max(end_noise, noise_floor))
    # global_span = float(max(span_raw, span_floor))

    # denom_used = max(step_amp, 0.12 * global_span, 6.0 * end_noise, noise_floor)


    # ---- robust GLOBAL scale (critical) ----
    # กัน no-step ทำให้ denom เล็กจน xN ระเบิด
    p95 = float(np.percentile(x, 95))
    p05 = float(np.percentile(x, 5))
  #  global_span = max(p95 - p05, 1e-6)

    step_raw = float(end_ref - start_ref)
    direction = 1.0 if step_raw >= 0 else -1.0
    step_amp = float(abs(step_raw))

    # denom = max(step_amp, small fraction of global span, noise floor)
   # denom_used = max(step_amp, 0.12 * global_span, 6.0 * end_noise, 1e-6)
       # ---- robust GLOBAL scale (critical) ----
    # ใช้ percentile แบบ safe + handle NaN/Inf
    xf = x[np.isfinite(x)]
    if xf.size >= 5:
        p95 = float(np.percentile(xf, 95))
        p05 = float(np.percentile(xf, 5))
    else:
        p95 = float(np.nanmax(x)) if N else 0.0
        p05 = float(np.nanmin(x)) if N else 0.0

    span_raw = float(max(p95 - p05, 0.0))

    step_raw = float(end_ref - start_ref)
    direction = 1.0 if step_raw >= 0 else -1.0
    step_amp = float(abs(step_raw))

    # scale reference (raw domain) for micro/nano
    sig_scale = max(abs(end_ref), abs(start_ref), span_raw, 1e-15)

    # floors that scale with signal magnitude (not fixed 1e-6)
    noise_floor = max(sig_scale * 1e-6, 1e-12)   # 1 ppm or >= 1e-12
    span_floor  = max(sig_scale * 1e-5, 1e-12)   # 10 ppm or >= 1e-12

    global_span = float(max(span_raw, span_floor))

    # end_noise already computed from tail window earlier; just floor it here
    end_noise = float(max(end_noise, noise_floor))

    denom_used = max(step_amp, 0.12 * global_span, 6.0 * end_noise, noise_floor)

    xN = (x - end_ref) / denom_used
    xN = np.clip(xN, -12.0, 12.0)

    meta = {
        "start_med": float(start_ref),
        "end_ref": float(end_ref),
        "step_raw": float(step_raw),
        "denom_used": float(denom_used),
        "direction": float(direction),
        "end_noise": float(end_noise),
        "global_span": float(global_span),
    }
    return xN, meta

# For labelling data for train (not used in this version)
def post_settle_hold(
    settle_idx: int,
    search_values: np.ndarray,
    target: float,
    tol: float,
    dt_ms: float,
    *,
    direction: float,
    has_undershoot: bool,
    is_damped_sine: bool,
    global_crossing_rate: float,
    crossing_rate: float,
) -> int:
    """
    กัน settle ผ่านเร็วเกิน (เข้า band ชั่วคราวแล้วหลุด/ไหลต่อ)

    สิ่งที่ทำเพิ่มจากเดิม:
    1) band-hold: ต้องอยู่ใน tol ต่อเนื่องช่วงหนึ่ง (hold window)
    2) slope-hold: กัน step/rise ที่ "ไหลต่อ" แม้อยู่ใน tol (เช็คความชันใน hold window)

    แนวคิด:
    - rise: มัก creep ต่อ -> hold เพิ่ม + ตรวจ slope
    - undershoot/rebound: hold เพิ่ม
    - damped/ringing: hold เพิ่ม
    """
    N = len(search_values)
    if N <= 5:
        return int(np.clip(settle_idx, 0, max(N - 1, 0)))

    # -------------------------
    # decide hold time (ms)
    # -------------------------
    hold_ms = 0.20  # base

    # rise มักไหลต่อ -> เพิ่ม hold
    if direction > 0:
        hold_ms += 0.06

    # undershoot/rebound -> เพิ่ม hold
    if has_undershoot:
        hold_ms += 0.05

    # ringing/damped -> เพิ่ม hold
    is_ringing = (global_crossing_rate > 0.05) or (crossing_rate > 0.08)
    if is_damped_sine or is_ringing:
        hold_ms += 0.15

    # clamp กันหลุดโลก
    hold_ms = float(np.clip(hold_ms, 0.10, 0.45))

    dt = max(float(dt_ms), 1e-9)
    HN = int(np.ceil(hold_ms / dt))
    if HN <= 1:
        return int(np.clip(settle_idx, 0, N - 1))

    settle_idx = int(np.clip(settle_idx, 0, N - 1))
    end = int(min(N, settle_idx + HN))
    if end - settle_idx < 2:
        return settle_idx

    # -------------------------
    # 1) band check in hold window
    # -------------------------
    post = search_values[settle_idx:end]
    dev = np.abs(post - target)

    bad = np.where(dev > tol)[0]
    if len(bad) > 0:
        # หลุด band ในช่วง hold -> เลื่อนหลังจุดหลุดสุดท้าย
        new_idx = settle_idx + int(bad[-1]) + 1
        return int(min(new_idx, N - 1))

    # -------------------------
    # 2) slope check (กัน "ไหลต่อ" โดยเฉพาะ rise)
    # -------------------------
    # ใช้ slope_tol ผูกกับ tol เพื่อให้ scale-invariant:
    # ถ้า tol แคบ -> slope ต้องนิ่งกว่า
    # ตั้ง slope_tol เป็น "การเปลี่ยนแปลงต่อเวลา" ใน normalized domain
    if end - settle_idx >= 3:
        dp = np.diff(search_values[settle_idx:end])
        slope = np.abs(dp) / dt

        # base slope tolerance
        slope_tol = (2.5 * tol) / dt

        # ถ้าเป็น rise ให้เข้มขึ้นนิด (กัน creep)
        if direction > 0:
            slope_tol *= 0.85  # tighter

        # ถ้าเป็น ringing/damped ให้ผ่อนนิด (กัน false positive)
        if is_damped_sine or is_ringing:
            slope_tol *= 1.25  # looser

        # หา index สุดท้ายที่ slope เกิน
        bad_s = np.where(slope > slope_tol)[0]
        if len(bad_s) > 0:
            last_bad = int(bad_s[-1])
            # +2 เพราะ slope index i คือช่วง [i -> i+1]
            new_idx = settle_idx + last_bad + 2
            return int(min(new_idx, N - 1))

    return settle_idx

def periodic_features(xN: np.ndarray, dt_ms: float) -> dict:
    N = len(xN)
    if N < 60:
        return {
            "per_score_tail": 0.0,
            "global_crossing_rate": 0.0,
            "tail_crossing_rate": 0.0,
            "tail_range": 0.0,
            "tail_std": 0.0,
        }

    # tail 2ms (หรือเท่าที่มี)
    tail_len = int(max(30, min(N, round(2.0 / max(dt_ms, 1e-6)))))
    tail = xN[-tail_len:]

    # periodic score ใช้ของเดิม
    per = periodic_score(tail)

    # crossing rate global (robust baseline)
    ref_g = float(np.median(xN))
    cross_g = float(np.sum(np.diff(np.sign(xN - ref_g)) != 0) / max(N, 1))

    # crossing rate tail (จับ sine/ringing ชัดกว่า)
    ref_t = float(np.median(tail))
    cross_t = float(np.sum(np.diff(np.sign(tail - ref_t)) != 0) / max(len(tail), 1))

    tail_range = float(np.ptp(tail))
    tail_std = float(np.std(tail))

    return {
        "per_score_tail": float(per),
        "global_crossing_rate": float(cross_g),
        "tail_crossing_rate": float(cross_t),
        "tail_range": float(tail_range),
        "tail_std": float(tail_std),
    }

# ----------------------------
# “Features” ที่ไม่ใช่คำเฉลยตรงๆ
# ----------------------------
def compute_features_from_row(x: np.ndarray, t: np.ndarray | None = None) -> dict:
    """
    leak-safe engineered features (ไม่ใช่คำเฉลยตรงๆ)
    เพิ่มฟีเจอร์เพื่อช่วยเคส fall / damp / ringing:
      - direction (rise/fall)
      - tail drift slope (จับ droop / ไหลต่อ)
      - monotonicity near tail (ยังไหลอยู่แม้ใกล้ target)
      - overshoot/undershoot relative to end
      - last crossing time (ยังสวิงข้าม end)
    """
    N = len(x)
    if N < 60:
        return {}

    # ใช้ time ถ้ามี ไม่งั้นใช้ index
    if t is None or len(t) != N:
        tt = np.arange(N, dtype=float)
    else:
        tt = t.astype(float)

    # robust start/end
    head_n = min(30, N)
    tail_n = min(50, N)

    x_head = x[:head_n]
    x_tail = x[-tail_n:]
    start_med = float(np.median(x_head))
    end_med = float(np.median(x_tail))

    delta_start_end = float(end_med - start_med)
    direction = 1.0 if delta_start_end >= 0 else -1.0  # +1 rise, -1 fall

    # basic
    x_end = float(x[-1])
    std_all = float(np.std(x))
    dx = np.diff(x)
    abs_dx = np.abs(dx)

    ringing_energy = float(np.sum(abs_dx[3:])) if len(abs_dx) > 3 else float(np.sum(abs_dx))
    max_slope = float(np.max(dx)) if len(dx) else 0.0
    min_slope = float(np.min(dx)) if len(dx) else 0.0

    # rolling std on second half (เดิม)
    half_idx = N // 2
    half_signal = x[half_idx:]
    window_size = 100
    if len(half_signal) > window_size:
        windows = np.lib.stride_tricks.sliding_window_view(half_signal, window_size)
        rolling_std = np.std(windows, axis=1)
        max_rolling_std_half = float(np.max(rolling_std))
    else:
        max_rolling_std_half = std_all

    tail_200 = x[-min(N, 200):]
    tail_100 = x[-min(N, 100):]
    std_tail_50 = float(np.std(x_tail))
    max_dev_tail_50 = float(np.max(np.abs(x_tail - end_med)))

    # crossing rate around tail median (เดิม)
    median_tail = np.median(tail_200)
    zero_crossings = len(np.where(np.diff(np.sign(tail_200 - median_tail)))[0])
    crossing_rate = zero_crossings / max(len(tail_200), 1)

    # drift score (เดิม)
    x_pd = pd.Series(x)
    smooth_fast = x_pd.rolling(10).mean().bfill().values
    smooth_slow = x_pd.rolling(50).mean().bfill().values
    drift_score = float(np.mean(np.abs(smooth_fast[-100:] - smooth_slow[-100:]))) if N >= 100 else float(np.mean(np.abs(smooth_fast - smooth_slow)))

    # ==========================
    # NEW: slope/drift features
    # ==========================
    def _fit_slope(y: np.ndarray, tx: np.ndarray) -> float:
        if len(y) < 5:
            return 0.0
        # robust-ish: polyfit degree1
        try:
            m = np.polyfit(tx, y, 1)[0]
            return float(m)
        except Exception:
            return 0.0

    # global slope (ช่วยแยก “ยังไหล”)
    global_slope = _fit_slope(x, tt)

    # tail slope (จับ droop/ไหลต่อ) — สำคัญกับ fall
    k200 = min(N, 200)
    tail200_slope = _fit_slope(x[-k200:], tt[-k200:])

    k100 = min(N, 100)
    tail100_slope = _fit_slope(x[-k100:], tt[-k100:])

    # normalize slope by step size กัน scale เพี้ยน
    step_amp = float(np.abs(delta_start_end)) + 1e-9
    tail200_slope_norm = float(tail200_slope / step_amp)
    tail100_slope_norm = float(tail100_slope / step_amp)

    # ==========================
    # NEW: fall recovery features
    # ==========================
    # อิงกับ end_med (median tail) เป็น reference ที่ robust

    # 1) minimum position/time ratio (ตกเร็วแค่ไหน)
    min_i = int(np.argmin(x))
    min_pos_ratio = float(min_i / max(N - 1, 1))
    min_time = float(tt[min_i])

    # ถ้า fall แล้วต่ำกว่า end_med -> end_med - min จะเป็นบวก
    min_to_end = float(end_med - float(x[min_i]))
    min_to_end_norm = float(min_to_end / (step_amp + 1e-12))

    # 2) rebound size: หลัง minimum เด้งกลับขึ้นไปได้เท่าไหร่
    if min_i < N - 2:
        post_min = x[min_i:]
        post_min_t = tt[min_i:]
    else:
        post_min = x[-2:]
        post_min_t = tt[-2:]

    post_min_peak = float(np.max(post_min))
    rebound = float(post_min_peak - float(x[min_i]))
    rebound_norm = float(rebound / (step_amp + 1e-12))

    # 3) slope หลัง minimum (จับ “ไต่กลับ”)
    k_post = min(200, len(post_min))
    post_min_slope = _fit_slope(post_min[:k_post], post_min_t[:k_post])
    post_min_slope_norm = float(post_min_slope / (step_amp + 1e-12))

    # 4) tail creep: median ช่วงท้าย-กลาง ต่างกันไหม (ยังไหลอยู่)
    # ใช้ median ช่วง [-200:-100] เทียบ [-100:]
    if N >= 220:
        mid_tail_med = float(np.median(x[-200:-100]))
        last_tail_med = float(np.median(x[-100:]))
    else:
        a = max(N - 120, 0)
        b = max(N - 60, 0)
        c = max(N - 60, 0)
        mid_tail_med = float(np.median(x[a:b])) if b > a else float(np.median(x[:max(N,1)]))
        last_tail_med = float(np.median(x[c:])) if N > 0 else 0.0

    tail_creep = float(last_tail_med - mid_tail_med)
    tail_creep_norm = float(tail_creep / (step_amp + 1e-12))

    # ==========================
    # NEW: monotonicity near tail
    # ==========================
    # ถ้า fall แล้วยังไหลลงต่อ dx จะติดลบเยอะ (direction=-1)
    tail_dx = np.diff(tail_200) if len(tail_200) > 1 else np.array([], dtype=float)
    if len(tail_dx):
        same_dir = (np.sign(tail_dx) == direction).mean()  # % ที่ไปทิศเดียวกับ step
        # บางที ringing จะสลับไปมา → same_dir ต่ำ
        tail_monotonicity = float(same_dir)
        tail_abs_slope_mean = float(np.mean(np.abs(tail_dx)))
        tail_signed_slope_mean = float(np.mean(tail_dx))
    else:
        tail_monotonicity = 0.0
        tail_abs_slope_mean = 0.0
        tail_signed_slope_mean = 0.0

    # ==========================
    # NEW: overshoot/undershoot vs end_med
    # ==========================
    # rise: overshoot = max(x - end), fall: undershoot = max(end - x)
    overshoot = float(np.max(x - end_med))
    undershoot = float(np.max(end_med - x))
    # ให้โมเดลรู้ว่า fall แบบ “ตกเลย + ต่ำกว่า end แล้วค่อยเด้ง” มี pattern
    overshoot_norm = float(overshoot / step_amp)
    undershoot_norm = float(undershoot / step_amp)

    # ==========================
    # NEW: last crossing time relative to end_med
    # ==========================
    # ถ้า damp sine ยาว: จะข้าม end_med หลายครั้ง → last_cross จะช้า
    diff_end = x - end_med
    cross_idx = np.where(np.diff(np.sign(diff_end)) != 0)[0]
    if len(cross_idx):
        last_cross_i = int(cross_idx[-1])
        last_cross_time = float(tt[last_cross_i])
        # ระยะจากท้าย (ยิ่งน้อย = ยังข้ามใกล้ๆท้าย)
        last_cross_tail_ratio = float((N - 1 - last_cross_i) / max(N, 1))
    else:
        last_cross_time = float(tt[0])
        last_cross_tail_ratio = 1.0

    # ==========================
    # NEW: envelope ratio (mid vs tail) แยก damp ยาว
    # ==========================
    mid_start = int(N * 0.35)
    mid_end = int(N * 0.65)
    mid_seg = x[mid_start:mid_end] if mid_end > mid_start else x
    std_mid = float(np.std(mid_seg)) if len(mid_seg) else std_all
    envelope_ratio = float(std_mid / (std_tail_50 + 1e-9))

    # ==========================
    # NEW: last significant edge timing (สำคัญกับ pulse/late-step)
    # ==========================
    # ใช้ derivative บนสัญญาณที่ smooth เบา ๆ กัน noise
    # (ถ้ามึงไม่อยากเอา clean_medium มาตรงนี้ ก็ใช้ x ได้เลย)
    y = x
    dy = np.diff(y)
    if len(dy) > 0:
        abs_dy = np.abs(dy)

        # threshold แบบ scale-invariant:
        # - ใช้ quantile ของ abs_dy เพื่อจับ edge จริง
        # - กันเคส noise เล็กๆ
        q90 = float(np.quantile(abs_dy, 0.90))
        q95 = float(np.quantile(abs_dy, 0.95))
        thr = max(0.02, 0.35 * q95, 0.60 * q90)  # ปรับได้

        edge_idx = np.where(abs_dy > thr)[0]  # index ของ dy (edge อยู่ระหว่าง i->i+1)

        if len(edge_idx):
            last_edge_i = int(edge_idx[-1] + 1)
            last_edge_pos_ratio = float(last_edge_i / max(N - 1, 1))
            last_edge_time = float(tt[last_edge_i])
            edge_count = int(len(edge_idx))
        else:
            last_edge_i = -1
            last_edge_pos_ratio = 0.0
            last_edge_time = float(tt[0])
            edge_count = 0
    else:
        last_edge_i = -1
        last_edge_pos_ratio = 0.0
        last_edge_time = float(tt[0])
        edge_count = 0

    # activity in last X ms (normalized domain)
    if t is None or len(t) != N:
        dt_est = 1.0
    else:
        dts = np.diff(t)
        dts = dts[np.isfinite(dts) & (dts > 0)]
        dt_est = float(np.median(dts)) if len(dts) else 1.0

    last_ms = 1.0  # ดู activity ใน ~1ms สุดท้าย
    k = int(last_ms / max(dt_est, 1e-9))
    k = max(10, min(k, N - 1))

    tail = x[-(k + 1):] if N > (k + 1) else x
    dy = np.abs(np.diff(tail))
    late_activity = float(np.mean(dy)) if len(dy) else 0.0



    return {
        # เดิม
        "x_end": x_end,
        "std_all": std_all,
        "ringing_energy": ringing_energy,
        "max_rolling_std_half": max_rolling_std_half,
        "std_tail_50": std_tail_50,
        "max_dev_tail_50": max_dev_tail_50,
        "mid_to_tail_ratio": max_rolling_std_half / (std_tail_50 + 1e-9),
        "crossing_rate": crossing_rate,
        "drift_score": drift_score,
        "max_slope": max_slope,

        # เพิ่มเพื่อช่วย fall / damp / ringing
        "start_med": start_med,
        "end_med": end_med,
        "delta_start_end": delta_start_end,
        "direction": direction,               # +1 rise, -1 fall
        "min_slope": min_slope,
        "global_slope": global_slope,
        "tail200_slope": tail200_slope,
        "tail100_slope": tail100_slope,
        "tail200_slope_norm": tail200_slope_norm,
        "tail100_slope_norm": tail100_slope_norm,
        "tail_monotonicity": tail_monotonicity,
        "tail_abs_slope_mean": tail_abs_slope_mean,
        "tail_signed_slope_mean": tail_signed_slope_mean,
        "overshoot_norm": overshoot_norm,
        "undershoot_norm": undershoot_norm,
        "last_cross_time": last_cross_time,
        "last_cross_tail_ratio": last_cross_tail_ratio,
        "envelope_ratio": envelope_ratio,
        "std_mid": std_mid,
        "min_pos_ratio": min_pos_ratio,
        "min_time": min_time,
        "min_to_end_norm": min_to_end_norm,
        "rebound_norm": rebound_norm,
        "post_min_slope_norm": post_min_slope_norm,
        "tail_creep_norm": tail_creep_norm,

        "edge_count": edge_count,
        "last_edge_time": last_edge_time,
        "last_edge_pos_ratio": last_edge_pos_ratio,
        "late_activity": late_activity,
    }

# Normalize version
# def compute_features_from_row(x: np.ndarray, t: np.ndarray | None = None) -> dict:
#     N = len(x)
#     if N < 60:
#         return {}

#     x = np.asarray(x, float)
#     if t is None or len(t) != N:
#         tt = np.arange(N, dtype=float)
#         dt_est = 1.0
#     else:
#         tt = np.asarray(t, float)
#         dts = np.diff(tt)
#         dts = dts[np.isfinite(dts) & (dts > 0)]
#         dt_est = float(np.median(dts)) if len(dts) else 1.0

#     # robust start/end
#     head_n = min(30, N)
#     tail_n = min(50, N)
#     start_med = float(np.median(x[:head_n]))
#     end_med   = float(np.median(x[-tail_n:]))

#     delta_start_end = float(end_med - start_med)
#     direction = 1.0 if delta_start_end >= 0 else -1.0
#     #step_amp = float(np.abs(delta_start_end)) + 1e-9
#     step_amp_raw = float(np.abs(delta_start_end))
#     # floor จาก span ของสัญญาณ (ใน normalized domain ก็ยังใช้ได้)
#     p95 = float(np.percentile(x, 95))
#     p05 = float(np.percentile(x, 5))
#     span = max(p95 - p05, 1e-6)

#     # step_amp_safe: กันเคส no-step / flat
#     step_amp = max(step_amp_raw, 0.12 * span, 1e-3)

#     dx = np.diff(x)
#     max_slope = float(np.max(dx)) if len(dx) else 0.0
#     min_slope = float(np.min(dx)) if len(dx) else 0.0

#     # tail stats
#     x_tail = x[-tail_n:]
#     std_tail_50 = float(np.std(x_tail))
#     max_dev_tail_50 = float(np.max(np.abs(x_tail - end_med)))

#     # tail slope norm (fit on last 200 samples)
#     def _fit_slope(y: np.ndarray, tx: np.ndarray) -> float:
#         if len(y) < 5:
#             return 0.0
#         try:
#             return float(np.polyfit(tx, y, 1)[0])
#         except Exception:
#             return 0.0

#     k200 = min(N, 200)
#     tail200_slope = _fit_slope(x[-k200:], tt[-k200:])
#     tail200_slope_norm = float(tail200_slope / step_amp)

#     # tail creep norm: median(last 100) - median(prev 100)
#     if N >= 220:
#         mid_tail_med  = float(np.median(x[-200:-100]))
#         last_tail_med = float(np.median(x[-100:]))
#     else:
#         a = max(N - 120, 0)
#         b = max(N - 60, 0)
#         c = max(N - 60, 0)
#         mid_tail_med  = float(np.median(x[a:b])) if b > a else float(np.median(x))
#         last_tail_med = float(np.median(x[c:])) if N > 0 else 0.0

#     tail_creep = float(last_tail_med - mid_tail_med)
#     tail_creep_norm = float(tail_creep / step_amp)

#     # overshoot/undershoot vs end_med (normalized)
#     overshoot = float(np.max(x - end_med))
#     undershoot = float(np.max(end_med - x))
#     overshoot_norm = float(overshoot / step_amp)
#     undershoot_norm = float(undershoot / step_amp)

#     # last significant edge timing (late-step / pulse)
#     if len(dx) > 0:
#         abs_dy = np.abs(dx)
#         q90 = float(np.quantile(abs_dy, 0.90))
#         q95 = float(np.quantile(abs_dy, 0.95))
#         thr = max(0.02, 0.35 * q95, 0.60 * q90)
#         edge_idx = np.where(abs_dy > thr)[0]
#         edge_count = int(len(edge_idx))
#         last_edge_pos_ratio = float((edge_idx[-1] + 1) / max(N - 1, 1)) if edge_count else 0.0
#     else:
#         edge_count = 0
#         last_edge_pos_ratio = 0.0

#     # late activity ~1ms (mean |dy| in last window)
#     last_ms = 1.0
#     k = int(last_ms / max(dt_est, 1e-9))
#     k = max(10, min(k, N - 1))
#     tail_seg = x[-(k + 1):] if N > (k + 1) else x
#     late_activity = float(np.mean(np.abs(np.diff(tail_seg)))) if len(tail_seg) > 1 else 0.0

#     return {
#         "direction": direction,
#         "delta_start_end": delta_start_end,

#         "std_tail_50": std_tail_50,
#         "max_dev_tail_50": max_dev_tail_50,

#         "max_slope": max_slope,
#         "min_slope": min_slope,

#         "tail200_slope_norm": tail200_slope_norm,
#         "tail_creep_norm": tail_creep_norm,

#         "overshoot_norm": overshoot_norm,
#         "undershoot_norm": undershoot_norm,

#         "edge_count": edge_count,
#         "last_edge_pos_ratio": last_edge_pos_ratio,
#         "late_activity": late_activity,
#     }

# def t_enter_band_ms_allow_spike(
#     t_ms: np.ndarray,
#     x: np.ndarray,
#     lo: np.ndarray,
#     hi: np.ndarray,
#     *,
#     stable_k: int = 60,   # หน้าต่างความนิ่ง K จุด
#     allow_m: int = 1,     # อนุญาตหลุด band ได้ M จุดใน window
# ) -> float:
#     """
#     หาเวลาแรกที่สัญญาณ "อยู่ใน band เกือบตลอด" ภายในหน้าต่าง stable_k จุด
#     โดยอนุญาตให้หลุด band ได้ allow_m จุด (กัน spike 1-2 จุด)
#     """
#     n = len(x)
#     if n < stable_k or n < 5:
#         return float(t_ms[-1]) if n else 0.0

#     ok = ((x >= lo) & (x <= hi)).astype(np.int32)
#     bad = 1 - ok  # 1 = หลุด band

#     bad_cnt = np.convolve(bad, np.ones(stable_k, dtype=np.int32), mode="valid")
#     idx = np.where(bad_cnt <= allow_m)[0]
#     if len(idx) == 0:
#         return float(t_ms[-1])

#     return float(t_ms[int(idx[0])])


def ringing_damping_features(x: np.ndarray, dt_ms: float = 0.01) -> dict:
    """
    ฟีเจอร์จับ ringing/damped sine โดยไม่แตะ label:
    - peak_count, mean_peak_spacing_ms -> บอกความถี่โดยคร่าว
    - peak_decay_ratio, damping_slope -> บอกการ decay
    - overshoot_ratio -> บอก overshoot หลัง step
    """
    N = len(x)
    if N < 80:
        return {}

    # remove DC / baseline
    xc = x - np.median(x[-min(N, 200):])

    # ใช้สัญญาณช่วงท้ายเป็นหลัก (ringing มักอยู่หลัง event)
    tail = xc[-min(N, 400):]
    tN = len(tail)
    if tN < 80:
        return {}

    # --- 1) peak count / spacing (ประมาณ freq) ---
    # peak = จุดที่ diff เปลี่ยน sign จาก + เป็น -
    d = np.diff(tail)
    s = np.sign(d)
    # peak indices in tail (exclude edges)
    peak_idx = np.where((s[:-1] > 0) & (s[1:] < 0))[0] + 1

    peak_count = int(len(peak_idx))

    if peak_count >= 2:
        spacings = np.diff(peak_idx) * dt_ms
        mean_peak_spacing_ms = float(np.mean(spacings))
        std_peak_spacing_ms = float(np.std(spacings))
    else:
        mean_peak_spacing_ms = 0.0
        std_peak_spacing_ms = 0.0

    # --- 2) envelope decay ---
    # ใช้ขนาดของ peak (abs) ดูว่ามันลดลงเร็วแค่ไหน
    if peak_count >= 3:
        peak_vals = np.abs(tail[peak_idx])
        # เอา peak แรก ๆ กับท้าย ๆ เทียบอัตราการลดลง
        k = min(5, len(peak_vals))
        first_mean = float(np.mean(peak_vals[:k]))
        last_mean = float(np.mean(peak_vals[-k:]))
        peak_decay_ratio = last_mean / (first_mean + 1e-12)

        # damping slope: fit log(peak) ~ a*i + b (a < 0 ยิ่ง negative ยิ่ง decay เร็ว)
        y = np.log(peak_vals + 1e-12)
        xidx = np.arange(len(y), dtype=float)
        a = float(np.polyfit(xidx, y, 1)[0])  # slope
        damping_slope = a
    else:
        peak_decay_ratio = 1.0
        damping_slope = 0.0

    # --- 3) overshoot ratio (หลัง step) ---
    # วัด overshoot เทียบกับ tail final
    tail_ref = float(np.median(tail[-min(50, tN):]))
    overshoot = float(np.max(tail) - tail_ref)
    undershoot = float(tail_ref - np.min(tail))
    denom = float(np.std(tail[-min(100, tN):]) + 1e-9)
    overshoot_z = overshoot / denom
    undershoot_z = undershoot / denom

    return {
        "ring_peak_count": peak_count,
        "ring_mean_peak_spacing_ms": mean_peak_spacing_ms,
        "ring_std_peak_spacing_ms": std_peak_spacing_ms,
        "ring_peak_decay_ratio": float(peak_decay_ratio),
        "ring_damping_slope": float(damping_slope),
        "ring_overshoot_z": float(overshoot_z),
        "ring_undershoot_z": float(undershoot_z),
    }

def tail_amplitude_features(xN: np.ndarray, dt_ms: float) -> dict:
    """
    Tail amplitude / envelope stability features (leak-safe)
    ใช้เฉพาะ tail → ไม่แตะ label / wait time
    """
    N = len(xN)
    if N < 80:
        return {
            "tail_p2p": 0.0,
            "tail_amp_cv": 1.0,
            "tail_env_slope_abs": 1.0,
        }

    # ใช้ tail ~2ms (หรือเท่าที่มี)
    tail_len = int(max(30, min(N, round(2.0 / max(dt_ms, 1e-6)))))
    tail = xN[-tail_len:]

    # -------------------------
    # (A) tail_p2p : peak-to-peak
    # -------------------------
    tail_p2p = float(np.ptp(tail))

    # -------------------------
    # (B) tail_amp_cv : amplitude stability
    # -------------------------
    amp = np.abs(tail)
    amp_mean = float(np.mean(amp)) + 1e-12
    amp_std  = float(np.std(amp))
    tail_amp_cv = float(amp_std / amp_mean)

    # -------------------------
    # (C) tail_env_slope_abs : envelope drift / decay
    # -------------------------
    # envelope = moving RMS (window ~0.2ms)
    win = int(max(10, round(0.2 / max(dt_ms, 1e-6))))
    if len(amp) > win:
        env = np.sqrt(
            np.convolve(amp**2, np.ones(win)/win, mode="valid")
        )
        t = np.arange(len(env)) * dt_ms
        try:
            slope = float(np.polyfit(t, env, 1)[0])
        except Exception:
            slope = 0.0
    else:
        slope = 0.0

    tail_env_slope_abs = float(abs(slope))

    return {
        "tail_p2p": tail_p2p,
        "tail_amp_cv": tail_amp_cv,
        "tail_env_slope_abs": tail_env_slope_abs,
    }

def tail_energy_decay_features(xN: np.ndarray, dt_ms: float) -> dict:
    """
    แยก pure sine vs damped sine โดยดู "envelope/energy decay" ใน tail
    - pure sine: energy_ratio ~ 1, env_slope ~ 0
    - damped:    energy_ratio < 1, env_slope < 0 (มักติดลบ)
    """
    N = len(xN)
    if N < 120:
        return {
            "tail_energy_ratio_2nd_over_1st": 1.0,
            "tail_env_slope_signed": 0.0,
            "tail_env_slope_norm": 0.0,
        }

    # ใช้ tail ยาวพอ (เช่น 2ms)
    tail_len = int(max(60, min(N, round(2.0 / max(dt_ms, 1e-6)))))
    tail = xN[-tail_len:]

    # --- (1) energy ratio ---
    half = len(tail) // 2
    a = tail[:half]
    b = tail[half:]

    rms_a = float(np.sqrt(np.mean(a * a)) + 1e-12)
    rms_b = float(np.sqrt(np.mean(b * b)) + 1e-12)
    energy_ratio = float(rms_b / rms_a)  # pure sine ~1, damped <1

    # --- (2) envelope slope (signed) ---
    amp = np.abs(tail)
    win = int(max(10, round(0.2 / max(dt_ms, 1e-6))))  # ~0.2ms
    if len(amp) > win:
        env = np.sqrt(np.convolve(amp**2, np.ones(win)/win, mode="valid"))
        t = np.arange(len(env), dtype=float) * dt_ms
        try:
            slope = float(np.polyfit(t, env, 1)[0])  # signed
        except Exception:
            slope = 0.0
    else:
        slope = 0.0

    # normalize slope ให้ scale-invariant
    env_scale = float(np.mean(env) + 1e-12) if "env" in locals() else float(np.mean(amp) + 1e-12)
    slope_norm = float(slope / env_scale)

    return {
        "tail_energy_ratio_2nd_over_1st": energy_ratio,
        "tail_env_slope_signed": float(slope),
        "tail_env_slope_norm": float(slope_norm),
    }

def _build_ml_features(xN, times, dt_ms=0.01):
    feats = {}
    feats.update(compute_features_from_row(xN, times))
    feats.update(ringing_damping_features(xN, dt_ms))
    feats.update(periodic_features(xN, dt_ms))   

    # NEW periodic/sine helpers (leak-safe)
    feats.update(autocorr_peak_features(xN, dt_ms))
    feats.update(fft_periodic_features(xN, dt_ms))
    feats.update(tail_zcr_features(xN))
    feats.update(tail_envelope_flatness(xN))

    feats.update(tail_amplitude_features(xN, dt_ms))
    feats.update(tail_energy_decay_features(xN, dt_ms))

    return feats

FAST_MS         = 0.1
MIN_OUT_MS      = 0.1
DEFAULT_WAIT_MS = 1.0
TAIL_MS_HEAVY   = 2.0
TAIL_MS_LIGHT   = 1.0

def extract_label_and_features(group: pd.DataFrame, *, mode: str) -> pd.Series:
    wave_id = int(group["wave_id"].iloc[0])
    times   = group["time_ms"].to_numpy(float)
    values  = group["value"].to_numpy(float)

    # --- compute features first (always) ---
    xN, meta = normalize_wave(values, times)   # เอา meta มาด้วย
    # --- optional: use sd as noise floor ---
    sd_wave = np.nan

    if "sd" in group.columns:
        try:
            sd_wave = float(np.nanmedian(group["sd"].to_numpy(float)))
        except Exception:
            sd_wave = np.nan

    if np.isfinite(sd_wave):
        meta["end_noise"] = float(max(float(meta.get("end_noise", 0.0)), sd_wave))
        
    dt_ms = _infer_dt_ms(times, default=0.01)
    feats = _build_ml_features(xN, times, dt_ms=dt_ms)
    CLIP_FEATS = ["overshoot_norm","undershoot_norm","rebound_norm","min_to_end_norm","post_min_slope_norm","tail_creep_norm"]
    for k in CLIP_FEATS:
        if k in feats:
            feats[k] = float(np.clip(feats[k], -12.0, 12.0))

    #feats = _build_ml_features(xN, times)
    for k, v in list(feats.items()):
        if not np.isfinite(v):
            feats[k] = 0.0

    # add normalize meta as features (ช่วย zero_clf แยก sine/step)
    step_raw = float(meta.get("step_raw", 0.0))
    span     = float(meta.get("global_span", 0.0))
    denom    = float(meta.get("denom_used", 0.0))
    noise    = float(meta.get("end_noise", 0.0))

    feats.update({
        "meta_global_span": span,
        "meta_end_noise": noise,
        "meta_denom_used": denom,
        "meta_step_raw": step_raw,
        "meta_abs_step": abs(step_raw),
        "meta_step_to_span": abs(step_raw) / (span + 1e-9),
        "meta_noise_to_span": noise / (span + 1e-9),
        "meta_abs_step_to_noise": abs(step_raw) / (noise + 1e-9),
    })
    # ---- in-band settle features (raw band) ----
    # t_enter = np.nan
    # bw = np.nan
    # inband_early = np.nan

    # if ("low_limit" in group.columns) and ("high_limit" in group.columns):
    #     lo = group["low_limit"].to_numpy(float)
    #     hi = group["high_limit"].to_numpy(float)

    #     # กัน NaN/inf
    #     good = np.isfinite(lo) & np.isfinite(hi)
    #     if np.any(good):
    #         # เติมค่าขาดด้วย median เฉพาะ wave
    #         lo2 = lo.copy()
    #         hi2 = hi.copy()
    #         lo2[~good] = float(np.nanmedian(lo2[good]))
    #         hi2[~good] = float(np.nanmedian(hi2[good]))

    #         t_enter = t_enter_band_ms_allow_spike(
    #             times, values, lo2, hi2,
    #             stable_k=60,
    #             allow_m=1,
    #         )
    #         bw = float(np.nanmedian(hi2 - lo2))

    #         # extra: สัดส่วน "อยู่ใน band" ในช่วงต้น 0.1ms (ช่วย sine-fast มาก)
    #         early_mask = times <= 0.10
    #         if np.any(early_mask):
    #             inband_early = float(np.mean((values[early_mask] >= lo2[early_mask]) &
    #                                         (values[early_mask] <= hi2[early_mask])))
    #         else:
    #             inband_early = 0.0
    #     else:
    #         t_enter = float(times[-1])
    #         bw = 0.0
    #         inband_early = 0.0
    # else:
    #     # ไม่มี low/high ก็ปล่อย NaN หรือ set เป็นท้ายคลื่น
    #     t_enter = float(times[-1]) if len(times) else 0.0
    #     bw = 0.0
    #     inband_early = 0.0

    # feats.update({
    #     "t_enter_band_ms": float(t_enter),
    #     "band_width": float(bw),
    #     "inband_ratio_early_0p1ms": float(inband_early),
    # })


    # label default
    wait_time_ms = np.nan
    label_reason = "unset"

    if len(values) < 20:
        label_reason = "too_short"

    # --- label only in train ---
    if mode == "train":
        if "true_is_zero" in group.columns and pd.notna(group["true_is_zero"].iloc[0]) and int(group["true_is_zero"].iloc[0]) == 1:
            wait_time_ms = FAST_MS
            label_reason = "gt_is_zero"
        elif "true_wait_time_ms" in group.columns and pd.notna(group["true_wait_time_ms"].iloc[0]):
            w = float(group["true_wait_time_ms"].iloc[0])
            wait_time_ms = float(np.clip(w, MIN_OUT_MS, float(np.nanmax(times))))
            label_reason = "gt_true_wait_time_ms"
        elif "wait_time_ms" in group.columns and pd.notna(group["wait_time_ms"].iloc[0]):
            w = float(group["wait_time_ms"].iloc[0])
            wait_time_ms = float(np.clip(w, MIN_OUT_MS, float(np.nanmax(times))))
            label_reason = "gt_wait_time_ms"
        else:
            raise RuntimeError(f"[TRAIN] missing label for wave_id={wave_id} (need true_wait_time_ms or wait_time_ms or true_is_zero)")

    return pd.Series({
        "wave_id": wave_id,
        "wait_time_ms": wait_time_ms,
        **feats,
        "dbg_label_reason": label_reason,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "pred"], required=True)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)

    ap.add_argument("--id-col", default="wave_id")
    ap.add_argument("--sample-col", default="sample") # ยังต้องใช้เพื่อ sort
    ap.add_argument("--time-col", default="time_ms")
    ap.add_argument("--value-col", default="value")

    ap.add_argument("--window-ms", type=float, default=10.0, help="ใช้ข้อมูลแค่ 0..window-ms (default 10ms)")

    
    # Meta columns
    ap.add_argument("--sd-col", default="sd")
    ap.add_argument("--type-col", default="type")
    ap.add_argument("--low-col", default="low_limit")
    ap.add_argument("--high-col", default="high_limit")

    
    # args.n_samples ไม่ต้องใช้แล้วเพราะเราไม่ทำ Wide
    
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {in_path}")
    df = pd.read_csv(in_path)

    # 1. Normalize column names
    rename_map = {
        args.id_col: "wave_id",
        args.sample_col: "sample",
        args.time_col: "time_ms",
        args.value_col: "value",
    }
    for k, v in rename_map.items():
        if k not in df.columns:
            raise KeyError(f"Missing required column: {k}")
    df = df.rename(columns=rename_map)

    # 2. Rename optional meta columns
    optional_map = {
        args.sd_col: "sd",
        args.type_col: "type",
        args.low_col: "low_limit",
        args.high_col: "high_limit",
    }
    for src, dst in optional_map.items():
        if src in df.columns:
            df = df.rename(columns={src: dst})
        else:
            df[dst] = np.nan

    # 3. Sort (สำคัญมากสำหรับการคำนวณ Feature)
    df["sample"] = df["sample"].astype(int)
    df = df.sort_values(["wave_id", "sample"])

    # ✅ 3.1 Cut window (0..window-ms)
    win = float(args.window_ms)
    df = df[df["time_ms"].astype(float) <= win].copy()

    # (optional) กันบาง wave เหลือจุดน้อยเกิน
    min_pts = 50
    valid_ids = df.groupby("wave_id")["sample"].count()
    keep_ids = valid_ids[valid_ids >= min_pts].index
    df = df[df["wave_id"].isin(keep_ids)]


    # --- [CHANGE] ตัดส่วน to_wide (Raw Data) ทิ้งไปเลย ---
    print("Skipping Raw Data (Wide format) generation to prevent overfitting...")

    # 4. Extract Engineered Features
    print("Extracting Engineered Features...")
    feat_df = df.groupby("wave_id", group_keys=False).apply(lambda g: extract_label_and_features(g, mode=args.mode)).reset_index(drop=True)
    if args.mode == "train":
        miss = int(feat_df["wait_time_ms"].isna().sum())
        if miss:
            raise RuntimeError(f"[TRAIN] wait_time_ms missing in {miss} rows after feature extraction")

    # ---------------- Leak Control ----------------
    DROP_PREFIX = ("dbg_", "logic_", "need_more")
    DROP_COLS = ["sd", "type", "low_limit", "high_limit"]

    drop_cols = [c for c in feat_df.columns if c.startswith(DROP_PREFIX)]
    feat_df = feat_df.drop(columns=drop_cols + DROP_COLS, errors="ignore")

    # ---------- Output assemble ----------
    if args.mode == "pred":
        # (optional) merge meta for debugging / traceability
        meta_cols = ["wave_id"] + [c for c in ["sd", "type", "low_limit", "high_limit"] if c in df.columns]
        meta_df = df[meta_cols].groupby("wave_id", as_index=False).first()
        out_df = meta_df.merge(feat_df, on="wave_id", how="left")

        # ถ้าไม่อยากให้ pred output มี meta จริง ๆ ให้เปิด 4 บรรทัดนี้
        # out_df = out_df.drop(columns=["sd","type","low_limit","high_limit"], errors="ignore")
    else:
        out_df = feat_df.copy()

    # Pred mode: ห้ามมี wait_time_ms หลุดไป
    if args.mode == "pred" and "wait_time_ms" in out_df.columns:
        out_df = out_df.drop(columns=["wait_time_ms"])
        print("Dropped 'wait_time_ms' for inference mode.")

    # drop label_reason (จริง ๆ ของมึงชื่อ dbg_label_reason)
    out_df = out_df.drop(columns=["label_reason", "dbg_label_reason"], errors="ignore")


    # Save
    out_df.to_csv(out_path, index=False)
    print(f"✅ Saved Features Only: {out_path}")
    print(f"   Rows: {len(out_df)}, Columns: {len(out_df.columns)}")
    print(f"   (No v0000..vXXXX columns included)")

if __name__ == "__main__":
    main()
