# scripts/extract_features_refactor_v25.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# Config (พยายามให้ "น้อย" และ "เข้าใจง่าย")
# ============================================================
CFG = {
    # window cut
    "WINDOW_MS": 10.0,
    "MIN_PTS": 50,

    # normalization / robust stats
    "START_MS": 0.4,          # ใช้ช่วงต้นกี่ ms หา start_ref
    "TAIL_MS": 2.0,           # ใช้ช่วงท้ายกี่ ms หา tail
    "TAIL_WIN_MS": 0.6,       # sliding window ภายใน tail เพื่อหา end_ref ที่นิ่งสุด
    "CLIP_LIMIT": 12.0,       # clip features ที่เป็น ratio/norm ที่เสี่ยง explode

    # periodic/fft
    "FFT_TAIL_MAX_N": 400,
    "FFT_MIN_N": 120,
    "LOWFREQ_HZ": 50.0,

    # ringing peaks
    "RING_TAIL_N": 400,

    # fast label policy
    "FAST_MS": 0.1,
    "MIN_OUT_MS": 0.1,
}

# ฟีเจอร์ที่อยาก clip (จุดเดียวจบ)
CLIP_FEATS = [
    "shape_overshoot_norm",
    "shape_undershoot_norm",
    "shape_rebound_norm",
    "shape_min_to_end_norm",
    "shape_post_min_slope_norm",
    "tail_creep_norm",
]


# ============================================================
# Helpers: robust utilities
# ============================================================
EPS = 1e-12

def _infer_dt_ms(times: np.ndarray, default=0.01) -> float:
    t = np.asarray(times, float)
    if t.size < 3:
        return float(default)
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else float(default)

def _safe_percentile(x: np.ndarray, q: float) -> float:
    xf = x[np.isfinite(x)]
    if xf.size == 0:
        return 0.0
    return float(np.percentile(xf, q))

def _mad(x: np.ndarray, med: float | None = None) -> float:
    xf = x[np.isfinite(x)]
    if xf.size == 0:
        return 0.0
    m = float(np.median(xf)) if med is None else float(med)
    return float(np.median(np.abs(xf - m)))

def _robust_std_from_mad(x: np.ndarray) -> float:
    # MAD -> robust std
    return 1.4826 * _mad(x)

def _clip_feats(feats: dict) -> dict:
    lim = float(CFG["CLIP_LIMIT"])
    for k in CLIP_FEATS:
        if k in feats and np.isfinite(feats[k]):
            feats[k] = float(np.clip(feats[k], -lim, lim))
    return feats


# ============================================================
# Normalize wave (data-driven floors, minimal manual)
# ============================================================
def _best_stable_tail_level(values: np.ndarray, times: np.ndarray, *, tail_ms: float, win_ms: float) -> tuple[float, float]:
    """
    เลือก end_ref จาก tail ที่ "นิ่งสุด" (robust std ต่ำสุด)
    คืน (end_ref, end_noise_std_robust)
    """
    x = np.asarray(values, float)
    t = np.asarray(times, float)
    N = x.size
    if N < 20:
        m = float(np.median(x)) if N else 0.0
        return m, float(_robust_std_from_mad(x))

    t_end = float(t[-1])
    t0 = t_end - float(tail_ms)
    idx_tail = np.where(t >= t0)[0]
    if idx_tail.size < 20:
        idx_tail = np.arange(max(0, N - 50), N)

    xt = x[idx_tail]
    tt = t[idx_tail]

    dt = _infer_dt_ms(tt, default=0.01)
    W = int(max(10, round(win_ms / max(dt, 1e-9))))

    if xt.size <= W:
        med = float(np.median(xt))
        rstd = float(_robust_std_from_mad(xt))
        return med, rstd

    best_std = float("inf")
    best_med = float(np.median(xt[-W:]))

    for i in range(0, xt.size - W + 1):
        w = xt[i:i+W]
        med = float(np.median(w))
        rstd = float(_robust_std_from_mad(w))
        if rstd < best_std:
            best_std = rstd
            best_med = med

    return best_med, float(best_std)

def normalize_wave(values: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, dict]:
    x = np.asarray(values, float)
    t = np.asarray(times, float)
    N = x.size

    # start_ref: median ในช่วงต้น START_MS
    t0 = float(t[0])
    idx_start = np.where(t <= t0 + float(CFG["START_MS"]))[0]
    if idx_start.size < 10:
        idx_start = np.arange(min(30, N))
    start_ref = float(np.median(x[idx_start])) if idx_start.size else float(np.median(x))

    # end_ref: จาก tail ที่นิ่งสุด
    end_ref, end_noise = _best_stable_tail_level(
        x, t,
        tail_ms=float(CFG["TAIL_MS"]),
        win_ms=float(CFG["TAIL_WIN_MS"]),
    )

    # global span จาก percentile (กัน outlier)
    p95 = _safe_percentile(x, 95)
    p05 = _safe_percentile(x, 5)
    span_raw = float(max(p95 - p05, 0.0))

    step_raw = float(end_ref - start_ref)
    direction = 1.0 if step_raw >= 0 else -1.0
    step_amp = float(abs(step_raw))

    # data-driven floors ที่ผูกกับสเกลของสัญญาณ (กัน micro/nano explode)
    sig_scale = max(abs(end_ref), abs(start_ref), span_raw, 1e-15)
    noise_floor = max(sig_scale * 1e-6, 1e-12)   # 1 ppm
    span_floor  = max(sig_scale * 1e-5, 1e-12)   # 10 ppm

    global_span = float(max(span_raw, span_floor))
    end_noise = float(max(end_noise, noise_floor))

    # denom_used: step หรือ span หรือ noise (เลือกใหญ่สุดแบบปลอดภัย)
    denom_used = max(step_amp, 0.12 * global_span, 6.0 * end_noise, noise_floor)

    xN = (x - end_ref) / denom_used
    xN = np.clip(xN, -float(CFG["CLIP_LIMIT"]), float(CFG["CLIP_LIMIT"]))

    meta = {
        "meta_start_ref": float(start_ref),
        "meta_end_ref": float(end_ref),
        "meta_step_raw": float(step_raw),
        #"meta_abs_step": float(abs(step_raw)),
        "meta_direction": float(direction),
        "meta_end_noise": float(end_noise),
        "meta_global_span": float(global_span),
        "meta_denom_used": float(denom_used),
        "meta_step_to_span": float(abs(step_raw) / (global_span + EPS)),
        "meta_noise_to_span": float(end_noise / (global_span + EPS)),
        "meta_abs_step_to_noise": float(abs(step_raw) / (end_noise + EPS)),
    }
    return xN, meta


# ============================================================
# Feature groups
# ============================================================
def base_features(xN: np.ndarray, t: np.ndarray, dt_ms: float) -> dict:
    # base stats in normalized domain (scale-safe)
    dx = np.diff(xN)
    dt_s = max(float(dt_ms) * 1e-3, EPS)
    dxdt = dx / dt_s if dx.size else np.array([], float)

    return {
        #"base_n": int(xN.size),
        #"base_t_end_ms": float(t[-1]) if t.size else 0.0,
        "base_mean": float(np.mean(xN)),
        "base_std": float(np.std(xN)),
        "base_min": float(np.min(xN)),
        "base_max": float(np.max(xN)),
        "base_p2p": float(np.ptp(xN)),
        "base_energy": float(np.sum(xN * xN)),
        "base_max_slope": float(np.max(np.abs(dxdt))) if dxdt.size else 0.0,
        "base_mean_abs_slope": float(np.mean(np.abs(dxdt))) if dxdt.size else 0.0,
    }

def tail_features(xN: np.ndarray, t: np.ndarray, dt_ms: float) -> dict:
    N = xN.size
    if N < 20:
        return {
            "tail_std": 0.0,
            "tail_p2p": 0.0,
            "tail_mean_abs_slope": 0.0,
            "tail_monotonicity": 0.0,
            "tail_creep_norm": 0.0,
            "tail_last_cross_tail_ratio": 1.0,
        }

    # tail length from TAIL_MS
    tail_len = int(max(30, min(N, round(float(CFG["TAIL_MS"]) / max(dt_ms, 1e-6)))))
    tail = xN[-tail_len:]
    dx = np.diff(tail)
    dt_s = max(float(dt_ms) * 1e-3, EPS)

    # monotonicity "ตามทิศทางของ step" จะใส่ใน shape group (ต้องรู้ direction)
    # ที่นี่เก็บ slope/ความนิ่งล้วน ๆ
    tail_std = float(np.std(tail))
    tail_p2p = float(np.ptp(tail))
    tail_mean_abs_slope = float(np.mean(np.abs(dx / dt_s))) if dx.size else 0.0

    # creep: median(ท้ายสุด) - median(ก่อนท้าย)
    if tail.size >= 120:
        a = tail[: tail.size // 2]
        b = tail[tail.size // 2 :]
        creep = float(np.median(b) - np.median(a))
    else:
        creep = float(np.median(tail[-max(10, tail.size//3):]) - np.median(tail[:max(10, tail.size//3)]))
    tail_creep_norm = float(creep)  # already normalized domain

    # last crossing around tail median
    ref = float(np.median(tail))
    s = np.sign(xN - ref)
    cross_idx = np.where(np.diff(s) != 0)[0]
    if cross_idx.size:
        last_i = int(cross_idx[-1])
        last_cross_tail_ratio = float((N - 1 - last_i) / max(N, 1))
    else:
        last_cross_tail_ratio = 1.0

    return {
        "tail_std": tail_std,
        "tail_p2p": tail_p2p,
        "tail_mean_abs_slope": tail_mean_abs_slope,
        "tail_creep_norm": tail_creep_norm,
        "tail_last_cross_tail_ratio": last_cross_tail_ratio,
    }

def tail_decay_slope(xN: np.ndarray, t: np.ndarray, dt_ms: float) -> dict:
    """
    วัดอัตราการ decay ของ |error| ใน tail
    ใช้ log(|xN|) vs time
    """
    N = xN.size
    if N < 60:
        return {"tail_decay_slope": 0.0}

    # เลือก tail
    tail_len = int(
        max(
            30,
            min(N, round(float(CFG["TAIL_MS"]) / max(dt_ms, 1e-6)))
        )
    )

    tail_x = xN[-tail_len:]
    tail_t = t[-tail_len:]

    # error = ระยะจาก end_ref (xN ถูก center แล้ว)
    err = np.abs(tail_x)

    # กัน log(0)
    err = np.clip(err, 1e-6, None)

    # log(error)
    y = np.log(err)

    # time (ms)
    tt = tail_t.astype(float)

    if y.size < 10:
        return {"tail_decay_slope": 0.0}

    # fit เส้นตรง y = a*t + b
    try:
        a = float(np.polyfit(tt, y, 1)[0])
    except Exception:
        a = 0.0

    return {
        "tail_decay_slope": float(a)  # หน่วย: log(norm) / ms
    }

def edge_stats(xN: np.ndarray, dt_ms: float, *, tail_ms: float) -> dict:
    N = xN.size
    if N < 5:
        return {
            "edge_thr": 0.0,
            "edge_count": 0,
            "edge_rate": 0.0,
            "first_edge_pos_ratio": 0.0,
            "last_edge_pos_ratio": 0.0,
            "edge_span_ratio": 0.0,
            "edge_density": 0.0,
            "tail_edge_count": 0,
            "tail_edge_rate": 0.0,
            "edge_max_ratio": 1.0,
        }

    dx = np.diff(xN)
    abs_dx = np.abs(dx)

    med = float(np.median(abs_dx))
    mad = float(_mad(abs_dx, med=med)) + EPS
    q95 = float(np.quantile(abs_dx, 0.95))

    thr = max(med + 6.0 * 1.4826 * mad, 0.25 * q95)

    edge_idx = np.where(abs_dx > thr)[0]
    edge_count = int(edge_idx.size)
    edge_rate = float(edge_count / max(abs_dx.size, 1))

    if edge_count:
        first_i = int(edge_idx[0])
        last_i  = int(edge_idx[-1])
        first_edge_pos_ratio = float((first_i + 1) / max(N - 1, 1))
        last_edge_pos_ratio  = float((last_i + 1) / max(N - 1, 1))
        edge_span_ratio      = float((last_i - first_i) / max(N - 1, 1))
        edge_density         = float(edge_count / max(last_i - first_i + 1, 1))
    else:
        first_edge_pos_ratio = 0.0
        last_edge_pos_ratio  = 0.0
        edge_span_ratio      = 0.0
        edge_density         = 0.0

    # tail edges (ใช้ tail_ms)
    tail_len = int(max(30, min(N, round(tail_ms / max(dt_ms, 1e-6)))))
    tail_start = max(0, abs_dx.size - tail_len)
    tail_edge_count = int(np.sum(edge_idx >= tail_start))
    tail_edge_rate  = float(tail_edge_count / max(tail_len, 1))

    edge_max_ratio = float(np.max(abs_dx) / (q95 + EPS)) if abs_dx.size else 1.0

    return {
        "edge_thr": float(thr),
        #"edge_count": edge_count,
        "edge_rate": edge_rate,
        "first_edge_pos_ratio": first_edge_pos_ratio,
        "last_edge_pos_ratio": last_edge_pos_ratio,
        #"edge_span_ratio": edge_span_ratio,     # (= glitch_span_ratio)
        "edge_density": edge_density,           # (= glitch_density)
        #"tail_edge_count": tail_edge_count,
        "tail_edge_rate": tail_edge_rate,
        "edge_max_ratio": edge_max_ratio,
    }

def glitch_phase_from_edges_dd(
    first_edge_pos_ratio: float,
    edge_span_ratio: float,
    tail_edge_rate: float,
    edge_count: int,
    thr: dict,
) -> int:
    # 0 = clean/no edge
    if edge_count <= 0:
        return 0

    EARLY_MAX     = float(thr.get("EARLY_MAX", 0.25))
    LATE_MIN      = float(thr.get("LATE_MIN", 0.60))
    PERSIST_MIN   = float(thr.get("PERSIST_MIN", 0.50))
    TAIL_RATE_MIN = float(thr.get("TAIL_RATE_MIN", 0.05))
    EDGES_MIN     = int(thr.get("EDGES_MIN", 3))

    # 1) early-only: เริ่มเร็ว + span สั้น(คือไม่ persist) + tail เงียบ + มี edge พอสมควร
    if (first_edge_pos_ratio <= EARLY_MAX) and (edge_span_ratio < PERSIST_MIN) and (tail_edge_rate <= TAIL_RATE_MIN) and (edge_count >= EDGES_MIN):
        return 1

    # 2) persistent: span ยาว
    if edge_span_ratio >= PERSIST_MIN:
        return 2

    # 3) tail-active หรือเริ่มช้า
    if (tail_edge_rate >= TAIL_RATE_MIN) or (first_edge_pos_ratio >= LATE_MIN):
        return 3

    return 2



def shape_features_clean(xN: np.ndarray, t: np.ndarray, dt_ms: float, meta: dict) -> dict:
    """
    Clean shape features ONLY (no edge/glitch stats here)
    - global shape: delta/overshoot/undershoot/min/rebound/slope
    - tail monotonicity (aligned with step direction)
    - late_activity (last ~1ms activity)
    """
    N = xN.size
    if N < 30:
        return {}

    direction = float(meta.get("meta_direction", 1.0))

    # -----------------------
    # Global shape (coarse)
    # -----------------------
    head_n = min(30, N)
    tail_n = min(50, N)

    start_med = float(np.median(xN[:head_n]))
    end_med   = float(np.median(xN[-tail_n:]))

    delta_start_end = float(end_med - start_med)

    overshoot  = float(np.max(xN - end_med))
    undershoot = float(np.max(end_med - xN))

    min_i = int(np.argmin(xN))
    min_pos_ratio = float(min_i / max(N - 1, 1))
    min_to_end_norm = float(end_med - xN[min_i])

    post = xN[min_i:] if min_i < N - 2 else xN[-2:]
    rebound_norm = float(np.max(post) - xN[min_i])

    # post-min slope (normalized / ms)
    k = min(200, post.size)
    if k >= 5:
        try:
            # t is in ms already → slope is norm/ms
            m = float(np.polyfit(t[min_i:min_i+k].astype(float), post[:k], 1)[0])
        except Exception:
            m = 0.0
    else:
        m = 0.0
    post_min_slope_norm = float(m)

    # -----------------------
    # Tail monotonicity (aligned with step direction)
    # -----------------------
    tail_len = int(max(30, min(N, round(float(CFG["TAIL_MS"]) / max(dt_ms, 1e-6)))))
    tail = xN[-tail_len:]
    tail_dx = np.diff(tail)
    tail_monotonicity = (
        float((np.sign(tail_dx) == np.sign(direction)).mean())
        if tail_dx.size else 0.0
    )

    # -----------------------
    # Late activity (last ~1ms)
    # -----------------------
    k_late = int(max(10, min(N - 1, round(1.0 / max(dt_ms, 1e-9)))))
    seg = xN[-(k_late + 1):] if N > k_late + 1 else xN
    late_activity = float(np.mean(np.abs(np.diff(seg)))) if seg.size > 1 else 0.0

    return {
        "shape_delta_start_end": delta_start_end,
        "shape_overshoot_norm": overshoot,
        #"shape_undershoot_norm": undershoot,
        "shape_min_pos_ratio": min_pos_ratio,
        "shape_min_to_end_norm": min_to_end_norm,
        "shape_rebound_norm": rebound_norm,
        "shape_post_min_slope_norm": post_min_slope_norm,

        "tail_monotonicity": tail_monotonicity,
        #"late_activity": late_activity,
    }

def early_quiet_features(xN: np.ndarray, t: np.ndarray, dt_ms: float) -> dict:
    N = xN.size
    if N < 50:
        return {"quiet_after_head_ratio": 0.0, "first_quiet_pos_ratio": 1.0, "post_head_std": 0.0}

    # เลือก "หัว" ที่จะตัดออก (กัน step ช่วงแรกหลอก)
    HEAD_MS = 0.8
    idx = np.where(t >= (float(t[0]) + HEAD_MS))[0]
    if idx.size < 10:
        idx = np.arange(N//2, N)

    seg = xN[idx]
    dx = np.abs(np.diff(seg))
    if dx.size < 5:
        return {"quiet_after_head_ratio": 0.0, "first_quiet_pos_ratio": 1.0, "post_head_std": float(np.std(seg))}

    # threshold ของ "นิ่ง" แบบ robust (data-driven)
    med = float(np.median(dx))
    mad = float(_mad(dx, med=med)) + EPS
    quiet_thr = med + 2.5 * 1.4826 * mad

    quiet_mask = dx <= quiet_thr
    quiet_after_head_ratio = float(np.mean(quiet_mask))

    # หา first index ที่นิ่งต่อเนื่อง W จุด
    W = int(max(5, round(0.3 / max(dt_ms, 1e-6))))  # 0.3ms worth of samples
    first_quiet = 1.0
    if quiet_mask.size >= W:
        # sliding window: all quiet in window
        for k in range(0, quiet_mask.size - W + 1):
            if np.all(quiet_mask[k:k+W]):
                # convert back to global position ratio
                global_i = int(idx[0] + k)
                first_quiet = float(global_i / max(N-1, 1))
                break

    post_head_std = float(np.std(seg))

    return {
        "quiet_after_head_ratio": quiet_after_head_ratio,
        "first_quiet_pos_ratio": first_quiet,
        "post_head_std": post_head_std,
    }

# def shape_features(xN: np.ndarray, t: np.ndarray, dt_ms: float, meta: dict) -> dict:
#     """
#     Clean shape + edge + glitch + late features
#     - dx computed ONCE
#     - no duplicated semantics
#     """
#     N = xN.size
#     if N < 30:
#         return {}

#     EPS = 1e-12
#     direction = float(meta.get("meta_direction", 1.0))

#     # ===============================
#     # 1) Global shape (coarse)
#     # ===============================
#     head_n = min(30, N)
#     tail_n = min(50, N)

#     start_med = float(np.median(xN[:head_n]))
#     end_med   = float(np.median(xN[-tail_n:]))

#     delta_start_end = float(end_med - start_med)

#     overshoot  = float(np.max(xN - end_med))
#     undershoot = float(np.max(end_med - xN))

#     min_i = int(np.argmin(xN))
#     min_pos_ratio = float(min_i / max(N - 1, 1))
#     min_to_end_norm = float(end_med - xN[min_i])

#     post = xN[min_i:] if min_i < N - 2 else xN[-2:]
#     rebound_norm = float(np.max(post) - xN[min_i])

#     # post-min slope (normalized / ms)
#     k = min(200, post.size)
#     if k >= 5:
#         try:
#             m = float(np.polyfit(
#                 t[min_i:min_i+k], post[:k], 1
#             )[0])
#         except Exception:
#             m = 0.0
#     else:
#         m = 0.0

#     post_min_slope_norm = float(m)

#     # ===============================
#     # 2) dx + robust edge detection
#     # ===============================
#     dx = np.diff(xN)
#     abs_dx = np.abs(dx)

#     if abs_dx.size:
#         med = float(np.median(abs_dx))
#         mad = float(_mad(abs_dx, med=med)) + EPS
#         q95 = float(np.quantile(abs_dx, 0.95))

#         thr = max(
#             med + 6.0 * 1.4826 * mad,
#             0.25 * q95,
#         )

#         edge_idx = np.where(abs_dx > thr)[0]
#         edge_count = int(edge_idx.size)
#         edge_rate = float(edge_count / max(abs_dx.size, 1))

#         last_edge_pos_ratio = (
#             float((edge_idx[-1] + 1) / max(N - 1, 1))
#             if edge_count else 0.0
#         )

#         edge_max_ratio = float(np.max(abs_dx) / (q95 + EPS))
#     else:
#         edge_count = 0
#         edge_rate = 0.0
#         last_edge_pos_ratio = 0.0
#         edge_max_ratio = 1.0
#         edge_idx = []

#     # ===============================
#     # 3) Glitch persistence (NEW)
#     # ===============================
#     if edge_count >= 2:
#         first_i = int(edge_idx[0])
#         last_i  = int(edge_idx[-1])

#         glitch_span_ratio = float(
#             (last_i - first_i) / max(N - 1, 1)
#         )

#         glitch_density = float(
#             edge_count / max(last_i - first_i + 1, 1)
#         )
#     else:
#         glitch_span_ratio = 0.0
#         glitch_density = 0.0

#     # ===============================
#     # 4) Late activity (tail noise)
#     # ===============================
#     k_late = int(max(10, min(N - 1, round(1.0 / max(dt_ms, 1e-9)))))
#     seg = xN[-(k_late + 1):] if N > k_late + 1 else xN

#     late_activity = (
#         float(np.mean(np.abs(np.diff(seg))))
#         if seg.size > 1 else 0.0
#     )

#     # tail monotonicity (aligned with step direction)
#     tail_len = int(max(30, min(N, round(2.0 / max(dt_ms, 1e-6)))))
#     tail = xN[-tail_len:]
#     tail_dx = np.diff(tail)

#     tail_monotonicity = (
#         float((np.sign(tail_dx) == np.sign(direction)).mean())
#         if tail_dx.size else 0.0
#     )

#     # ===============================
#     # OUTPUT
#     # ===============================
#     return {
#         # global shape
#         "shape_delta_start_end": delta_start_end,
#         "shape_overshoot_norm": overshoot,
#         "shape_undershoot_norm": undershoot,
#         "shape_min_pos_ratio": min_pos_ratio,
#         "shape_min_to_end_norm": min_to_end_norm,
#         "shape_rebound_norm": rebound_norm,
#         "shape_post_min_slope_norm": post_min_slope_norm,

#         # edge stats
#         "edge_count": edge_count,
#         "edge_rate": edge_rate,
#         "edge_max_ratio": edge_max_ratio,
#         "last_edge_pos_ratio": last_edge_pos_ratio,

#         # glitch persistence
#         "glitch_span_ratio": glitch_span_ratio,
#         "glitch_density": glitch_density,

#         # late / tail
#         "late_activity": late_activity,
#         "tail_monotonicity": tail_monotonicity,
#     }


def periodic_autocorr_features(xN: np.ndarray, dt_ms: float) -> dict:
    N = xN.size
    if N < int(CFG["FFT_MIN_N"]):
        return {"per_tail_ac_best": 0.0, "per_tail_ac_lag_ms": 0.0}

    tail = xN[-min(N, int(CFG["FFT_TAIL_MAX_N"])):].copy()
    tail = tail - float(np.median(tail))
    n = tail.size
    if n < 60:
        return {"per_tail_ac_best": 0.0, "per_tail_ac_lag_ms": 0.0}

    denom = float(np.dot(tail, tail) + EPS)
    best = -1.0
    best_lag = 0
    max_lag = min(n // 2, 220)

    for lag in range(3, max_lag):
        c = float(np.dot(tail[:-lag], tail[lag:]) / denom)
        if c > best:
            best = c
            best_lag = lag

    return {
        "per_tail_ac_best": float(best),
        "per_tail_ac_lag_ms": float(best_lag * dt_ms),
    }

def periodic_fft_features(xN: np.ndarray, dt_ms: float) -> dict:
    out0 = {
        "per_fft_peak_freq_hz": 0.0,
        "per_fft_peak_power_ratio": 0.0,
        "per_fft_lowfreq_power_ratio": 0.0,
        "per_fft_peak_to_2nd_ratio": 1.0,
        "per_fft_spectral_entropy": 1.0,
        "per_tail_crest_factor": 0.0,
    }
    N = xN.size
    if N < int(CFG["FFT_MIN_N"]):
        return out0

    tail = xN[-min(N, int(CFG["FFT_TAIL_MAX_N"])):].copy()
    tail = tail - float(np.median(tail))
    n = tail.size
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
    if pwr.size <= 3:
        return out0

    pwr[0] = 0.0
    total = float(np.sum(pwr) + EPS)

    k = int(np.argmax(pwr))
    peak_freq = float(freqs[k])
    peak_ratio = float(pwr[k] / total)

    low_mask = freqs <= float(CFG["LOWFREQ_HZ"])
    low_ratio = float(np.sum(pwr[low_mask]) / total) if np.any(low_mask) else 0.0

    pwr2 = pwr.copy()
    pwr2[k] = 0.0
    k2 = int(np.argmax(pwr2))
    peak_to_2nd = float(pwr[k] / (pwr2[k2] + EPS)) if pwr[k] > 0 else 1.0

    prob = np.clip(pwr / total, 1e-15, 1.0)
    H = -float(np.sum(prob * np.log(prob)))
    Hn = float(H / (np.log(prob.size) + EPS))

    rms = float(np.sqrt(np.mean(tail ** 2)) + EPS)
    crest = float(np.max(np.abs(tail)) / rms)

    return {
        "per_fft_peak_freq_hz": peak_freq,
        "per_fft_peak_power_ratio": peak_ratio,
        "per_fft_lowfreq_power_ratio": low_ratio,
        "per_fft_peak_to_2nd_ratio": peak_to_2nd,
        "per_fft_spectral_entropy": Hn,
        "per_tail_crest_factor": crest,
    }

def periodic_zcr_features(xN: np.ndarray) -> dict:
    N = xN.size
    if N < 120:
        return {"per_tail_zc_rate": 0.0}

    tail = xN[-min(N, int(CFG["FFT_TAIL_MAX_N"])):]
    m = float(np.median(tail))
    s = np.sign(tail - m)
    zc = int(np.sum(np.diff(s) != 0))
    rate = float(zc / max(tail.size - 1, 1))
    return {"per_tail_zc_rate": float(rate)}

def ring_features(xN: np.ndarray, dt_ms: float) -> dict:
    """
    ringing แบบ data-driven: peak count / spacing / decay
    """
    N = xN.size
    if N < 80:
        return {
            "ring_peak_count": 0,
            "ring_mean_peak_spacing_ms": 0.0,
            "ring_std_peak_spacing_ms": 0.0,
            "ring_peak_decay_ratio": 1.0,
            "ring_damping_slope": 0.0,
        }

    tail = xN[-min(N, int(CFG["RING_TAIL_N"])):]
    tail = tail - float(np.median(tail))

    d = np.diff(tail)
    s = np.sign(d)
    peak_idx = np.where((s[:-1] > 0) & (s[1:] < 0))[0] + 1
    peak_count = int(peak_idx.size)

    if peak_count >= 2:
        spacings = np.diff(peak_idx) * dt_ms
        mean_sp = float(np.mean(spacings))
        std_sp = float(np.std(spacings))
    else:
        mean_sp = 0.0
        std_sp = 0.0

    if peak_count >= 3:
        peak_vals = np.abs(tail[peak_idx])
        k = min(5, peak_vals.size)
        first_mean = float(np.mean(peak_vals[:k]))
        last_mean = float(np.mean(peak_vals[-k:]))
        decay_ratio = float(last_mean / (first_mean + EPS))

        y = np.log(peak_vals + EPS)
        xidx = np.arange(y.size, dtype=float)
        try:
            a = float(np.polyfit(xidx, y, 1)[0])
        except Exception:
            a = 0.0
    else:
        decay_ratio = 1.0
        a = 0.0

    return {
        #"ring_peak_count": int(peak_count),
        "ring_mean_peak_spacing_ms": float(mean_sp),
        "ring_std_peak_spacing_ms": float(std_sp),
        "ring_peak_decay_ratio": float(decay_ratio),
        "ring_damping_slope": float(a),
    }

def max_contiguous_activity_ms(xN, t, dt_ms):
    dx = np.abs(np.diff(xN))
    if dx.size < 5:
        return 0.0

    med = np.median(dx)
    mad = np.median(np.abs(dx - med)) + 1e-12
    thr = med + 4.0 * 1.4826 * mad

    active = dx > thr

    max_run = 0
    cur = 0
    for a in active:
        if a:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0

    return max_run * dt_ms

def edge_duty_ratio(xN: np.ndarray, dt_ms: float) -> float:
    dx = np.abs(np.diff(xN))
    if dx.size < 5:
        return 0.0

    med = float(np.median(dx))
    mad = float(_mad(dx, med=med)) + EPS
    q95 = float(np.quantile(dx, 0.95))

    # ใช้ threshold แบบเดียวกับ edge_stats → consistent
    thr = max(med + 6.0 * 1.4826 * mad, 0.25 * q95)

    active = dx > thr
    return float(np.mean(active))   # ratio [0,1]


# ============================================================
# Label handling (train only)
# ============================================================
def get_label_from_group(g: pd.DataFrame) -> tuple[float, str]:
    FAST_MS = float(CFG["FAST_MS"])
    MIN_OUT_MS = float(CFG["MIN_OUT_MS"])
    tmax = float(np.nanmax(g["time_ms"].to_numpy(float))) if len(g) else FAST_MS

    if "true_is_zero" in g.columns and pd.notna(g["true_is_zero"].iloc[0]) and int(g["true_is_zero"].iloc[0]) == 1:
        return FAST_MS, "gt_true_is_zero"
    if "true_wait_time_ms" in g.columns and pd.notna(g["true_wait_time_ms"].iloc[0]):
        w = float(g["true_wait_time_ms"].iloc[0])
        return float(np.clip(w, MIN_OUT_MS, tmax)), "gt_true_wait_time_ms"
    if "wait_time_ms" in g.columns and pd.notna(g["wait_time_ms"].iloc[0]):
        w = float(g["wait_time_ms"].iloc[0])
        return float(np.clip(w, MIN_OUT_MS, tmax)), "gt_wait_time_ms"

    raise RuntimeError("Missing label columns (need true_wait_time_ms/true_is_zero/wait_time_ms)")


# ============================================================
# Main per-wave extraction
# ============================================================
def extract_one_wave(g: pd.DataFrame, mode: str) -> dict:
    t = g["time_ms"].to_numpy(float)
    x = g["value"].to_numpy(float)

    dt_ms = _infer_dt_ms(t, default=0.01)

    xN, meta = normalize_wave(x, t)

    feats = {}
    # meta group
    feats.update(meta)
    # base group
    feats.update(base_features(xN, t, dt_ms))
    # tail group
    feats.update(tail_features(xN, t, dt_ms))
    feats.update(tail_decay_slope(xN, t, dt_ms))
    # shape group
    feats.update(shape_features_clean(xN, t, dt_ms, meta))

    #feats.update(edge_stats(xN, dt_ms, tail_ms=float(CFG["TAIL_MS"])))
    # periodic group
    # feats.update(periodic_autocorr_features(xN, dt_ms))
    # feats.update(periodic_fft_features(xN, dt_ms))
    # feats.update(periodic_zcr_features(xN))
    # ring group
    feats.update(ring_features(xN, dt_ms))

    feats.update(early_quiet_features(xN, t, dt_ms))

    feats["max_contiguous_activity_ms"] = max_contiguous_activity_ms(xN, t, dt_ms)
    #feats["edge_duty_ratio"] = edge_duty_ratio(xN, dt_ms)


    # ✅ 1) edge_stats ก่อน
    es = edge_stats(xN, dt_ms, tail_ms=float(CFG["TAIL_MS"]))
    feats.update(es)

    # ✅ 2) glitch phase (ใช้ค่าจาก edge_stats)
    # แนะนำ: เก็บ threshold ไว้ใน CFG (hybrid / หรือ data-driven ก็ได้ แต่เริ่มแบบนี้ก่อน)
    thr_phase = CFG.get("GLITCH_PHASE_THR", {
        "EARLY_MAX": 0.25,
        "LATE_MIN": 0.60,
        "PERSIST_MIN": 0.50,
        "TAIL_RATE_MIN": 0.05,
        "EDGES_MIN": 3,
    })

    feats["glitch_phase"] = int(glitch_phase_from_edges_dd(
        first_edge_pos_ratio=float(es.get("first_edge_pos_ratio", 0.0)),
        edge_span_ratio=float(es.get("edge_span_ratio", 0.0)),
        tail_edge_rate=float(es.get("tail_edge_rate", 0.0)),
        edge_count=int(es.get("edge_count", 0)),
        thr=thr_phase,
    ))

    # clip risky feats (จุดเดียว)
    feats = _clip_feats(feats)

    out = {
        "wave_id": int(g["wave_id"].iloc[0]),
        **feats,
    }

    if mode == "train":
        w, reason = get_label_from_group(g)
        out["wait_time_ms"] = float(w)
        out["dbg_label_reason"] = str(reason)

    # optional meta passthrough (debug)
    if "type" in g.columns:
        out["type"] = str(g["type"].iloc[0]) if pd.notna(g["type"].iloc[0]) else "unknown"
    if "sd" in g.columns:
        try:
            out["sd"] = float(np.nanmedian(g["sd"].to_numpy(float)))
        except Exception:
            out["sd"] = np.nan

    return out


def normalize_columns(df: pd.DataFrame, id_col: str, sample_col: str, time_col: str, value_col: str) -> pd.DataFrame:
    out = df.copy()
    rename_map = {id_col: "wave_id", sample_col: "sample", time_col: "time_ms", value_col: "value"}
    for k in rename_map:
        if k not in out.columns:
            raise KeyError(f"Missing required column: {k}")
    out = out.rename(columns=rename_map)

    # optional columns
    for opt in ["sd", "type", "low_limit", "high_limit", "true_wait_time_ms", "true_is_zero", "wait_time_ms"]:
        if opt not in out.columns:
            out[opt] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "pred"], required=True)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)

    ap.add_argument("--id-col", default="wave_id")
    ap.add_argument("--sample-col", default="sample")
    ap.add_argument("--time-col", default="time_ms")
    ap.add_argument("--value-col", default="value")

    ap.add_argument("--window-ms", type=float, default=float(CFG["WINDOW_MS"]))
    ap.add_argument("--min-pts", type=int, default=int(CFG["MIN_PTS"]))

    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(in_path)
    raw = normalize_columns(raw, args.id_col, args.sample_col, args.time_col, args.value_col)

    raw["sample"] = raw["sample"].astype(int)
    raw = raw.sort_values(["wave_id", "sample"])

    # cut window
    win = float(args.window_ms)
    raw = raw[raw["time_ms"].astype(float) <= win].copy()

    # drop too-short waves
    cnt = raw.groupby("wave_id")["sample"].count()
    keep = cnt[cnt >= int(args.min_pts)].index
    raw = raw[raw["wave_id"].isin(keep)].copy()

    rows = []
    for wid, g in raw.groupby("wave_id"):
        try:
            rows.append(extract_one_wave(g, mode=args.mode))
        except Exception as e:
            # ถ้าอยาก strict ให้ raise; ถ้าอยากทนก็ log แล้ว skip
            raise RuntimeError(f"wave_id={wid} feature extraction failed: {e}")

    out = pd.DataFrame(rows)

    # pred mode: กัน label หลุดไป
    if args.mode == "pred" and "wait_time_ms" in out.columns:
        out = out.drop(columns=["wait_time_ms"], errors="ignore")

    out.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path} | rows={len(out)} cols={len(out.columns)}")


if __name__ == "__main__":
    main()
