from __future__ import annotations
import argparse
import os
import sys
import pandas as pd
import numpy as np
from scipy.signal import medfilt
 
import numpy as np
import pandas as pd
from scipy.signal import medfilt

def periodic_score(x_tail: np.ndarray) -> float:
    """
    วัดความเป็น periodic (0–1)
    sine / pulse train -> สูง (> 0.45)
    ringing / step      -> ต่ำ
    noise               -> ต่ำ
    """
    x = x_tail - np.median(x_tail)
    n = len(x)
    if n < 30:
        return 0.0

    denom = np.dot(x, x) + 1e-12
    best = 0.0

    # ดู lag ที่เป็นไปได้ (ตัด DC, ตัด lag ใหญ่เกิน)
    # เริ่มที่ 3 เพื่อเลี่ยง correlation ของ noise ข้างเคียง
    for lag in range(3, min(n // 2, 150)):
        c = np.dot(x[:-lag], x[lag:]) / denom
        if c > best:
            best = c

    return float(best)

def compute_features_from_row(x: np.ndarray, N: int) -> dict:
    if N < 50: return {}

    # --- 1. Global Scale & Basic Stats ---
    x_end = float(x[-1])
    std_all = float(np.std(x))
    dx = np.diff(x)
    ringing_energy = float(np.sum(np.abs(dx[3:])))

    # --- 2. Meso Scale (Rolling Window) ---
    half_idx = N // 2
    half_signal = x[half_idx:]
    window_size = 100
    if len(half_signal) > window_size:
        windows = np.lib.stride_tricks.sliding_window_view(half_signal, window_size)
        rolling_std = np.std(windows, axis=1)
        max_rolling_std_half = float(np.max(rolling_std))
    else:
        max_rolling_std_half = std_all

    # --- 3. Micro Scale (Tail) ---
    tail_200 = x[-min(N, 200):]
    tail_50 = x[-min(N, 50):]
    std_tail_50 = float(np.std(tail_50))
    max_dev_tail_50 = float(np.max(np.abs(tail_50 - x_end)))

    # --- [NEW] 4. Special Features for Sine & Pulse ---
    # Crossing Rate: บอก AI ว่าถ้าค่านี้สูง = Sine Wave (นิ่งแล้ว)
    median_tail = np.median(tail_200)
    zero_crossings = len(np.where(np.diff(np.sign(tail_200 - median_tail)))[0])
    crossing_rate = zero_crossings / len(tail_200)

    # Drift Score: เช็คว่ากราฟเอียงลงหรือไม่ (แยก Sine ออกจาก Slow Decay)
    x_pd = pd.Series(x)
    smooth_fast = x_pd.rolling(10).mean().bfill().values
    smooth_slow = x_pd.rolling(50).mean().bfill().values
    drift_score = float(np.mean(np.abs(smooth_fast[-100:] - smooth_slow[-100:])))

    return {
        "x_end": x_end,
        "std_all": std_all,
        "ringing_energy": ringing_energy,
        "max_rolling_std_half": max_rolling_std_half,
        "std_tail_50": std_tail_50,
        "max_dev_tail_50": max_dev_tail_50,
        "mid_to_tail_ratio": max_rolling_std_half / (std_tail_50 + 1e-9),
        "crossing_rate": crossing_rate,      # <--- Key Feature for Sine
        "drift_score": drift_score,          # <--- Key Feature for Slope
        "max_slope": float(np.max(dx))
    }

def extract_features_and_label(group):
    values = group["value"].to_numpy(dtype=float)
    times  = group["time_ms"].to_numpy(dtype=float)
    N = len(values)

    from scipy.signal import medfilt

    DEFAULT_WAIT_MS = 0.0

    # ===============================
    # 1) Filters
    # ===============================
    clean_light = medfilt(values, kernel_size=3)
    
    kernel_medium = 11
    clean_medium = medfilt(values, kernel_size=kernel_medium) # หา Sine

    kernel_heavy = 51 if N > 51 else 11
    clean_heavy = medfilt(values, kernel_size=kernel_heavy)   # ดูทรงกราฟ

    # ===============================
    # 2) Tail & Segment Analysis
    # ===============================
    tail_len = max(int(N * 0.30), 10)

    tail_raw    = values[-tail_len:]
    tail_heavy  = clean_heavy[-tail_len:]
    tail_medium = clean_medium[-tail_len:]
    
    # เอาส่วนหัวมาเทียบส่วนหาง (Return to Baseline Check)
    head_heavy = clean_heavy[:tail_len]
    
    # ===============================
    # 3) Metrics Calculation
    # ===============================
    # Basic
    heavy_amp_tail = float(np.max(tail_heavy) - np.min(tail_heavy))
    global_heavy_range = float(np.max(clean_heavy) - np.min(clean_heavy))
    
    # Noise / Spike
    diff = np.abs(tail_raw - tail_heavy)
    noise_energy = float(np.mean(diff))
    max_spike = float(np.max(diff))

    # Baseline Difference: หัวกับหางต่างกันแค่ไหน?
    # Glitch -> ต่ำ (กลับที่เดิม) | Step -> สูง (เปลี่ยนค่า)
    baseline_diff = float(abs(np.median(head_heavy) - np.median(tail_heavy)))

    # Tail Stability: หางนิ่งจริงไหม (ดูจาก Heavy)
    tail_heavy_range = float(np.max(tail_heavy) - np.min(tail_heavy))

    # Crossing Rate (Secondary check)
    mid = (np.max(tail_heavy) + np.min(tail_heavy)) / 2
    crossings = int(np.sum(np.diff(np.sign(tail_heavy - mid)) != 0))
    crossing_rate = crossings / max(len(tail_heavy), 1)

    # Periodic Score
    per_score = periodic_score(tail_heavy)

    # ===============================
    # 4) Master Gate Logic
    # ===============================
    flag_is_continuous = 0
    flag_is_glitch = 0
    wait_time_ms = DEFAULT_WAIT_MS

    # ---- [Case A] Continuous Wave (Sine / Pulse) ----
    AMP_FLOOR = 0.003
    
    # Ratio: หางขยับเยอะเมื่อเทียบกับทั้งเส้น (Sine ~ 1.0, Ringing < 0.5)
    ratio_tail_global = heavy_amp_tail / (global_heavy_range + 1e-9)
    
    is_periodic = per_score > 0.45
    amp_ok = heavy_amp_tail > AMP_FLOOR
    # เพิ่มเงื่อนไข Ratio เพื่อความชัวร์ (Sine ต้องไม่หยุด)
    is_sustained = ratio_tail_global > 0.2 
    
    if is_periodic and amp_ok and is_sustained:
        wait_time_ms = 0.0
        flag_is_continuous = 1

    # ---- [Case B] Glitch Detector (Improved) ----
    # Type 1: Glitch เล็ก/มาตรฐาน (Logic V19 เดิม)
    elif (max_spike > 0.02) and (global_heavy_range < 0.03):
        wait_time_ms = 0.0
        flag_is_glitch = 1
        
    # Type 2: Glitch ใหญ่ (ที่อาจหลุด V19)
    # เงื่อนไข: มี glitch + กลับมาที่เดิม (Baseline ต่ำ) + หางนิ่งแล้ว
    elif (max_spike > 0.015) and (baseline_diff < 0.02) and (tail_heavy_range < 0.015):
        wait_time_ms = 0.0
        flag_is_glitch = 2 # ให้รู้ว่าเป็น Type 2

    # ---- [Case C] Search Mode (Ringing / Step) ----
    else:
        is_noisy = noise_energy > 0.005

        if is_noisy:
            # Noise เยอะ -> เชื่อ Heavy
            search_values = clean_heavy
            target = np.median(tail_heavy)
            tolerance = max(abs(target * 0.01), 2 * np.std(tail_heavy), 0.003)
            M = 50
        else:
            # Noise น้อย -> เชื่อ Medium (ละเอียดกว่า)
            search_values = clean_medium
            target = np.median(tail_raw)
            sd_ref = np.std(clean_medium[-tail_len:]) # ใช้ SD ช่วงหาง
            tolerance = max(abs(target * 0.01), 3 * sd_ref, 0.001)
            M = 80

        settle_idx = 0
        for i in range(N - M, -1, -1):
            window = search_values[i : i + M]
            if np.any(np.abs(window - target) > tolerance):
                settle_idx = i + M
                break

        wait_time_ms = float(times[min(settle_idx, len(times) - 1)])
        
        # ปัดเศษ 0.1ms เพื่อกัน Noise หลอกว่า 0
        if wait_time_ms > 0 and wait_time_ms < 0.1: wait_time_ms = 0.1

    # ===============================
    # 5) Features for ML (สำคัญมาก!)
    # ===============================
    computed_features = compute_features_from_row(values, N)

    logic_features = {
        "logic_heavy_amp": heavy_amp_tail,
        "logic_global_range": global_heavy_range,
        "logic_noise_energy": noise_energy,
        "logic_crossing_rate": crossing_rate,
        "logic_per_score": per_score,
        
        # [KEY] ตัวช่วย AI แยก Glitch vs Step
        "logic_baseline_diff": baseline_diff, 
        "logic_tail_range": tail_heavy_range,
        
        "logic_flag_continuous": flag_is_continuous,
        "logic_flag_glitch": flag_is_glitch,
    }

    return pd.Series({
        "wave_id": group["wave_id"].iloc[0],
        "wait_time_ms": wait_time_ms,
        **computed_features,
        **logic_features,
    })

def make_wide_plus_features(in_path, out_path, id_col, sample_col, value_col, label_col=None):
    """ 
    ใช้สำหรับโหมด Inference: 
    แปลงข้อมูลเป็น Wide และสกัดฟีเจอร์ด้วย Logic V18 
    **โดยจะไม่เก็บ column 'wait_time_ms' ในฐานะ label แต่จะเปลี่ยนชื่อเป็น 'logic_rule_based_pred' แทน**
    """
    df = pd.read_csv(in_path)
    
    # 1. Pivot to Wide (เผื่อ Model ต้องใช้ Raw Data)
    print(f"Pivoting raw data...")
    wide = df.pivot(index=id_col, columns=sample_col, values=value_col)
    wide.columns = [f"i_{int(c)}" for c in wide.columns]
    
    # เรียง column i_0, i_1...
    i_cols = sorted([c for c in wide.columns if c.startswith("i_")], key=lambda s: int(s.split("_")[1]))
    wide = wide[i_cols].reset_index()

    # เติม 0 ในส่วนข้อมูลดิบที่ว่าง
    wide[i_cols] = wide[i_cols].fillna(0)

    # 2. Calculate Robust Features + Logic Flags (V18)
    print(f"Calculating robust features (V18) for {len(wide)} waves...")
    
    # เรียกใช้ฟังก์ชันหลัก (ตัวเดียวกับที่ใช้ Train)
    features_df = df.groupby(id_col, group_keys=False).apply(extract_features_and_label).reset_index(drop=True)
    
    # --- [จุดสำคัญสำหรับ Inference] ---
    # ค่า wait_time_ms ที่ออกมาจากฟังก์ชัน คือค่าที่ Logic "เดา" (ไม่ใช่เฉลยจริง)
    # เราจะเปลี่ยนชื่อมันเพื่อไม่ให้ AutoGluon เข้าใจผิดว่าเป็น Target Label
    if "wait_time_ms" in features_df.columns:
        features_df = features_df.rename(columns={"wait_time_ms": "logic_rule_based_pred"})
        print(" Renamed 'wait_time_ms' from logic to 'logic_rule_based_pred' for inference safety.")

    # 3. Merge Wide + Features
    final_df = pd.merge(wide, features_df, on=id_col, how="left")

    # 4. Save
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    final_df.to_csv(out_path, index=False)
    print(f"✅ Inference CSV saved: {out_path}")
    print(f"   (Includes features, flags, and logic prediction, BUT NO target label)")
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train", choices=["train", "inference"])
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--id-col", default="wave_id")
    ap.add_argument("--sample-col", default="sample")
    ap.add_argument("--value-col", default="value")
    ap.add_argument("--label-col", default="wait_time_ms")
    args = ap.parse_args()
 
    try:
        if args.mode == "train":
            print(f"Processing Train Data: {args.in_path}")
            df_raw = pd.read_csv(args.in_path)
            train_features = df_raw.groupby(args.id_col, group_keys=False).apply(extract_features_and_label).reset_index(drop=True)
            os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
            train_features.to_csv(args.out_path, index=False)
            print(f" Training features with labels saved: {args.out_path}")
        else:
            print(f"Processing Inference Data: {args.in_path}")
            make_wide_plus_features(args.in_path, args.out_path, args.id_col, args.sample_col, args.value_col, args.label_col)
    except Exception as e:
        print(f" ERROR: {e}")
        sys.exit(1)
 
if __name__ == "__main__":
    main()
    
# V18 Logic
# def extract_features_and_label(group):
#     values = group["value"].to_numpy(dtype=float)
#     times  = group["time_ms"].to_numpy(dtype=float)
#     N = len(values)

#     # Config
#     DEFAULT_WAIT_MS = 0.0

#     # ===============================
#     # 1) Filters
#     # ===============================
#     clean_light = medfilt(values, kernel_size=3)

#     kernel_medium = 11
#     clean_medium = medfilt(values, kernel_size=kernel_medium)

#     kernel_heavy = 51 if N > 51 else 11
#     clean_heavy = medfilt(values, kernel_size=kernel_heavy)

#     # ===============================
#     # 2) Tail
#     # ===============================
#     tail_len = max(int(N * 0.30), 10)

#     tail_raw    = values[-tail_len:]
#     tail_heavy  = clean_heavy[-tail_len:]
#     tail_medium = clean_medium[-tail_len:]

#     # ===============================
#     # 3) Metrics
#     # ===============================
#     heavy_amp_tail = float(np.max(tail_heavy) - np.min(tail_heavy))
#     global_heavy_range = float(np.max(clean_heavy) - np.min(clean_heavy))

#     diff = np.abs(tail_raw - tail_heavy)
#     noise_energy = float(np.mean(diff))
#     max_spike = float(np.max(diff))

#     # crossing rate (secondary)
#     mid = (np.max(tail_heavy) + np.min(tail_heavy)) / 2
#     crossings = int(np.sum(np.diff(np.sign(tail_heavy - mid)) != 0))
#     crossing_rate = crossings / max(len(tail_heavy), 1)

#     # ⭐ periodic score (KEY)
#     per_score = periodic_score(tail_heavy)

#     # ===============================
#     # 4) Master Gate (Sine / Pulse)
#     # ===============================
#     flag_is_continuous = 0
#     flag_is_glitch = 0

#     # ---- Sine / Pulse-train Logic ----
#     tail_sd = float(np.std(tail_heavy) + 1e-12)

#     # Thresholds
#     AMP_FLOOR = 0.003
#     SD_FLOOR  = 0.0005   # กัน flat+noise หลอก per_score

#     is_periodic = per_score > 0.45
#     amp_ok = heavy_amp_tail > AMP_FLOOR
#     sd_ok  = tail_sd > SD_FLOOR

#     # กัน step/ringing หลอกเพิ่ม (ถ้า global range สูง แปลว่ามี movement ไม่ใช่ noise ราบเรียบ)
#     not_step_like = global_heavy_range > AMP_FLOOR 

#     if is_periodic and amp_ok and sd_ok and not_step_like:
#         wait_time_ms = 0.0
#         flag_is_continuous = 1

#     # ---- Glitch ----
#     elif (max_spike > 0.02) and (global_heavy_range < 0.03):
#         wait_time_ms = DEFAULT_WAIT_MS
#         flag_is_glitch = 1

#     # ===============================
#     # 5) Settle Search
#     # ===============================
#     else:
#         is_noisy = noise_energy > 0.005

#         if is_noisy:
#             # Noise เยอะ -> ใช้ Heavy เดินหา
#             search_values = clean_heavy
#             target = np.median(tail_heavy)
#             tolerance = max(abs(target * 0.01), 2 * np.std(tail_heavy), 0.003)
#             M = 50
#         else:
#             # Signal ค่อนข้างเรียบ -> ใช้ Medium เดินหา
#             search_values = clean_medium
#             target = np.median(tail_raw)
#             sd_medium = np.std(tail_medium)
#             tolerance = max(abs(target * 0.01), 3 * sd_medium, 0.001)
#             M = 80

#         settle_idx = 0
#         for i in range(N - M, -1, -1):
#             window = search_values[i : i + M]
#             if np.any(np.abs(window - target) > tolerance):
#                 settle_idx = i + M
#                 break

#         wait_time_ms = float(times[min(settle_idx, len(times) - 1)])

#     # ===============================
#     # 6) Features for ML
#     # ===============================
#     computed_features = compute_features_from_row(values, N)

#     logic_features = {
#         "logic_heavy_amp": heavy_amp_tail,
#         "logic_global_range": global_heavy_range,
#         "logic_noise_energy": noise_energy,
#         "logic_crossing_rate": crossing_rate,
#         "logic_per_score": per_score,
#         "logic_flag_continuous": flag_is_continuous,
#         "logic_flag_glitch": flag_is_glitch,
#     }

#     return pd.Series({
#         "wave_id": group["wave_id"].iloc[0],
#         "wait_time_ms": wait_time_ms,
#         **computed_features,
#         **logic_features,
#     })


# V17 (sine wave มีหลุด)
# def extract_features_and_label(group):
#     values = group["value"].to_numpy(dtype=float)
#     times  = group["time_ms"].to_numpy(dtype=float)
#     N = len(values)

#     from scipy.signal import medfilt
    
#     # --- [CONFIG] ---
#     DEFAULT_WAIT_MS = 0.0

#     # 1. Filters
#     clean_light = medfilt(values, kernel_size=3)
    
#     # --- [NEW] Medium Filter ---
#     # ใช้ Kernel 11 เพื่อกรอง Noise ที่หางให้เรียบ แต่ยังเก็บทรงกราฟ Ringing ไว้ได้
#     kernel_medium = 11
#     clean_medium = medfilt(values, kernel_size=kernel_medium)
    
#     # Heavy Filter (สำหรับ Glitch/Sine)
#     kernel_heavy = 51 if N > 51 else 11
#     clean_heavy = medfilt(values, kernel_size=kernel_heavy)

#     # 2. Tail Analysis (30%)
#     tail_len = int(N * 0.30)
#     if tail_len < 10: tail_len = 10
    
#     tail_raw   = values[-tail_len:]
#     tail_heavy = clean_heavy[-tail_len:]
    
#     # [METRIC 1] Heavy Amplitude @ Tail
#     heavy_amp_tail = np.max(tail_heavy) - np.min(tail_heavy)
    
#     # [METRIC 2] Global Heavy Range (ใช้ทั้งเส้น ห้ามตัด Margin!)
#     # --- FIXED: ต้องใช้ทั้งเส้นเพื่อให้เห็นการดีดตัวของ Step/Ringing ---
#     global_heavy_range = np.max(clean_heavy) - np.min(clean_heavy)

#     # [METRIC 3] Noise Energy
#     diff = np.abs(tail_raw - tail_heavy)
#     noise_energy = np.mean(diff)
#     max_spike = np.max(diff)

#     # [METRIC 4] Crossing Rate บน Heavy Filter
#     heavy_mid = (np.max(tail_heavy) + np.min(tail_heavy)) / 2
#     crossings = np.sum(np.diff(np.sign(tail_heavy - heavy_mid)) != 0)
#     heavy_crossing_rate = crossings / len(tail_heavy)

#     # ---------- 3) Master Decision Logic ----------
    
#     flag_is_continuous = 0
#     flag_is_glitch = 0
    
#     # [Case A] Continuous Wave (Sine / Pulse Train)
#     is_active_wave = heavy_amp_tail > 0.015  
#     is_sine_behavior = (heavy_crossing_rate > 0.08) or (heavy_amp_tail > 0.05)

#     if is_active_wave and is_sine_behavior and (crossings > 0):
#         wait_time_ms = DEFAULT_WAIT_MS
#         flag_is_continuous = 1

#     # [Case B] Glitch Detector
#     # ใช้ global_heavy_range แบบเต็มเส้น (ไม่ตัดขอบ) เพื่อแยก Glitch ออกจาก Ringing
#     elif (max_spike > 0.02) and (global_heavy_range < 0.025):
#         wait_time_ms = DEFAULT_WAIT_MS
#         flag_is_glitch = 1     

#     else:
#         # [Case C] Search Mode
        
#         is_noisy = noise_energy > 0.005
        
#         if is_noisy:
#             # ถ้า Noise หนักมาก (เช่น Glitch รัวๆ) -> ใช้ Heavy ตรวจ
#             search_values = clean_heavy
#             target = np.median(tail_heavy)
#             tolerance = max(abs(target * 0.01), 2 * np.std(tail_heavy), 0.003)
#             M = 50 
#         else:
#             # Ringing/Step/General Case -> ใช้ Medium Filter ตรวจ (ตามที่คุณขอ!)
#             # การใช้ Medium จะช่วยกรอง Noise ฝอยๆ ที่ปลายหางออก ทำให้หาจุดนิ่งได้แม่นยำขึ้น
#             search_values = clean_medium 
#             target = np.median(tail_raw)
            
#             # คำนวณ SD จากกราฟ Medium (ซึ่งจะต่ำกว่า Raw)
#             sd_medium = np.std(clean_medium[-tail_len:])
            
#             # Tolerance: 1% หรือ 3เท่าของ SD (Medium)
#             tolerance = max(abs(target * 0.01), 3 * sd_medium, 0.001)
            
#             # Window ใหญ่หน่อย (80) เพื่อกันหลอกตา
#             M = 80

#         settle_idx_label = 0
#         for i in range(N - M, -1, -1):
#             window = search_values[i : i + M]
#             if np.any(np.abs(window - target) > tolerance):
#                 settle_idx_label = i + M
#                 break

#         wait_time_ms = float(times[min(settle_idx_label, len(times) - 1)])
        
#         # บังคับขั้นต่ำ
#         if wait_time_ms < DEFAULT_WAIT_MS: wait_time_ms = DEFAULT_WAIT_MS

#     # คำนวณ Features พื้นฐาน
#     computed_features = compute_features_from_row(values, N)

#     # [KEY] เพิ่ม Logic Features กลับเข้ามา (สำคัญมากสำหรับการเทรน)
#     logic_features = {
#         "logic_heavy_amp": heavy_amp_tail,
#         "logic_global_range": global_heavy_range,
#         "logic_noise_energy": noise_energy,
#         "logic_heavy_crossing": heavy_crossing_rate,
#         "logic_flag_continuous": flag_is_continuous, 
#         "logic_flag_glitch": flag_is_glitch          
#     }

#     return pd.Series({
#         "wave_id": group["wave_id"].iloc[0],
#         "wait_time_ms": wait_time_ms,
#         **computed_features,
#         **logic_features 
#     })

## V-15 pred 0.1ms 

# def extract_features_and_label(group):
#     values = group["value"].to_numpy(dtype=float)
#     times  = group["time_ms"].to_numpy(dtype=float)
#     N = len(values)

#     from scipy.signal import medfilt

#     # 1. Filters
#     clean_light = medfilt(values, kernel_size=3)

#     kernel_medium = 11
#     clean_medium = medfilt(values, kernel_size=kernel_medium)
    
#     kernel_heavy = 51 if N > 51 else 11
#     clean_heavy = medfilt(values, kernel_size=kernel_heavy)

#     # 2. Tail Analysis (30%)
#     tail_len = int(N * 0.30)
#     if tail_len < 10: tail_len = 10
    
#     tail_raw   = values[-tail_len:]
#     tail_heavy = clean_heavy[-tail_len:]

#     # [NEW] Global Analysis (ดูภาพรวมทั้งเส้น)
#     # ตัดขอบซ้ายขวาออกหน่อยเพื่อกัน edge effect ของ medfilt (ที่ชอบตกลงศูนย์)
#     margin = kernel_heavy // 2
#     if N > 2 * margin:
#         valid_heavy = clean_heavy[margin:-margin]
#     else:
#         valid_heavy = clean_heavy
        
#     # เช็คว่าโครงสร้างหลักมีการขยับตัวไหม? (Ringing/Step จะขยับ, Glitch จะนิ่ง)
#     global_heavy_range = np.max(valid_heavy) - np.min(valid_heavy)
    
#     # คำนวณความต่าง (Spike) ที่หาง
#     diff = np.abs(tail_raw - tail_heavy)
#     max_spike = np.max(diff)

#     # คำนวณ Amplitude หาง (สำหรับ Sine Check)
#     heavy_amp_tail = np.max(tail_heavy) - np.min(tail_heavy)

#     # ---------- 3) Master Decision Logic ----------
    
#     # [Case A] Continuous Wave (Sine / Pulse Train)
#     # ดูเฉพาะที่หาง: ถ้าหางยังแกว่งแรง แสดงว่าไม่ยอมหยุด
#     is_active_wave = heavy_amp_tail > 0.015  
#     heavy_mid = (np.max(tail_heavy) + np.min(tail_heavy)) / 2
#     crossings = np.sum(np.diff(np.sign(tail_heavy - heavy_mid)) != 0)

#     if is_active_wave and (crossings > 0):
#         # Sine Wave, Pulse Train, Triangle
#         wait_time_ms = 0.0

#     # [Case B] Glitch Detector (แก้จุดผิดตรงนี้!)
#     # เงื่อนไข:
#     # 1. มีหนามแหลมที่หาง (max_spike สูง)
#     # 2. แต่โครงสร้างหลักต้อง "นิ่งสนิท" (global_heavy_range ต่ำ) 
#     #    -> ถ้าเป็น Ringing ค่า global_heavy_range จะสูง (เพราะกราฟมันวิ่งขึ้นลง) ทำให้ไม่เข้าเงื่อนไขนี้
#     elif (max_spike > 0.02) and (global_heavy_range < 0.035):
#         # เป็น Glitch โดดๆ บนพื้นเรียบ -> Force 0ms
#         wait_time_ms = 0.0

#     else:
#         # [Case C] Search Mode (Ringing, Step Response, หรือ Glitch ที่เนียนๆ)
#         # ID:46 (Ringing) จะตกมาที่นี่ เพราะ global_heavy_range มันสูง (~1.5V)
        
#         noise_energy = np.mean(diff)
#         # Threshold noise energy (ถ้าสูงแปลว่า Raw ขรุขระกว่า Heavy เยอะ)
#         is_noisy = noise_energy > 0.005

#         if is_noisy:
#             # Glitch/Noise เยอะ -> ใช้ Heavy ตรวจ
#             search_values = clean_heavy
#             target = np.median(tail_heavy)
#             tolerance = max(abs(target * 0.01), 2 * np.std(tail_heavy), 0.003)
#         else:
#             # Ringing/Step -> ใช้ Light ตรวจ
#             search_values = clean_light
#             target = np.median(tail_raw)
#             # Tolerance 1% หรือ 4*SD
#             tolerance = max(abs(target * 0.01), 3 * np.std(clean_light[-tail_len:]), 0.001)

#         # Backward Search
#         settle_idx_label = 0
#         M = 80
#         for i in range(N - M, -1, -1):
#             window = search_values[i : i + M]
#             if np.any(np.abs(window - target) > tolerance):
#                 settle_idx_label = i + M
#                 break

#         wait_time_ms = float(times[min(settle_idx_label, len(times) - 1)])

#     computed_features = compute_features_from_row(values, N)

#     return pd.Series({
#         "wave_id": group["wave_id"].iloc[0],
#         "wait_time_ms": wait_time_ms,
#         **computed_features,
#     })


# def extract_features_and_label(group):
    
#     values = group['value'].values 
#     times = group['time_ms'].values
#     N = len(values)
    
#     # 1. กรองสัญญาณ (ใช้ Medfilt ลบ Spike)
#     from scipy.signal import medfilt
#     clean_values = medfilt(values, kernel_size=21)
    
#     # 2. วิเคราะห์หางกราฟ (Tail Analysis) เพื่อเลือกโหมด
#     tail_len = int(N * 0.20) # ดู % สุดท้าย
#     tail_values = values[-tail_len:]
#     final_target = np.median(tail_values)
#     tail_sd = np.std(tail_values)
    
#     # --- [NEW] DUAL-MODE LOGIC ---
#     # ถ้า SD ของหางกราฟต่ำมาก (น้อยกว่า 0.5% ของค่าเป้าหมาย) แสดงว่าเป็นกราฟนิ่ง (Type 0, 1)
#     # เราต้องใช้ Tolerance ที่แคบมากๆ เพื่อจับ Ringing
#     is_precision_signal = tail_sd < (abs(final_target) * 0.005)
    
#     if is_precision_signal:
#         # โหมดเข้มงวด: สำหรับกราฟ Ringing ทั่วไป (แก้ ID:91, 98)
#         # บังคับให้นิ่งจริงๆ ที่ 1.5% ถึงจะยอม
#         tolerance = max(abs(final_target * 0.015), 3 * tail_sd)
#     else:
#         # โหมดใจดี: สำหรับ Sine Wave และ Pulse (แก้ ID:81, 83)
#         # ยอมให้แกว่งได้ตามธรรมชาติของมัน (6 sigma ครอบคลุม 99.7% ของการแกว่ง)
#         tolerance = 6 * tail_sd

#     # 3. Backward Search (เหมือนเดิม)
#     settle_idx_label = 0
#     M = 50 
#     for i in range(N - M, -1, -1):
#         window = clean_values[i : i + M]
#         if np.any(np.abs(window - final_target) > tolerance):
#             settle_idx_label = i + M
#             break
            
#     wait_time_ms = times[min(settle_idx_label, len(times)-1)]
    
#     # ... (ส่วน feature extraction เหมือนเดิม) ...
#     computed_features = compute_features_from_row(values, N)
#     return pd.Series(
#         {'wave_id': group['wave_id'].iloc[0]
#          , 'wait_time_ms': wait_time_ms, **computed_features})

#--- OLD VERSION ---

    # values = group['value'].values 
    # times = group['time_ms'].values
    # N = len(values)
    
    # # --- [UPGRADE 1] Median Filter แรงๆ เพื่อลบ Spike และ Sine ---
    # # ใช้ kernel=21 เพื่อรีดกราฟให้เป็นเส้นตรง (เฉพาะตอนทำเฉลย Label)
    # clean_values = medfilt(values, kernel_size=21) 
    
    # # 1. หาค่าเป้าหมาย (Final Target)
    # final_target = np.median(values[-int(N*0.1):])
    
    # # --- [UPGRADE 2] Tolerance กว้างขึ้น (Dynamic) ---
    # # ใช้ 10% เพื่อยอมรับ Sine Wave ที่แกว่งแรงๆ ได้
    # noise_sd = np.std(values[-int(N*0.1):])
    # tolerance = max(abs(final_target * 0.15), 6 * noise_sd)
    
    # # 3. ROBUST BACKWARD SEARCH
    # # หาจากหลังมาหน้า (Backward) ถ้าเจอจุดที่หลุด Band ให้หยุด
    # settle_idx_label = 0
    # M = 50 # ต้องนิ่งต่อเนื่อง 50 จุด
    
    # for i in range(N - M, -1, -1):
    #     window = clean_values[i : i + M]
    #     # ถ้าหน้าต่างนี้มีส่วนไหนหลุด Tolerance แสดงว่ายังไม่นิ่ง
    #     if np.any(np.abs(window - final_target) > tolerance):
    #         settle_idx_label = i + M
    #         break
            
    # wait_time_ms = times[min(settle_idx_label, len(times)-1)]
    
    # # คำนวณ Features (ใช้ค่าดิบ values ส่งให้ AI)
    # computed_features = compute_features_from_row(values, N)
    
    # ordered_output = {
    #     'wave_id': group['wave_id'].iloc[0], 
    #     'wait_time_ms': wait_time_ms, 
    #     **computed_features
    # }
    
    # return pd.Series(ordered_output)
