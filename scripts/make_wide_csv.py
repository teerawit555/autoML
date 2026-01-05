# # make_wide_csv.py
# from __future__ import annotations

# import argparse
# import os
# import sys
# import pandas as pd
# import numpy as np

# def make_wide(
#     in_path: str,
#     out_path: str,
#     id_col: str = "wave_id",
#     # *** แก้ไข: ใช้ "sample" เป็น Default ***
#     sample_idx_col: str = "sample",
#     # *** แก้ไข: ใช้ "value" เป็น Default ***
#     value_col: str = "value",
#     label_col: str = "wait_time_ms",
# ) -> None:
    
#     """ Original wide conversion logic for inference data """
    
#     if not os.path.exists(in_path):
#         raise FileNotFoundError(f'Input file not found: "{in_path}"')

#     df = pd.read_csv(in_path)

#     required = {id_col, sample_idx_col, value_col}
#     missing = required - set(df.columns)
#     if missing:
#         raise ValueError(f"Missing required columns in input: {sorted(missing)}")

#     # label is optional (for inference), but recommended
#     has_label = label_col in df.columns

#     # Identify meta columns (anything not id/sample/value/label)
#     cols_to_exclude = {id_col, sample_idx_col, value_col, label_col}
#     meta_cols = [c for c in df.columns if c not in cols_to_exclude]

#     # Pivot to wide i_0..i_N
#     wide = df.pivot(index=id_col, columns=sample_idx_col, values=value_col)

#     # Rename columns to i_{k}
#     wide.columns = [f"i_{int(c)}" for c in wide.columns]
#     wide = wide.reset_index()

#     # --- Attach meta columns (sd, low limit, high limit, time, force_mA, etc.) ---
#     if meta_cols:
#         # Groupby และเอาค่าแรกของคอลัมน์ meta data
#         meta_first = df.groupby(id_col, as_index=False)[meta_cols].first()
#         wide = wide.merge(meta_first, on=id_col, how="left")

#     # --- Attach label (wait_time_ms) ---
#     if has_label:
#         y_first = df.groupby(id_col, as_index=False)[[label_col]].first()
#         wide = wide.merge(y_first, on=id_col, how="left")

#     # --- ปรับปรุงตรรกะการจัดเรียงคอลัมน์ (แก้ไขปัญหา KeyError/IndexError) ---
#     i_cols = [c for c in wide.columns if c.startswith("i_")]
#     i_cols_sorted = sorted(i_cols, key=lambda s: int(s.split("_")[1]))
    
#     # สร้างรายการคอลัมน์สุดท้าย
#     final_cols = [id_col]              # 1. wave_id (ID Column)
#     final_cols.extend(i_cols_sorted)   # 2. i_0, i_1, i_2, ... (Waveform Data)
    
#     # 3. Meta Data และ Label ที่เหลือ (ต้องไม่ซ้ำกับ ID หรือ i_cols)
#     remaining_cols = [c for c in wide.columns if c not in final_cols]
#     final_cols.extend(remaining_cols)
    
#     wide = wide[final_cols] # จัดเรียง DataFrame ด้วยรายการคอลัมน์ใหม่

#     wide.to_csv(out_path, index=False)
#     print(f" Wrote wide CSV: {out_path}")
#     print(f"Rows(waves): {len(wide)} | i_cols: {len(i_cols_sorted)} | meta_cols: {len(meta_cols)} | label: {has_label}")

# def extract_features_and_label(group):
#     """
#     This function processes each wave_id to extract both features and 
#     the settling time (wait_time_ms) as a label.
#     """
#     values = group['value'].values # Your new column name
#     times = group['time_ms'].values
    
#     # --- 1. SETTLING TIME CALCULATION (LABEL) ---
#     last_10_pct_idx = max(1, int(len(values) * 0.1))
#     mean_last = np.mean(values[-last_10_pct_idx:])
#     std_last = np.std(values[-last_10_pct_idx:])
    
#     # Target value and tolerance for labeling
#     tolerance = abs(mean_last * 0.01) # 1% Threshold
    
#     settle_idx = len(values) - 1
#     for i in range(len(values) - 1, -1, -1):
#         if abs(values[i] - mean_last) > tolerance:
#             settle_idx = i + 1
#             break
#     wait_time_ms = times[min(settle_idx, len(times)-1)]

#     # --- 2. ADVANCED FEATURE EXTRACTION ---
#     # Slopes
#     slopes = np.diff(values) if len(values) > 1 else [0]
    
#     # Ringing Energy (Sum of squared differences from the final mean)
#     ringing_energy = np.sum((values[-last_10_pct_idx:] - mean_last)**2)
    
#     # Band 3-Sigma (3 times the standard deviation of the last portion)
#     band_3std_last = 3 * std_last

#     # --- 2. FEATURE EXTRACTION ---
#     # Add all your existing feature calculations here
#     features = {
#         'wave_id': group['wave_id'].iloc[0],
#         'time': times[0],                     # Start time
#         'sd': np.std(values),                 # Standard deviation (same as std_all)
#         'low_limit': mean_last - tolerance,    # Dynamic low limit
#         'high_limit': mean_last + tolerance,   # Dynamic high limit
#         'x0': values[0],                      # Initial value
#         'x_end': values[-1],                  # Final value
#         'mean_all': np.mean(values),
#         'std_all': np.std(values),
#         'mean_last': mean_last,
#         'std_last': std_last,
#         'peak_rel': np.max(values) - mean_last,
#         'trough_rel': mean_last - np.min(values),
#         'max_slope': np.max(slopes),
#         'min_slope': np.min(slopes),
#         'ringing_energy': ringing_energy,
#         'settle_idx': settle_idx,             # The index where it settled
#         'band_3std_last': band_3std_last,
#         'wait_time_ms': wait_time_ms          # Target Label for Training
#     }
#     return pd.Series(features)

# def main():

#     ap = argparse.ArgumentParser(description="Convert long-format waveform CSV to wide-format (i_0..i_N).")
#     # ปรับ Default Input และ Value Col
#     ap.add_argument("--mode", default="train", choices=["train", "inference"], help="Processing mode")
#     ap.add_argument("--in", dest="in_path", default="data/raw/data1000samples_test.csv", help="Input long CSV path (default: data1000samples_test.csv)")
#     ap.add_argument("--out", dest="out_path", default="data/processed/inference/wide.csv", help="Output wide CSV path (default: wide.csv)")
#     ap.add_argument("--id-col", default="wave_id")
#     ap.add_argument("--sample-idx-col", default="sample") 
#     ap.add_argument("--value-col", default="value")
#     ap.add_argument("--label-col", default="wait_time_ms", help="Label column name (used for exclusion from features)") # เปลี่ยนคำอธิบาย
#     args = ap.parse_args()

#     try:
#             # --- MODE 1: TRAIN (Extract Stats + Automated Label) ---
#         if args.mode == "train":
#             print(f"Reading raw data for training: {args.in_path}")
#             df_raw = pd.read_csv(args.in_path)
            
#             print("Extracting features and calculating settling times (Labeling)...")
#             # Group by wave_id and apply the automated logic
#             train_features = df_raw.groupby(args.id_col, group_keys=False).apply(extract_features_and_label).reset_index(drop=True)
            
#             os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
#             train_features.to_csv(args.out_path, index=False)
#             print(f"Done! Training features saved to: {args.out_path}")

#         # --- MODE 2: INFERENCE (Convert to i_0...i_N for Predictor) ---
#         else:
#             print(f"Converting to wide format for inference: {args.in_path}")
#             make_wide(
#                 in_path=args.in_path,
#                 out_path=args.out_path,
#                 id_col=args.id_col,
#                 sample_idx_col=args.sample_idx_col,
#                 value_col=args.value_col,
#                 label_col=args.label_col,
#             )
                
#     except Exception as e:
#         print(f" ERROR: {e}")
#         sys.exit(1)

#     # Load raw data
#     df_raw = pd.read_csv('data/raw/data_for_train.csv')
    
#     # Process each wave_id
#     print("Extracting features and calculating settling times...")
#     # Group by wave_id and apply the function
#     train_features = df_raw.groupby('wave_id').apply(extract_features_and_label).reset_index(drop=True)
    
#     # Save to the final training file
#     output_path = 'data/processed/train/train_features.csv'
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     train_features.to_csv(output_path, index=False)
#     print(f"Done! File saved to: {output_path}")

# if __name__ == "__main__":
#     main()

#------ 1 Data for training (with labels) ------#
# python scripts/make_wide_csv.py --mode train --in data/raw/data_for_train.csv --out data/processed/train/train_features.csv
#------ 2 Data for inference (without labels) ------#
# python scripts/make_wide_csv.py --mode inference --in data/raw/data_1000_samples_to_pred.csv --out data\processed\inference\wide_1000_samples_to_pred.csv

from __future__ import annotations
import argparse
import os
import sys
import pandas as pd
import numpy as np
from scipy.signal import medfilt # เพิ่ม Library สำหรับกรองเข็มแหลม
 
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
    values = group['value'].values 
    times = group['time_ms'].values
    N = len(values)
    
    # 1. เตรียมข้อมูล 2 ชุด (ทางใครทางมัน)
    from scipy.signal import medfilt
    
    # [ชุด A] Light Filter: สำหรับตรวจ Ringing/Step (ต้องการความละเอียด)
    clean_light = medfilt(values, kernel_size=3) 
    
    # [ชุด B] Heavy Filter: สำหรับตรวจ Sine/Glitch (ต้องการความเรียบ)
    # ใช้ Kernel ใหญ่ (51) เพื่อบดขยี้ Sine/Glitch ให้เป็นเส้นตรงกลาง
    kernel_heavy = 51 if N > 51 else 11
    clean_heavy = medfilt(values, kernel_size=kernel_heavy) 
    
    # 2. วิเคราะห์หาง (Tail Analysis) เพื่อเลือกเส้นทาง
    tail_len = int(N * 0.30)
    if tail_len < 10: tail_len = 10
    
    tail_raw = values[-tail_len:]
    tail_heavy = clean_heavy[-tail_len:]
    
    # คำนวณค่าสถิติ
    heavy_amplitude = np.max(tail_heavy) - np.min(tail_heavy) # ความสูงของคลื่นหลัง Filter
    heavy_target = np.median(tail_heavy)                      # Target ของชุด Heavy
    
    raw_target = np.median(tail_raw)                          # Target ของชุด Light/Raw
    raw_sd = np.std(tail_raw)
    
    # --- [DECISION LOGIC] เลือกเส้นทาง ---
    
    # เช็คว่ากราฟยังแกว่งเป็นลอนคลื่นชัดเจนไหม (แม้จะโดน Heavy Filter แล้ว)
    # ถ้าใช่ แสดงว่าเป็น Sine Wave หรือ Pulse ต่อเนื่อง -> บังคับ 0ms
    is_continuous_wave = heavy_amplitude > (abs(heavy_target) * 0.015)
    
    if is_continuous_wave:
        # [Path 1: Sine Wave / Continuous Pulse]
        # ไม่ต้อง Search ให้เสียเวลา เพราะมันคือสภาวะปกติของคลื่นประเภทนี้
        wait_time_ms = 0.0
        
    else:
        # [Path 2: Ringing, Step, or Glitch] -> ต้องหาจุดนิ่ง
        
        # เช็คว่ามี Noise/Glitch เยอะไหม?
        # ดูผลต่างระหว่าง Raw กับ Heavy (ถ้าต่างกันเยอะแสดงว่ามี Glitch/Noise)
        noise_energy = np.mean(np.abs(tail_raw - tail_heavy))
        is_noisy_or_glitchy = noise_energy > (abs(heavy_target) * 0.005)
        
        if is_noisy_or_glitchy:
            # [Case: Glitch / Burst Noise] -> ใช้ Heavy Filter ตรวจ
            search_values = clean_heavy
            target = heavy_target # สำคัญ! ต้องใช้ Target ของ Heavy ให้ตรงกัน
            
            # Tolerance ขั้นต่ำ 1% (เผื่อรอยขรุขระจากการกรอง)
            tolerance = max(abs(target * 0.01), 2 * np.std(tail_heavy))
            
        else:
            # [Case: Ringing / Step Response] -> ใช้ Light Filter ตรวจ
            search_values = clean_light
            target = raw_target   # ใช้ Target ของ Raw/Light
            
            # Tolerance เข้มงวด 1% หรือ 4 เท่าของ Noise พื้นฐาน
            tail_sd_light = np.std(clean_light[-tail_len:])
            tolerance = max(abs(target * 0.01), 4 * tail_sd_light)

        # --- Backward Search ---
        settle_idx_label = 0
        M = 50 
        for i in range(N - M, -1, -1):
            window = search_values[i : i + M]
            if np.any(np.abs(window - target) > tolerance):
                settle_idx_label = i + M
                break
        
        wait_time_ms = times[min(settle_idx_label, len(times)-1)]
    
    # Features (เหมือนเดิม)
    computed_features = compute_features_from_row(values, N)
    
    return pd.Series({
        'wave_id': group['wave_id'].iloc[0], 
        'wait_time_ms': wait_time_ms, 
        **computed_features
    })

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
 
def make_wide_plus_features(in_path, out_path, id_col, sample_col, value_col, label_col):
    """ ใช้สำหรับโหมด Inference: แปลงข้อมูลเป็น Wide และสกัดฟีเจอร์แบบทนทานต่อ NaN """
    df = pd.read_csv(in_path)
   
    # 1. Pivot to Wide (i_0...i_N)
    wide = df.pivot(index=id_col, columns=sample_col, values=value_col)
    wide.columns = [f"i_{int(c)}" for c in wide.columns]
    i_cols = sorted(list(wide.columns), key=lambda s: int(s.split("_")[1]))
    wide = wide.reset_index()
 
    # 2. Meta Columns
    meta_exclude = {id_col, sample_col, value_col, label_col, 'time', 'time_ms'}
    meta_cols = [c for c in df.columns if c not in meta_exclude]
    if meta_cols:
        meta_df = df.groupby(id_col)[meta_cols].first().reset_index()
        wide = wide.merge(meta_df, on=id_col, how='left')
 
    # 3. Calculate Features per Wave
    print(f"Calculating robust features for {len(wide)} waves...")
    feats_list = []
    for _, row in wide.iterrows():
        # ลบ NaN ออกเพื่อให้ได้ waveform จริง (กรณี wave สั้นไม่เท่ากัน)
        waveform = row[i_cols].dropna().to_numpy(dtype=float)
        feat = compute_features_from_row(waveform, len(waveform))
        feats_list.append(feat)
   
    feat_df = pd.DataFrame(feats_list)
    final_df = pd.concat([wide, feat_df], axis=1)
   
    # เติม 0 ในส่วนข้อมูลดิบที่ว่างเพื่อให้ Model ไม่พัง
    final_df[i_cols] = final_df[i_cols].fillna(0)
 
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    final_df.to_csv(out_path, index=False)
    print(f"✅ Wide CSV with Robust Features saved: {out_path}")
 
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