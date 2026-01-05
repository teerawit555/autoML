import pandas as pd
import numpy as np
from scipy.signal import medfilt

# --- จำลองฟังก์ชันที่คุณใช้จริง (หรือ import มาจากไฟล์ make_wide_csv) ---
def debug_label_logic(values, times, wave_id):
    N = len(values)
    
    # 1. จำลองการกรอง (ดูว่า medfilt มันรีดกราฟแบนจริงมั้ย?)
    clean_values = medfilt(values, kernel_size=21) # <--- ค่าที่คุณใช้จริง
    
    # 2. จำลองการหา Target
    final_target = np.median(values[-int(N*0.1):])
    
    # 3. จำลอง Tolerance (ดูว่ากว้างพอครอบ Sine มั้ย?)
    noise_sd = np.std(values[-int(N*0.1):])
    # สูตรล่าสุดที่คุณใช้
    tolerance_calc = max(abs(final_target * 0.15), 6 * noise_sd) 
    
    # 4. จำลอง Backward Search
    settle_idx = 0
    M = 50
    fail_reason = "None (Settled at 0)"
    
    # ลอง print ค่าช่วง Sine Wave ดูว่าหลุด Tolerance มั้ย
    is_always_in_band = True
    first_fail_idx = -1
    
    for i in range(N - M, -1, -1):
        window = clean_values[i : i + M]
        # เช็คว่ามีจุดไหนหลุด Band มั้ย
        max_diff = np.max(np.abs(window - final_target))
        
        if max_diff > tolerance_calc:
            settle_idx = i + M
            is_always_in_band = False
            first_fail_idx = i + M
            fail_reason = f"Window broke tolerance at index {i} (Max Diff: {max_diff:.4f} > Tol: {tolerance_calc:.4f})"
            break
            
    wait_time_ms = times[min(settle_idx, len(times)-1)]
    
    # 5. คำนวณ Features (เช็คว่า Crossing Rate ขึ้นมั้ย)
    tail_200 = values[-min(N, 200):]
    median_tail = np.median(tail_200)
    zero_crossings = len(np.where(np.diff(np.sign(tail_200 - median_tail)))[0])
    crossing_rate = zero_crossings / len(tail_200)

    print(f"\n--- Debug Report for Wave ID: {wave_id} ---")
    print(f"Final Target: {final_target:.4f}")
    print(f"Calculated Tolerance (10% or 5*SD): {tolerance_calc:.4f}")
    print(f"Max Amplitude Swing (in tail): {np.ptp(tail_200)/2:.4f}")
    print(f"Crossing Rate: {crossing_rate:.2f} (Should be > 0.05 for Sine)")
    print(f"Did it settle at 0ms? : {'YES' if wait_time_ms == 0 else 'NO'}")
    print(f"Calculated Wait Time: {wait_time_ms} ms")
    if wait_time_ms > 0:
        print(f"WHY? -> {fail_reason}")
    print("-" * 40)

# --- เรียกใช้งาน ---
# โหลดข้อมูล Raw ของคุณมา
df = pd.read_csv('data/raw/data_for_train_new_ver.csv') # หรือไฟล์ที่คุณเพิ่ง Gen

# ใส่ ID ที่คุณสงสัยลงไปตรงนี้ (เช่น ID ที่เป็น Sine หรือ Spike)
problem_ids = [83, 81, 65, 73] 

for wid in problem_ids:
    subset = df[df['wave_id'] == wid]
    if not subset.empty:
        debug_label_logic(subset['value'].values, subset['time_ms'].values, wid)
    else:
        print(f"Wave ID {wid} not found.")