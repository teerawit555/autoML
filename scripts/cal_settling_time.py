import pandas as pd
import numpy as np
import os
import argparse

def calculate_settling_time(df_raw, threshold_percent=1.0):
    """
    Calculates settling time for each wave_id.
    Logic: The point where the signal enters and stays within (FinalValue +/- Threshold).
    """
    results = []
    wave_ids = df_raw['wave_id'].unique()

    for wid in wave_ids:
        wave_data = df_raw[df_raw['wave_id'] == wid].sort_values('time_ms')
        
        time = wave_data['time_ms'].values
        force = wave_data['force_mA'].values
        
        # 1. Define final value (average of last 10% of the signal)
        last_10_percent = int(len(force) * 0.1)
        final_value = np.mean(force[-last_10_percent:])
        
        # 2. Define tolerance band
        tolerance = abs(final_value * (threshold_percent / 100.0))
        lower_bound = final_value - tolerance
        upper_bound = final_value + tolerance
        
        # 3. Find settling point (search backwards)
        settle_idx = len(force) - 1
        for i in range(len(force) - 1, -1, -1):
            if force[i] < lower_bound or force[i] > upper_bound:
                # If it ever exits the band, the point after this was the settling point
                settle_idx = i + 1
                break
            if i == 0: # Signal was always within band
                settle_idx = 0
        
        # Ensure index doesn't exceed array
        settle_idx = min(settle_idx, len(time) - 1)
        settling_time = time[settle_idx]
        
        results.append({
            'wave_id': wid,
            'wait_time_ms': settling_time
        })
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Calculate settling time labels from raw data.")
    parser.add_argument("--input", default="data/raw/data1000samples.csv", help="Path to raw CSV")
    parser.add_argument("--output", default="data/processed/train/labels.csv", help="Path to save labels")
    parser.add_argument("--threshold", type=float, default=1.0, help="Tolerance threshold percentage (default: 1.0%)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    print(f"Processing {args.input}...")
    df_raw = pd.read_csv(args.input)
    
    labels_df = calculate_settling_time(df_raw, args.threshold)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    labels_df.to_csv(args.output, index=False)
    
    print(f"Done! Labels saved to: {args.output}")
    print(labels_df.head())

if __name__ == "__main__":
    main()

#python scripts/calculate_settling_time.py --input data/raw/data1000samples.csv --threshold 1.0