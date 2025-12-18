# scripts/generate_predict_sample.py
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
 
def make_wave(t, final_value, settle_time_ms, low, high, seed=0):
    """
    Generate waveforms for Test: Focus on High Variance.
    Varying frequency, amplitude, noise, and damping to test model robustness.
    """
    rng = np.random.default_rng(seed)
 
    # 1. Randomize Physics Parameters (Distinct physics for every wave)
    # Frequency: Broad range 100Hz - 1200Hz
    freq_hz = rng.uniform(100, 1200)
    w = 2 * np.pi * freq_hz
    # Overshoot: Some waves shoot very high (20x), others low
    band_half = (high - low) / 2.0
    overshoot_scale = rng.uniform(1.5, 20.0) 
    direction = rng.choice([1, -1])  # Random direction (Up/Down)
    A0 = band_half * overshoot_scale * direction
 
    # Noise: From very quiet to heavy noise (0.001 - 0.04)
    sd = rng.uniform(0.001, 0.04)
 
    # 2. Calculation
    settle_s = settle_time_ms / 1000.0
    # Decay & Rise Time: Randomize graph shape to be unique
    decay_factor = rng.uniform(2.5, 6.0) 
    tau = max(settle_s / decay_factor, 1e-6)
 
    rise_factor = rng.uniform(3.0, 10.0)
    ts = max(settle_s / rise_factor, 1e-6)
 
    # Base Response
    base = final_value * (1 - np.exp(-t / ts))
 
    # Ringing
    ringing = A0 * np.exp(-t / tau) * np.sin(w * t)
 
    # Add Noise
    noise = rng.normal(0.0, sd, size=len(t))
 
    y = base + ringing + noise
 
    # 3. Force Settle (Ensure valid ground truth for verification)
    settled_mask = (t >= settle_s)
    final_noise = rng.normal(0.0, sd * 0.5, size=settled_mask.sum())
    y[settled_mask] = final_value + final_noise
 
    return y, sd
 
def main():
    ap = argparse.ArgumentParser()
    # Default Output is for Test file
    ap.add_argument("--out", default="data/raw/data1000samples_test.csv")
    ap.add_argument("--n_waves", type=int, default=100) # Adjust default count as needed
    ap.add_argument("--dt_ms", type=float, default=0.01)
    ap.add_argument("--t_end_ms", type=float, default=9.9)
    args = ap.parse_args()
 
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
 
    t_ms = np.arange(0, args.t_end_ms + 1e-12, args.dt_ms)
    t_s = t_ms / 1000.0
 
    rows = []
    # Use separate Seed from Train file (Train uses 123) for fair testing
    rng = np.random.default_rng(999)
 
    print(f"🧪 Generating TEST Dataset ({args.n_waves} waves) with Savage Physics...")
 
    for wave_id in range(1, args.n_waves + 1):
        # 1. Random Target Value
        final_value = rng.uniform(0.5, 3.5)
        band = rng.uniform(0.05, 0.20)
        low, high = final_value - band/2, final_value + band/2
 
        # 2. Random Settle Time (Ground Truth)
        settle_time_ms = rng.uniform(2.0, 8.0)
 
        # (Optional) If you want to keep initial waves simple for sanity check, add logic here.
        # But for hard testing, let it be fully random.
        # 3. Generate Wave
        # *** Important: Use Seed Offset 5000 to avoid repeating Train graphs (which use 1000) ***
        y, used_sd = make_wave(
            t=t_s,
            final_value=final_value,
            settle_time_ms=settle_time_ms,
            low=low,
            high=high,
            seed=5000 + wave_id 
        )
 
        # 4. Save Data
        for i, (tm, val) in enumerate(zip(t_ms, y)):
            rows.append({
                "wave_id": wave_id,
                "sample": i,
                "time": float(tm),
                "value": float(val),
                "sd": float(used_sd),
                "low_limit": float(low),
                "high_limit": float(high),
                # Include ground truth (but do not let AI see this column during actual prediction)
                "wait_time_ms": float(settle_time_ms), 
            })
 
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"✅ Wrote Test Data: {out_path} (rows={len(df)})")
    print("✨ Test set is ready! Graphs are now wild and unpredictable.")
 
if __name__ == "__main__":
    main()