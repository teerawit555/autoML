# scripts/generate_predict_sample.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from generate_train_sample import (
    generate_step_response,
    generate_high_start_oscillation,
    generate_continuous_triangular_pulses,
    generate_low_swing_sine_wave,
    generate_overdamped_decay,
    generate_pulse_train,
)


def build_generation_plan(n_waves: int, ratios, rng: np.random.Generator):
    plan = []
    allocated = 0

    for func, r in ratios:
        cnt = int(n_waves * float(r))
        plan.extend([func] * cnt)
        allocated += cnt

    remainder = n_waves - allocated
    if remainder > 0:
        plan.extend([ratios[0][0]] * remainder)

    rng.shuffle(plan)
    return plan


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic PREDICT waveform data (no labels).")
    ap.add_argument("--out", default="data/raw/data_predict.csv")
    ap.add_argument("--n_waves", type=int, default=200)
    ap.add_argument("--dt_ms", type=float, default=0.01)
    ap.add_argument("--t_end_ms", type=float, default=9.9)
    ap.add_argument("--predict_noise_scale", type=float, default=0.6)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_ms = np.arange(0.0, args.t_end_ms + 1e-12, args.dt_ms)
    t_s = t_ms / 1000.0

    ratios = [
        (generate_step_response,               0.25),  # Type 0
        (generate_high_start_oscillation,      0.25),  # Type 1
        (generate_overdamped_decay,            0.18),  # Type 4
        (generate_pulse_train,                 0.14),  # Type 5
        (generate_continuous_triangular_pulses,0.10),  # Type 2
        (generate_low_swing_sine_wave,         0.08),  # Type 3
    ]

    master_rng = np.random.default_rng(20240107)
    gen_sequence = build_generation_plan(args.n_waves, ratios, master_rng)

    rows = []
    print(f"Generating PREDICT dataset ({args.n_waves} waves)...")

    for wave_id, gen_func in enumerate(gen_sequence, start=1):
        # ---- waveform-level parameters ----
        final_value = float(master_rng.uniform(0.5, 3.5))
        band_pct = float(master_rng.uniform(0.05, 0.15))
        band = final_value * band_pct

        low = final_value - band / 2.0
        high = final_value + band / 2.0

        settle_time_ms = float(master_rng.uniform(2.0, 8.0))
        settle_s = settle_time_ms / 1000.0

        wave_rng = np.random.default_rng(500000 + wave_id)

        y, used_sd, _, _ = gen_func(
            t_s, final_value, settle_s, low, high, wave_rng
        )

        # ---- predict-only disturbance ----
        extra_rng = np.random.default_rng(900000 + wave_id)
        extra_sd = max(float(used_sd) * args.predict_noise_scale, 1e-6)
        y = y + extra_rng.normal(0.0, extra_sd, size=len(y))

        # ---- export ----
        for i, (tm, val) in enumerate(zip(t_ms, y)):
            row = {
                "wave_id": wave_id,
                "sample": i,
                "time_ms": float(tm),
                "value": float(val),
                "low_limit": float(low),
                "high_limit": float(high),
            }


            rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved PREDICT data to: {out_path}")


if __name__ == "__main__":
    main()
