"""Generate synthetic LLM sessions to validate the pipeline anywhere.

Two scenarios (choose with argv[1]):
  volume     — runaway sessions drift upward in mean surprisal
               (the trivial baseline SHOULD win here)
  structural — runaway sessions oscillate with preserved mean
               (loops/repetition signature; latent occupancy SHOULD win,
                the volume baseline is blind)

Usage: python examples/make_synthetic.py structural data_synth
"""
import json, sys
from pathlib import Path
import numpy as np

mode = sys.argv[1] if len(sys.argv) > 1 else "structural"
out = Path(sys.argv[2] if len(sys.argv) > 2 else f"data_{mode}")
out.mkdir(exist_ok=True)
rng = np.random.default_rng(7 if mode == "structural" else 42)

def base(n):
    pos = np.arange(n)
    return 3.0 + 2.0 * np.exp(-pos / 30) + rng.normal(0, 0.6, n)

for k in range(120):
    runaway = rng.random() < 0.35
    if runaway:
        n = int(rng.integers(400, 700))
        s = base(n)
        onset = int(rng.integers(60, 180))
        ramp = np.maximum(0, np.arange(n) - onset)
        if mode == "volume":
            s = s + ramp * rng.uniform(0.004, 0.012)
        else:
            amp = ramp * rng.uniform(0.010, 0.022)
            s = s + amp * np.sign(np.sin(np.arange(n) / rng.uniform(2.5, 4.5)))
        fr = "length"
    else:
        n = int(rng.integers(150, 500))
        s = base(n); fr = "stop"
    toks = [{"top1_logprob": -float(v), "entropy": 1.5, "gap": 0.8} for v in s]
    json.dump({"session_id": f"s{k:03d}", "model": "m1",
               "turns": [{"tokens": toks, "finish_reason": fr}]},
              open(out / f"s{k:03d}.json", "w"))
print(f"{mode}: 120 sessions -> {out}/")
