"""Exploratory study — LLM domain (draft).

Question: does latent-collapse occupancy in the first K tokens discriminate
sessions that degenerate (finish_reason = "length") from sessions that
conclude (finish_reason = "stop")?

Mandatory baseline: mean surprisal over the same K tokens (both causal,
same window). Permutation null on AUC.

STATUS: exploratory (AS-1 §8). Confirmatory claims require the P1
pre-registration in PREREGISTRATION_P1.md to be committed first.

Usage:
    python scripts/latent_llm_test.py <sessions_dir>
"""

import sys

import numpy as np

sys.path.insert(0, "src")
from PRAMA_Protokol.ingest import load_sessions
from PRAMA_Protokol.omega import omega_sessions

# Kernel: the certified package. Never a local copy (AS-1 P7).
from prama_protokol import KernelConfig, project
from prama_protokol import compliance

# ---------------------------------------------------------------------------
# Kernel configuration — P1 DECISION D3 (see PREREGISTRATION_P1.md).
# Empirical constraint (synthetic sweep, 2026-07): tau_memory must be small
# relative to the evaluation window; the validated grid values (336/24) are
# MECHANICALLY inapplicable at K=256 token windows (accumulator never reaches
# regime; AUC collapses to 0.22). Robust plateau: tau in [16, 64] (AUC .90-.98
# for all g_smooth in {8,16,24}). Default below sits on that plateau. Any
# change is a D3 amendment, never a silent edit.
# ---------------------------------------------------------------------------
CFG = KernelConfig(tau_memory=64.0, g_smooth=16)
K = 256                       # evaluation window (tokens) — P1 decision D4
MIN_TOKENS = 32               # minimum evaluable session length — P1 decision D4

data_dir = sys.argv[1]
df = load_sessions(data_dir)
print(f"tokens: {len(df)}  sesiones: {df['session_id'].nunique()}")

lat_occ, triv, y = [], [], []
all_omega, all_delta = [], []
for sid, fr, omega, expected in omega_sessions(df):
    omega_k, expected_k = omega[:K], expected[:K]
    if len(omega_k) < MIN_TOKENS:
        continue
    gamma = project(omega_k, expected_k, CFG)
    valid = gamma["valid"].to_numpy()
    lat = gamma["latent_collapse"].to_numpy()
    lat_occ.append(lat[valid].mean() if valid.any() else 0.0)
    triv.append(omega_k.mean())
    y.append(1 if fr == "length" else 0)
    all_omega.append(omega_k[valid])
    all_delta.append(gamma["delta"].to_numpy()[valid])

lat_occ, triv, y = map(np.array, (lat_occ, triv, y))
print(f"evaluables: {len(y)}  length: {y.sum()}  stop: {(y == 0).sum()}")

# --- Compliance record BEFORE trusting any output (AS-1 §8) ---------------
c3 = compliance.check_degeneration(np.concatenate(all_delta), np.concatenate(all_omega))
print(f"[{'PASS' if c3['passed'] else 'FAIL'}] {c3['check']}: {c3['detail']}")
if not c3["passed"]:
    print("Δ degenerated into activity — study invalid (NYISO failure mode). Aborting.")
    sys.exit(1)


def ratio(score, y):
    thr = np.quantile(score, 0.75)
    hi, lo = y[score > thr], y[score <= thr]
    p_hi = hi.mean() if len(hi) else 0.0
    p_lo = lo.mean() if len(lo) else 0.0
    return p_hi / max(p_lo, 1e-9), p_hi, p_lo


def auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()


for name, s in [("latente", lat_occ), ("trivial", triv)]:
    r, p_hi, p_lo = ratio(s, y)
    print(f"{name:>8}: P(length|alto)={p_hi:.3f} P(length|bajo)={p_lo:.3f} "
          f"ratio={r:.2f}  AUC={auc(s, y):.3f}")

rng = np.random.default_rng(0)
null = []
yy = y.copy()
for _ in range(1000):
    rng.shuffle(yy)
    null.append(auc(lat_occ, yy))
p = (np.array(null) >= auc(lat_occ, y)).mean()
print(f"perm p (AUC latente): {p:.4f}")
