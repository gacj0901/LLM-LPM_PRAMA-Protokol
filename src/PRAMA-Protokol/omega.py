"""Observation interface O_D for the LLM domain (draft, exploratory).

Event = token. Observable stream ω = surprisal (−top1 logprob).

Causal expectation ω̂ (C2/C3): per position-bucket mean of surprisal over
*previous sessions* of the same model. Strictly causal across sessions
(statistics are updated only after a session is emitted); no intra-session
leakage. Warm-up: the first `min_sessions` sessions per model are consumed
for statistics and not emitted; buckets never seen before yield NaN
(handled as invalid by the kernel).

Design note (P1 decision, declared): "the system" whose expected behavior
Δ measures is the *deployed model*, hence the population-level ω̂. The
kernel computes Δ = |ω − ω̂|/(ω̂ + 1) itself — this module never computes Δ
(single source of truth: prama-protokol).

Empirical check (synthetic, 2026-07): this construction passes the C3
degeneration statistic (r_Δω ≈ −0.03 against degenerate baseline −0.06).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BUCKETS = np.array([0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])


def bucket_of(pos: np.ndarray) -> np.ndarray:
    return np.searchsorted(BUCKETS, pos, side="right") - 1


def omega_sessions(df: pd.DataFrame, min_sessions: int = 5):
    """Yield (session_id, finish_reason, omega, expected) per emitted session.

    omega    : surprisal array of the session (token order)
    expected : strictly causal ω̂ per token (NaN where no causal statistic
               exists yet — the kernel marks those rows invalid)
    """
    nb = len(BUCKETS)
    out = []
    for model, dfm in df.groupby("model", sort=False):
        sums = np.zeros(nb)
        counts = np.zeros(nb)
        seen = 0
        for sid, g in dfm.groupby("session_id", sort=False):
            g = g.sort_values("pos")
            b = bucket_of(g["pos"].to_numpy())
            s = g["surprisal"].to_numpy(dtype=float)
            if seen >= min_sessions:
                expected = np.where(counts[b] > 0, sums[b] / np.maximum(counts[b], 1), np.nan)
                out.append((sid, g["finish_reason"].iloc[0], s, expected))
            # update AFTER emitting — strict causality across sessions
            np.add.at(sums, b, s)
            np.add.at(counts, b, 1)
            seen += 1
    return out
