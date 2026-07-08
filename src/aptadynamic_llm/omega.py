"""Observation interface O_D for the LLM domain (draft, exploratory).

Event = token. Observable stream omega = surprisal (-top1 logprob).

Causal expectation omega_hat (C2/C3): per position-bucket mean of surprisal
over previous generations of the same model. Strictly causal across
generations: statistics are updated only after a generation is emitted, with no
intra-generation leakage. Warm-up generations are consumed for statistics and
not emitted; buckets never seen before yield NaN (handled as invalid by the
kernel).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BUCKETS = np.array([0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])


def bucket_of(pos: np.ndarray) -> np.ndarray:
    return np.searchsorted(BUCKETS, pos, side="right") - 1


def omega_sessions(df: pd.DataFrame, min_sessions: int = 5):
    """Yield (generation_id, finish_reason, omega, expected) per generation.

    omega    : surprisal array in token order
    expected : strictly causal expectation per token (NaN where no causal
               statistic exists yet)
    """
    nb = len(BUCKETS)
    out = []
    group_col = "generation_id" if "generation_id" in df.columns else "session_id"
    for model, dfm in df.groupby("model", sort=False):
        sums = np.zeros(nb)
        counts = np.zeros(nb)
        seen = 0
        for gid, g in dfm.groupby(group_col, sort=False):
            g = g.sort_values("pos")
            b = bucket_of(g["pos"].to_numpy())
            s = g["surprisal"].to_numpy(dtype=float)
            if seen >= min_sessions:
                expected = np.where(counts[b] > 0, sums[b] / np.maximum(counts[b], 1), np.nan)
                out.append((gid, g["finish_reason"].iloc[0], s, expected))
            # update AFTER emitting - strict causality across generations
            np.add.at(sums, b, s)
            np.add.at(counts, b, 1)
            seen += 1
    return out
