"""Kernel π — idéntico a Aptadynamic-VPA. Opera sobre delta ya construido."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ProjectionConfig:
    tau_memory: float = 64.0
    lambda_eq: float = 1.0
    lambda_recovery: float = 0.005
    lambda_min: float = 0.1
    theta_scale: float = 2.0
    kappa: float = 0.05
    g_smooth: int = 16


def project(delta: np.ndarray, cfg: ProjectionConfig = ProjectionConfig()) -> pd.DataFrame:
    n = len(delta)
    a = np.exp(-1.0 / cfg.tau_memory)
    xi = np.zeros(n)
    lam = np.full(n, cfg.lambda_eq)
    A = np.zeros(n)
    theta = np.zeros(n)
    theta[0] = cfg.theta_scale * cfg.lambda_eq

    for i in range(1, n):
        xi[i] = a * xi[i - 1] + (1 - a) * delta[i]
        excess = max(xi[i] - theta[i - 1], 0.0)
        A[i] = A[i - 1] + excess
        d_lam = -cfg.kappa * excess + cfg.lambda_recovery * (cfg.lambda_eq - lam[i - 1])
        lam[i] = np.clip(lam[i - 1] + d_lam, cfg.lambda_min, cfg.lambda_eq)
        theta[i] = cfg.theta_scale * lam[i]

    m = theta - xi
    g = np.gradient(pd.Series(m).rolling(cfg.g_smooth, min_periods=1).mean().to_numpy())
    latent = (m >= 0) & (g < 0)
    return pd.DataFrame({"delta": delta, "xi": xi, "lambda": lam,
                         "theta": theta, "M": m, "G": g, "latent": latent})
