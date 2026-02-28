"""
Monte Carlo Risk Engine
- VaR (95%)
- CVaR (Expected Shortfall)
- Max drawdown probability
- Probability of ruin
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass

from config import N_SIMULATIONS, VAR_CONFIDENCE


@dataclass
class RiskMetrics:
    var_95: float
    cvar_95: float
    max_dd_prob: float
    prob_ruin: float
    mean_return: float
    vol: float
    sharpe: float


class RiskEngine:
    """Monte Carlo simulation for portfolio risk."""

    def __init__(self, n_sims: int = N_SIMULATIONS, var_confidence: float = VAR_CONFIDENCE):
        self.n_sims = n_sims
        self.var_confidence = var_confidence

    def simulate_paths(
        self,
        mu: float,
        sigma: float,
        days: int = 252,
        n_paths: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate return paths using geometric Brownian motion."""
        n = n_paths or self.n_sims
        rng = np.random.default_rng(seed)
        dt = 1 / 252
        drift = (mu - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)
        Z = rng.standard_normal((n, days))
        log_returns = drift + vol * Z
        return np.exp(np.cumsum(log_returns, axis=1)) - 1  # Cumulative return

    def compute_var_cvar(self, returns: np.ndarray, confidence: Optional[float] = None) -> tuple[float, float]:
        """VaR and CVaR from return distribution."""
        conf = confidence or self.var_confidence
        var = np.percentile(returns, (1 - conf) * 100)
        cvar = returns[returns <= var].mean()
        if np.isnan(cvar):
            cvar = var
        return float(var), float(cvar)

    def max_drawdown_path(self, path: np.ndarray) -> float:
        """Max drawdown for a single path (cumulative returns)."""
        cum = 1 + path
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        return float(np.min(dd))

    def run(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        days: int = 252,
        ruin_level: float = -0.5,
        seed: Optional[int] = 42,
        return_paths: bool = False,
    ):
        """Full Monte Carlo risk analysis. If return_paths=True, returns (RiskMetrics, paths, daily_returns)."""
        # Portfolio return distribution from historical
        port_ret = (returns * weights).sum(axis=1).dropna()
        mu = port_ret.mean() * 252
        sigma = port_ret.std() * np.sqrt(252)

        paths = self.simulate_paths(mu, sigma, days=days, n_paths=self.n_sims, seed=seed)
        # Daily returns from paths
        daily_ret = np.diff(np.concatenate([np.ones((self.n_sims, 1)), 1 + paths], axis=1), axis=1) - 1
        flat_ret = daily_ret.flatten()

        var_95, cvar_95 = self.compute_var_cvar(flat_ret)
        max_dds = np.array([self.max_drawdown_path(p) for p in paths])
        max_dd_prob = float(np.mean(max_dds < ruin_level))
        prob_ruin = float(np.mean(np.any(1 + paths < 1 + ruin_level, axis=1)))

        rm = RiskMetrics(
            var_95=var_95,
            cvar_95=cvar_95,
            max_dd_prob=max_dd_prob,
            prob_ruin=prob_ruin,
            mean_return=mu,
            vol=sigma,
            sharpe=(mu - 0.05) / sigma if sigma > 0 else 0,
        )
        if return_paths:
            return rm, paths, daily_ret, max_dds
        return rm
