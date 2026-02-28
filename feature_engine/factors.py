"""
Factor Engine - Fama-French inspired signals
- Momentum: r_12m - r_1m (skip 1 month)
- Volatility: rolling std, drawdown
- Value: P/E relative ranking (proxy via price momentum inverse)
- Risk factors: size, value, quality, momentum
"""

import pandas as pd
import numpy as np
from typing import Optional


class FactorEngine:
    """Compute alpha signals and risk factors."""

    def __init__(self, window_short: int = 21, window_long: int = 252):
        self.window_short = window_short
        self.window_long = window_long

    def momentum_12_1(self, prices: pd.DataFrame) -> pd.DataFrame:
        """r_12m - r_1m (skip most recent month to avoid reversal)."""
        ret_12 = prices.pct_change(252)
        ret_1 = prices.pct_change(21)
        return ret_12 - ret_1

    def momentum_1m(self, prices: pd.DataFrame) -> pd.DataFrame:
        """1-month momentum."""
        return prices.pct_change(21)

    def momentum_12m(self, prices: pd.DataFrame) -> pd.DataFrame:
        """12-month momentum."""
        return prices.pct_change(252)

    def volatility_rolling(self, prices: pd.DataFrame, window: int = 21) -> pd.DataFrame:
        """Annualized rolling std of returns."""
        ret = prices.pct_change()
        vol = ret.rolling(window).std() * np.sqrt(252)
        return vol

    def drawdown_pct(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Peak-to-trough drawdown in %."""
        rolling_max = prices.rolling(min_periods=1, window=len(prices)).max()
        return (prices - rolling_max) / rolling_max

    def value_factor_proxy(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Value proxy: inverse of momentum (cheap = underperformed)."""
        mom = self.momentum_12m(prices)
        return -mom  # Lower momentum = higher value score

    def quality_factor_proxy(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Quality proxy: low volatility + positive momentum."""
        vol = self.volatility_rolling(prices, 63)
        mom = self.momentum_12m(prices)
        vol_rank = vol.rank(axis=1, pct=True)
        mom_rank = mom.rank(axis=1, pct=True)
        quality = (1 - vol_rank) * 0.5 + mom_rank * 0.5
        return quality

    def compute_all_factors(self, prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Compute all factors for portfolio construction."""
        return {
            "momentum_12_1": self.momentum_12_1(prices),
            "momentum_12m": self.momentum_12m(prices),
            "volatility": self.volatility_rolling(prices, self.window_short),
            "drawdown": self.drawdown_pct(prices),
            "value": self.value_factor_proxy(prices),
            "quality": self.quality_factor_proxy(prices),
        }
