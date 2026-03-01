"""
Execution Simulator
- Transaction costs
- Slippage
- Turnover control / risk budgeting
"""

import pandas as pd
import numpy as np
from typing import Optional

from config import TRANSACTION_COST_BPS, SLIPPAGE_BPS, TURNOVER_TARGET


class ExecutionSimulator:
    """Simulate realistic execution with costs."""

    def __init__(
        self,
        cost_bps: float = TRANSACTION_COST_BPS,
        slippage_bps: float = SLIPPAGE_BPS,
        max_turnover: float = TURNOVER_TARGET,
    ):
        self.cost_bps = cost_bps / 1e4
        self.slippage_bps = slippage_bps / 1e4
        self.max_turnover = max_turnover

    def apply_turnover_limit(self, target_weights: pd.DataFrame, prev_weights: pd.DataFrame) -> pd.DataFrame:
        """Clip rebalancing to max turnover."""
        delta = target_weights - prev_weights
        total_turnover = delta.abs().sum(axis=1)
        scale = np.minimum(1, self.max_turnover / (total_turnover + 1e-10))
        new_weights = prev_weights + delta * scale.values.reshape(-1, 1)
        return new_weights / new_weights.sum(axis=1).values.reshape(-1, 1)

    def execution_cost(self, turnover: float) -> float:
        """Cost = (commission + slippage) * turnover."""
        return turnover * (self.cost_bps + self.slippage_bps)
