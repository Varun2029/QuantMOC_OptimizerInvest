"""
Backtesting Engine
- Sharpe, Sortino, Max Drawdown, CAGR
- Rolling Sharpe
- Regime-wise performance
- Transaction costs, slippage, turnover control
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

from config import (
    INITIAL_CAPITAL,
    RISK_FREE_RATE,
    TRANSACTION_COST_BPS,
    SLIPPAGE_BPS,
    TURNOVER_TARGET,
)


@dataclass
class BacktestMetrics:
    sharpe: float
    sortino: float
    max_drawdown: float
    cagr: float
    total_return: float
    volatility: float
    turnover: float
    costs_pct: float


class Backtester:
    """Production-style backtester with costs and slippage."""

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        risk_free_rate: float = RISK_FREE_RATE,
        cost_bps: float = TRANSACTION_COST_BPS,
        slippage_bps: float = SLIPPAGE_BPS,
        turnover_target: float = TURNOVER_TARGET,
    ):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.cost_bps = cost_bps / 1e4
        self.slippage_bps = slippage_bps / 1e4
        self.turnover_target = turnover_target

    def run(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        regime: Optional[pd.Series] = None,
    ) -> tuple[pd.Series, BacktestMetrics]:
        """Run backtest. weights: (T x N), prices: (T x N)."""
        weights = weights.reindex(prices.index).ffill().bfill().fillna(0)
        ret = prices.ffill().pct_change(fill_method=None).dropna()
        weights = weights.reindex(ret.index).ffill().bfill().fillna(0)
        weights = weights / weights.sum(axis=1).values.reshape(-1, 1)

        port_ret = (ret * weights.shift(1)).sum(axis=1)
        turnover = weights.diff().abs().sum(axis=1)
        costs = turnover * (self.cost_bps + self.slippage_bps)
        port_ret = port_ret - costs
        port_ret = port_ret.fillna(0)

        equity = self.initial_capital * (1 + port_ret).cumprod()
        total_ret = equity.iloc[-1] / self.initial_capital - 1
        n_years = len(equity) / 252
        cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1 if n_years > 0 else 0

        excess = port_ret - self.risk_free_rate / 252
        sharpe = excess.mean() / port_ret.std() * np.sqrt(252) if port_ret.std() > 0 else 0
        downside = port_ret[port_ret < 0].std()
        sortino = (port_ret.mean() * 252 - self.risk_free_rate) / (downside * np.sqrt(252)) if downside and downside > 0 else 0

        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd = float(drawdown.min())

        avg_turnover = turnover.mean()
        total_costs = costs.sum()
        costs_pct = total_costs  # As fraction of capital

        metrics = BacktestMetrics(
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_dd,
            cagr=cagr,
            total_return=total_ret,
            volatility=port_ret.std() * np.sqrt(252),
            turnover=avg_turnover,
            costs_pct=costs_pct,
        )
        return equity, metrics

    def regime_performance(
        self,
        equity: pd.Series,
        regime: pd.Series,
        returns: pd.Series,
    ) -> pd.DataFrame:
        """Performance by regime."""
        regime = regime.reindex(returns.index).ffill().bfill()
        from config import REGIME_LABELS
        rows = []
        for r in regime.dropna().unique():
            mask = regime == r
            ret_r = returns[mask]
            if ret_r.empty:
                continue
            sharpe_r = ret_r.mean() / ret_r.std() * np.sqrt(252) if ret_r.std() > 0 else 0
            rows.append({"regime": REGIME_LABELS.get(r, str(r)), "sharpe": sharpe_r, "days": mask.sum()})
        return pd.DataFrame(rows)
