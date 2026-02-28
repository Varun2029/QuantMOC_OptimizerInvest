"""
Hedge Fund Style System - Main Pipeline
Multi-market: India, USA, UK
"""

import pandas as pd
import numpy as np

from config import DEFAULT_START, DEFAULT_END, MARKET_TICKERS, FALLBACK_TICKERS
from data.loader import DataLoader
from feature_engine.factors import FactorEngine
from regime_model.hmm_regime import RegimeDetector
from optimizer.portfolio import PortfolioOptimizer
from risk_engine.monte_carlo import RiskEngine
from backtester.engine import Backtester


def main(
    market: str = "usa",
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
):
    loader = DataLoader()
    market_df, vol, macro = loader.get_universe(market, start, end)
    if market_df.empty or len(market_df) < 100:
        np.random.seed(42)
        dates = pd.date_range(start, end, freq="B")
        tickers = MARKET_TICKERS.get(market, FALLBACK_TICKERS)[:5]
        market_df = pd.DataFrame(
            np.random.randn(len(dates), len(tickers)).cumsum(axis=0) * 0.0005 + 1,
            index=dates,
            columns=tickers[:5] if len(tickers) >= 5 else ["A", "B", "C", "D", "E"],
        )
        vol = pd.Series(20, index=dates)
        macro = pd.DataFrame(index=dates)

    if hasattr(market_df.columns, "get_level_values"):
        market_df.columns = [c[-1] if isinstance(c, tuple) else c for c in market_df.columns]
    market_df = market_df.loc[:, ~market_df.columns.duplicated()].dropna(how="all")
    if market_df.empty:
        market_df = pd.DataFrame(
            np.random.randn(500, 5).cumsum(axis=0) * 0.0005 + 1,
            index=pd.date_range(start, end, freq="B")[:500],
            columns=["EQ1", "EQ2", "BOND", "GOLD", "CASH"],
        )
        vol = pd.Series(20, index=market_df.index)

    fe = FactorEngine()
    factors = fe.compute_all_factors(market_df)
    returns = market_df.pct_change()
    regime_detector = RegimeDetector(n_regimes=4)
    regime_detector.fit(returns, vol, factors.get("drawdown"))
    regime = regime_detector.predict(returns, vol, factors.get("drawdown"))
    regime = regime.reindex(market_df.index).ffill().bfill().fillna(0)

    opt = PortfolioOptimizer()
    cols = list(market_df.columns)
    n = len(cols)
    asset_map = {
        "equity": list(range(min(3, n))),
        "bonds": [i for i in [2, 3] if i < n],
        "gold": [i for i in [4] if i < n],
        "cash": [],
    }
    weights = opt.optimize_by_regime(market_df, regime, asset_map)
    weights = weights / weights.sum(axis=1).values.reshape(-1, 1)

    bt = Backtester()
    equity, metrics = bt.run(market_df, weights, regime)
    port_ret = market_df.pct_change().mul(weights.shift(1)).sum(axis=1).dropna()
    regime_perf = bt.regime_performance(equity, regime, port_ret)

    risk = RiskEngine(n_sims=5000)
    w_avg = weights.mean().values
    ret_df = market_df.pct_change().dropna().reindex(columns=weights.columns).fillna(0)
    mc_result = risk.run(w_avg, ret_df, days=252, return_paths=True)
    rm, paths, _, max_dds = mc_result[0], mc_result[1], mc_result[2], mc_result[3]

    mom_12m = fe.momentum_12m(market_df)
    current_regime = int(regime.iloc[-1]) if len(regime) > 0 else 0
    signals = opt.get_signals(market_df, weights, mom_12m, current_regime)

    return {
        "market": market,
        "market_df": market_df,
        "equity": equity,
        "regime": regime,
        "weights": weights,
        "metrics": metrics,
        "risk_metrics": rm,
        "mc_paths": paths,
        "mc_max_dds": max_dds,
        "regime_perf": regime_perf,
        "signals": signals,
        "current_regime": current_regime,
    }
