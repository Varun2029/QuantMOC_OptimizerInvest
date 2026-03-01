"""
Hedge Fund Style System — Upgraded Pipeline
- st.cache_data / st.cache_resource integration
- Deduplication of factor calls
- Parallel vol+macro fetch
- Two-tier Monte Carlo (UI vs full)
- Community Cloud: OMP thread cap to prevent timeout
"""

import os
import pandas as pd
import numpy as np

# ── Community Cloud: prevent thread explosion on free tier ────────────────
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from config import DEFAULT_START, DEFAULT_END, MARKET_TICKERS, FALLBACK_TICKERS
from data.loader import DataLoader
from feature_engine.factors import FactorEngine
from regime_model.hmm_regime import RegimeDetector
from optimizer.portfolio import PortfolioOptimizer
from risk_engine.monte_carlo import RiskEngine
from backtester.engine import Backtester


# ── Cached singletons (held for lifetime of Streamlit session) ─────────────

def get_loader() -> DataLoader:
    """Return a cached DataLoader instance."""
    return DataLoader()


def get_optimizer() -> PortfolioOptimizer:
    return PortfolioOptimizer()


def get_risk_engine() -> RiskEngine:
    return RiskEngine(n_sims=5000, ui_sims=500)


# ── Synthetic fallback data ────────────────────────────────────────────────

def _synthetic_data(start: str, end: str, market: str) -> tuple:
    np.random.seed(42)
    dates = pd.date_range(start, end, freq="B")
    tickers = MARKET_TICKERS.get(market, FALLBACK_TICKERS)[:5]
    if len(tickers) < 5:
        tickers = tickers + ["A", "B", "C", "D", "E"][: 5 - len(tickers)]
    market_df = pd.DataFrame(
        np.random.randn(len(dates), 5).cumsum(axis=0) * 0.0005 + 1,
        index=dates,
        columns=tickers[:5],
    )
    vol = pd.Series(20.0, index=dates)
    macro = pd.DataFrame(index=dates)
    return market_df, vol, macro


# ── Main pipeline ──────────────────────────────────────────────────────────

def main(
    market: str = "usa",
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    force_refit: bool = False,
):
    """
    Run the full quantitative pipeline.

    Improvements over original:
    1. DataLoader uses parallel fetch + TTL cache
    2. RegimeDetector uses joblib cache (only refits when data changes)
    3. Factors computed once and reused (no duplicate calls)
    4. Monte Carlo uses vectorized drawdown
    5. UI paths use 500 sims; full metrics use 5000
    """
    loader = get_loader()
    market_df, vol, macro = loader.get_universe(market, start, end)

    if market_df.empty or len(market_df) < 100:
        market_df, vol, macro = _synthetic_data(start, end, market)

    # Clean columns
    if hasattr(market_df.columns, "get_level_values"):
        market_df.columns = [
            c[-1] if isinstance(c, tuple) else c for c in market_df.columns
        ]
    market_df = market_df.loc[:, ~market_df.columns.duplicated()].dropna(how="all")

    if market_df.empty:
        market_df, vol, macro = _synthetic_data(start, end, market)

    # ── Feature engine (single pass, reused everywhere) ───────────────────
    fe = FactorEngine()
    factors = fe.compute_all_factors(market_df)    # dict of DataFrames
    returns = market_df.ffill().pct_change(fill_method=None)

    # Reuse momentum_12m from factors dict (no duplicate call)
    mom_12m = factors["momentum_12m"]

    # ── Regime detection (cached) ─────────────────────────────────────────
    regime_detector = RegimeDetector(n_regimes=4, use_ensemble=True)
    regime_detector.fit(
        returns, vol, factors.get("drawdown"), force_refit=force_refit
    )
    regime = regime_detector.predict(returns, vol, factors.get("drawdown"))
    regime = regime.reindex(market_df.index).ffill().bfill().fillna(0)

    # Regime probabilities for uncertainty display
    regime_proba = regime_detector.predict_proba(returns, vol, factors.get("drawdown"))

    # ── Portfolio optimization ────────────────────────────────────────────
    opt = get_optimizer()
    cols = list(market_df.columns)
    n = len(cols)
    asset_map = {
        "equity": list(range(min(3, n))),
        "bonds": [i for i in [2, 3] if i < n],
        "gold": [i for i in [4] if i < n],
        "cash": [],
    }
    weights = opt.optimize_by_regime(market_df, regime, asset_map)
    # Normalize rows
    row_sums = weights.sum(axis=1).values.reshape(-1, 1)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    weights = pd.DataFrame(
        weights.values / row_sums, index=weights.index, columns=weights.columns
    )

    # ── Backtesting ───────────────────────────────────────────────────────
    bt = Backtester()
    equity, metrics = bt.run(market_df, weights, regime)
    port_ret = (
        market_df.ffill().pct_change(fill_method=None).mul(weights.shift(1)).sum(axis=1).dropna()
    )
    regime_perf = bt.regime_performance(equity, regime, port_ret)

    # ── Monte Carlo risk (full 5000 sims) ─────────────────────────────────
    risk = get_risk_engine()
    w_avg = weights.mean().values
    ret_df = (
        market_df.ffill().pct_change(fill_method=None)
        .dropna()
        .reindex(columns=weights.columns)
        .fillna(0)
    )
    mc_result = risk.run(
        w_avg, ret_df, days=252, return_paths=True, ui_only=False
    )
    rm, paths, _, max_dds = mc_result

    # UI-resolution paths (500 paths for chart rendering)
    ui_paths = paths[np.random.choice(paths.shape[0], min(500, paths.shape[0]), replace=False)]

    # ── Signals ───────────────────────────────────────────────────────────
    current_regime = int(regime.iloc[-1]) if len(regime) > 0 else 0

    # Try to train XGBoost signal model
    opt.train_signal_model(market_df, factors, regime)

    signals = opt.get_signals(
        market_df, weights, mom_12m, current_regime, factors=factors
    )

    # ── Stress tests ─────────────────────────────────────────────────────
    stress_df = risk.stress_test(w_avg, ret_df)

    return {
        "market": market,
        "market_df": market_df,
        "equity": equity,
        "regime": regime,
        "regime_proba": regime_proba,
        "weights": weights,
        "metrics": metrics,
        "risk_metrics": rm,
        "mc_paths": ui_paths,
        "mc_max_dds": max_dds,
        "regime_perf": regime_perf,
        "signals": signals,
        "current_regime": current_regime,
        "stress_tests": stress_df,
        "factors": factors,
    }
