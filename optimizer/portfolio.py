"""
Smart Portfolio Optimizer
- Mean-Variance (Markowitz)
- Risk Parity
- Regime-dependent asset allocation
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    import cvxpy as cp
except ImportError:
    cp = None

from config import REGIME_LABELS


class PortfolioOptimizer:
    """Regime-aware portfolio optimization."""

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self._regime_weights = self._default_regime_weights()

    def _default_regime_weights(self) -> dict:
        """Default allocation by regime (asset classes: equity, bonds, gold, cash)."""
        return {
            0: {"equity": 0.7, "bonds": 0.2, "gold": 0.05, "cash": 0.05},   # Bull
            1: {"equity": 0.3, "bonds": 0.5, "gold": 0.1, "cash": 0.1},     # Bear
            2: {"equity": 0.4, "bonds": 0.35, "gold": 0.15, "cash": 0.1},   # High Vol
            3: {"equity": 0.2, "bonds": 0.2, "gold": 0.3, "cash": 0.3},     # Crisis
        }

    def mean_variance(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        target_return: Optional[float] = None,
        long_only: bool = True,
    ) -> np.ndarray:
        """Markowitz mean-variance optimization."""
        if cp is None:
            raise ImportError("cvxpy required. pip install cvxpy")
        n = len(mu)
        w = cp.Variable(n)
        if target_return is not None:
            constraints = [
                cp.sum(w) == 1,
                mu @ w >= target_return,
            ]
        else:
            constraints = [cp.sum(w) == 1]
        if long_only:
            constraints.append(w >= 0)
        obj = cp.Minimize(cp.quad_form(w, Sigma))
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.ECOS)
        return np.array(w.value).flatten()

    def risk_parity(self, Sigma: np.ndarray) -> np.ndarray:
        """Equal risk contribution weights."""
        n = Sigma.shape[0]
        w = cp.Variable(n, nonneg=True)
        rc = cp.multiply(w, Sigma @ w) / cp.quad_form(w, Sigma)
        # Minimize sum of squared differences from equal RC
        target_rc = 1 / n
        obj = cp.Minimize(cp.sum_squares(rc - target_rc))
        prob = cp.Problem(obj, [cp.sum(w) == 1])
        prob.solve(solver=cp.ECOS)
        return np.array(w.value).flatten()

    def min_volatility(self, Sigma: np.ndarray, long_only: bool = True) -> np.ndarray:
        """Minimum variance portfolio."""
        n = Sigma.shape[0]
        w = cp.Variable(n)
        constraints = [cp.sum(w) == 1]
        if long_only:
            constraints.append(w >= 0)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
        prob.solve(solver=cp.ECOS)
        return np.array(w.value).flatten()

    def max_sharpe(self, mu: np.ndarray, Sigma: np.ndarray, rf: Optional[float] = None) -> np.ndarray:
        """Maximum Sharpe ratio (tangency portfolio)."""
        rf = rf or self.risk_free_rate
        mu_excess = mu - rf / 252  # Daily
        n = len(mu)
        w = cp.Variable(n, nonneg=True)
        constraints = [cp.sum(w) == 1]
        # Max Sharpe = min variance of excess return with fixed mean
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, Sigma)),
            constraints + [mu_excess @ w >= 1],
        )
        prob.solve(solver=cp.ECOS)
        w_raw = np.array(w.value).flatten()
        return w_raw / w_raw.sum()

    def black_litterman(
        self,
        mu_equilibrium: np.ndarray,
        Sigma: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        tau: float = 0.025,
        Omega: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Black-Litterman: blend equilibrium returns with views.
        P: pick matrix (n_views x n_assets), Q: view returns (n_views,)
        tau: scaling of prior uncertainty, Omega: view uncertainty (diagonal)"""
        if cp is None:
            raise ImportError("cvxpy required. pip install cvxpy")
        n = len(mu_equilibrium)
        tau_Sigma_inv = np.linalg.inv(tau * Sigma)
        if Omega is None:
            Omega = np.diag(np.diag(tau * P @ Sigma @ P.T))  # Idzorek approach
        Omega_inv = np.linalg.inv(Omega)
        # Posterior precision and mean
        post_prec = tau_Sigma_inv + P.T @ Omega_inv @ P
        post_mean = np.linalg.solve(
            post_prec,
            tau_Sigma_inv @ mu_equilibrium + P.T @ Omega_inv @ Q,
        )
        # Mean-variance with posterior returns
        return self.mean_variance(post_mean, Sigma, long_only=True)

    def cvar_minimize(
        self,
        returns: np.ndarray,
        alpha: float = 0.05,
        long_only: bool = True,
    ) -> np.ndarray:
        """Minimize CVaR (Expected Shortfall). alpha = tail probability (e.g. 0.05 for 95% VaR).
        Rockafellar-Uryasev formulation. returns: (T x n) historical returns."""
        if cp is None:
            raise ImportError("cvxpy required. pip install cvxpy")
        T, n = returns.shape
        w = cp.Variable(n)
        u = cp.Variable(T)
        var = cp.Variable()
        constraints = [cp.sum(w) == 1]
        if long_only:
            constraints.append(w >= 0)
        # u_t >= -r_t'w - VaR (excess shortfall)
        constraints += [u >= -returns @ w - var, u >= 0]
        # CVaR = VaR + (1/(alpha*T)) * sum(u)
        alpha = max(alpha, 1e-6)
        cvar = var + cp.sum(u) / (alpha * T)
        prob = cp.Problem(cp.Minimize(cvar), constraints)
        prob.solve(solver=cp.ECOS)
        return np.array(w.value).flatten()

    def regime_weights(self, regime: int) -> dict:
        """Get allocation by regime."""
        return self._regime_weights.get(regime, self._regime_weights[0])

    def optimize_by_regime(
        self,
        prices: pd.DataFrame,
        regime: pd.Series,
        asset_classes: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Regime-dependent optimization across asset classes."""
        ret = prices.pct_change().dropna()
        regimes = regime.reindex(ret.index).ffill().bfill().fillna(0).astype(int)
        cols = list(prices.columns)
        n = len(cols)
        weights = []
        for t in ret.index:
            r = min(int(regimes.loc[t]), 3)
            alloc = self.regime_weights(r)
            # Map by column names: equity=indices/etfs, bonds=bond tickers, gold=gold, cash=TLT as proxy
            w = np.zeros(n)
            if asset_classes:
                for ac, pct in alloc.items():
                    indices = asset_classes.get(ac, list(range(n)))
                    if isinstance(indices, list) and indices:
                        count = sum(1 for i in indices if i < n)
                        for i in indices:
                            if i < n:
                                w[i] = pct / max(count, 1)
            if w.sum() < 0.99:
                w = np.ones(n) / n
            w = w / w.sum()
            weights.append(w)
        return pd.DataFrame(weights, index=ret.index, columns=cols)

    def get_signals(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        momentum: pd.DataFrame,
        regime: int,
    ) -> pd.DataFrame:
        """Buy/Hold/Sell signals. weights=optimal, momentum=12m return."""
        cols = list(prices.columns)
        signals = []
        w_latest = weights.iloc[-1] if len(weights) > 0 else pd.Series(1 / len(cols), index=cols)
        mom = momentum.iloc[-1] if len(momentum) > 0 else pd.Series(0, index=cols)
        for i, c in enumerate(cols):
            w = w_latest.get(c, 0)
            m = mom.get(c, 0)
            # High weight + positive momentum = Buy; low weight + negative = Sell
            if w > 0.15 and m > 0.05:
                sig = "Buy"
                reason = "Strong allocation + momentum"
            elif w > 0.1 and m > 0:
                sig = "Hold"
                reason = "Adequate allocation"
            elif w < 0.05 or m < -0.1:
                sig = "Sell"
                reason = "Underweight or weak momentum"
            else:
                sig = "Hold"
                reason = "Neutral"
            signals.append({"ticker": c, "signal": sig, "reason": reason, "weight": w, "momentum": m})
        return pd.DataFrame(signals)
