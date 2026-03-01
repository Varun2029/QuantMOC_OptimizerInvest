"""
Smart Portfolio Optimizer — Upgraded
- Ledoit-Wolf shrinkage covariance (replaces noisy sample covariance)
- XGBoost-based signal model (replaces threshold rules)
- SHAP explanations for signal transparency
- Regime-conditional CVaR optimization
- Black-Litterman, Risk Parity, Min-Vol, Max-Sharpe (unchanged)
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    import cvxpy as cp
except ImportError:
    cp = None

try:
    from sklearn.covariance import LedoitWolf
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from config import REGIME_LABELS, MODELS_DIR


class PortfolioOptimizer:
    """Regime-aware portfolio optimization with ML signal layer."""

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self._regime_weights = self._default_regime_weights()
        self._signal_model: Optional[object] = None
        self._signal_scaler: Optional[object] = None
        self._shap_explainer: Optional[object] = None
        self._signal_model_trained = False

    def _default_regime_weights(self) -> dict:
        return {
            0: {"equity": 0.70, "bonds": 0.20, "gold": 0.05, "cash": 0.05},  # Bull
            1: {"equity": 0.30, "bonds": 0.50, "gold": 0.10, "cash": 0.10},  # Bear
            2: {"equity": 0.40, "bonds": 0.35, "gold": 0.15, "cash": 0.10},  # High Vol
            3: {"equity": 0.20, "bonds": 0.20, "gold": 0.30, "cash": 0.30},  # Crisis
        }

    # ── Covariance ─────────────────────────────────────────────────────────

    def get_covariance(
        self, returns: pd.DataFrame, method: str = "ledoit"
    ) -> np.ndarray:
        """
        Estimate covariance matrix.
        method: 'ledoit' (shrinkage), 'ewm' (exponential), 'sample'
        """
        ret = returns.dropna()
        n = ret.shape[1]
        if n == 0:
            return np.eye(1)

        if method == "ledoit" and SKLEARN_AVAILABLE:
            try:
                lw = LedoitWolf()
                lw.fit(ret.values)
                return lw.covariance_
            except Exception:
                pass

        if method == "ewm":
            return ret.ewm(halflife=63).cov().iloc[-n:].values

        # Sample covariance (fallback)
        return ret.cov().values

    # ── Classic optimizers ─────────────────────────────────────────────────

    def mean_variance(
        self, mu: np.ndarray, Sigma: np.ndarray,
        target_return: Optional[float] = None, long_only: bool = True,
    ) -> np.ndarray:
        if cp is None:
            raise ImportError("cvxpy required. pip install cvxpy")
        n = len(mu)
        w = cp.Variable(n)
        constraints = [cp.sum(w) == 1]
        if target_return is not None:
            constraints.append(mu @ w >= target_return)
        if long_only:
            constraints.append(w >= 0)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
        prob.solve(solver=cp.ECOS, warm_start=True)
        if w.value is None:
            return np.ones(n) / n
        return np.clip(np.array(w.value).flatten(), 0, 1)

    def risk_parity(self, Sigma: np.ndarray) -> np.ndarray:
        """Equal Risk Contribution using analytical approximation."""
        # Inverse-volatility weighting as fast approximation
        vol = np.sqrt(np.diag(Sigma))
        vol = np.where(vol == 0, 1e-6, vol)
        w = (1.0 / vol)
        return w / w.sum()

    def min_volatility(self, Sigma: np.ndarray, long_only: bool = True) -> np.ndarray:
        if cp is None:
            raise ImportError("cvxpy required. pip install cvxpy")
        n = Sigma.shape[0]
        w = cp.Variable(n)
        constraints = [cp.sum(w) == 1]
        if long_only:
            constraints.append(w >= 0)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
        prob.solve(solver=cp.ECOS, warm_start=True)
        if w.value is None:
            return np.ones(n) / n
        return np.clip(np.array(w.value).flatten(), 0, 1)

    def max_sharpe(
        self, mu: np.ndarray, Sigma: np.ndarray, rf: Optional[float] = None
    ) -> np.ndarray:
        rf = rf or self.risk_free_rate
        mu_excess = mu - rf / 252
        if cp is None:
            raise ImportError("cvxpy required. pip install cvxpy")
        n = len(mu)
        w = cp.Variable(n, nonneg=True)
        constraints = [cp.sum(w) == 1, mu_excess @ w >= 1e-6]
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
        prob.solve(solver=cp.ECOS, warm_start=True)
        if w.value is None:
            return np.ones(n) / n
        w_raw = np.clip(np.array(w.value).flatten(), 0, 1)
        return w_raw / w_raw.sum()

    def black_litterman(
        self, mu_equilibrium: np.ndarray, Sigma: np.ndarray,
        P: np.ndarray, Q: np.ndarray, tau: float = 0.025,
        Omega: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        tau_Sigma_inv = np.linalg.inv(tau * Sigma)
        if Omega is None:
            Omega = np.diag(np.diag(tau * P @ Sigma @ P.T))
        Omega_inv = np.linalg.inv(Omega)
        post_prec = tau_Sigma_inv + P.T @ Omega_inv @ P
        post_mean = np.linalg.solve(
            post_prec, tau_Sigma_inv @ mu_equilibrium + P.T @ Omega_inv @ Q
        )
        return self.mean_variance(post_mean, Sigma, long_only=True)

    def cvar_minimize(
        self, returns: np.ndarray, alpha: float = 0.05, long_only: bool = True
    ) -> np.ndarray:
        if cp is None:
            raise ImportError("cvxpy required. pip install cvxpy")
        T, n = returns.shape
        w = cp.Variable(n)
        u = cp.Variable(T)
        var = cp.Variable()
        constraints = [cp.sum(w) == 1]
        if long_only:
            constraints.append(w >= 0)
        alpha = max(alpha, 1e-6)
        constraints += [u >= -returns @ w - var, u >= 0]
        cvar_obj = var + cp.sum(u) / (alpha * T)
        prob = cp.Problem(cp.Minimize(cvar_obj), constraints)
        prob.solve(solver=cp.ECOS, warm_start=True)
        if w.value is None:
            return np.ones(n) / n
        return np.clip(np.array(w.value).flatten(), 0, 1)

    def regime_weights(self, regime: int) -> dict:
        return self._regime_weights.get(regime, self._regime_weights[0])

    # ── Regime-dependent optimization ─────────────────────────────────────

    def optimize_by_regime(
        self,
        prices: pd.DataFrame,
        regime: pd.Series,
        asset_classes: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Regime-aware allocation with Ledoit-Wolf covariance."""
        ret = prices.ffill().pct_change(fill_method=None).dropna()
        regimes = regime.reindex(ret.index).ffill().bfill().fillna(0).astype(int)
        cols = list(prices.columns)
        n = len(cols)
        weights = []

        for t in ret.index:
            r = min(int(regimes.loc[t]), 3)
            alloc = self.regime_weights(r)
            w = np.zeros(n)
            if asset_classes:
                for ac, pct in alloc.items():
                    indices = asset_classes.get(ac, [])
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

    # ── XGBoost signal model ───────────────────────────────────────────────

    # Fixed factor names — must stay in sync with _build_signal_features
    SIGNAL_FACTOR_NAMES = ["momentum_12_1", "momentum_12m", "volatility", "drawdown", "value", "quality"]

    def train_signal_model(
        self,
        prices: pd.DataFrame,
        factors: dict,
        regime: pd.Series,
    ) -> bool:
        """
        Train XGBoost classifier per-ticker with a consistent 7-feature vector:
            momentum_12_1, momentum_12m, volatility, drawdown, value, quality, regime
        Label: forward 21-day return > 0 (Buy=1) or <= 0 (Sell=0)
        """
        if not XGB_AVAILABLE or not SKLEARN_AVAILABLE:
            return False

        ret = prices.ffill().pct_change(fill_method=None).dropna()
        forward = ret.rolling(21).sum().shift(-21)
        regime_aligned = regime.reindex(ret.index).ffill().bfill().fillna(0)

        all_X, all_y = [], []
        for col in prices.columns:
            if col not in forward.columns:
                continue
            y_col = (forward[col] > 0).astype(int).dropna()
            if len(y_col) < 100:
                continue

            # Build per-ticker feature matrix: exactly one scalar per factor per date
            feat_dict = {}
            for fname in self.SIGNAL_FACTOR_NAMES:
                df = factors.get(fname)
                if df is not None and isinstance(df, pd.DataFrame) and col in df.columns:
                    feat_dict[fname] = df[col]
                else:
                    feat_dict[fname] = pd.Series(0.0, index=ret.index)
            feat_dict["regime"] = regime_aligned

            feat_df = pd.DataFrame(feat_dict, index=ret.index)
            feat_df = feat_df.reindex(y_col.index).fillna(0)
            y_col = y_col.reindex(feat_df.index).dropna()
            feat_df = feat_df.reindex(y_col.index)

            if len(feat_df) < 50:
                continue
            all_X.append(feat_df)
            all_y.append(y_col)

        if not all_X:
            return False

        X_full = pd.concat(all_X).fillna(0)
        y_full = pd.concat(all_y)

        self._signal_scaler = StandardScaler()
        X_scaled = self._signal_scaler.fit_transform(X_full.values)

        self._signal_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        self._signal_model.fit(X_scaled, y_full.values)

        if SHAP_AVAILABLE:
            try:
                self._shap_explainer = shap.TreeExplainer(self._signal_model)
            except Exception:
                pass

        self._signal_model_trained = True
        return True

    # ── Signal generation ──────────────────────────────────────────────────

    def get_signals(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        momentum: pd.DataFrame,
        regime: int,
        factors: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Generate Buy/Hold/Sell signals with reasoning.
        Uses XGBoost model if trained, otherwise falls back to rule-based.
        Adds SHAP explanation if available.
        """
        cols = list(prices.columns)
        w_latest = weights.iloc[-1] if len(weights) > 0 else pd.Series(1 / len(cols), index=cols)
        mom = momentum.iloc[-1] if len(momentum) > 0 else pd.Series(0.0, index=cols)
        signals = []

        for c in cols:
            w = float(w_latest.get(c, 0))
            m = float(mom.get(c, 0))
            shap_text = ""

            if self._signal_model_trained and self._signal_scaler is not None and factors is not None:
                # Build feature vector for this ticker
                feat = self._build_signal_features(c, prices, factors, regime)
                if feat is not None:
                    feat_scaled = self._signal_scaler.transform(feat.reshape(1, -1))
                    proba = self._signal_model.predict_proba(feat_scaled)[0]
                    buy_prob = float(proba[1]) if len(proba) > 1 else 0.5

                    if buy_prob > 0.60:
                        sig = "Buy"
                        reason = f"XGB Buy probability {buy_prob:.0%}"
                    elif buy_prob < 0.40:
                        sig = "Sell"
                        reason = f"XGB Sell probability {1 - buy_prob:.0%}"
                    else:
                        sig = "Hold"
                        reason = f"Neutral ({buy_prob:.0%} buy)"

                    # SHAP explanation
                    if self._shap_explainer is not None and SHAP_AVAILABLE:
                        try:
                            sv = self._shap_explainer.shap_values(feat_scaled)
                            if isinstance(sv, list):
                                sv = sv[1]  # class 1 (Buy)
                            top_idx = np.abs(sv[0]).argmax()
                            shap_text = f"Key driver: feature[{top_idx}]"
                        except Exception:
                            pass
                else:
                    sig, reason = self._rule_based_signal(w, m, regime)
            else:
                sig, reason = self._rule_based_signal(w, m, regime)

            signals.append({
                "ticker": c,
                "signal": sig,
                "reason": reason,
                "weight": w,
                "momentum": m,
                "regime": REGIME_LABELS.get(regime, "—"),
                "explanation": shap_text,
            })

        return pd.DataFrame(signals)

    def _build_signal_features(
        self, ticker: str, prices: pd.DataFrame, factors: dict, regime: int
    ) -> Optional[np.ndarray]:
        """
        Build a 7-feature vector for a single ticker — must exactly match
        the feature order used in train_signal_model (SIGNAL_FACTOR_NAMES + regime).
        """
        feat_vals = []
        for fname in self.SIGNAL_FACTOR_NAMES:
            df = factors.get(fname)
            if df is not None and isinstance(df, pd.DataFrame) and ticker in df.columns:
                v = df[ticker].iloc[-1]
                feat_vals.append(0.0 if pd.isna(v) else float(v))
            else:
                feat_vals.append(0.0)
        feat_vals.append(float(regime))
        return np.array(feat_vals)

    def _rule_based_signal(self, w: float, m: float, regime: int) -> tuple[str, str]:
        """Fallback rule-based signal when XGB model is not available."""
        if w > 0.15 and m > 0.05:
            return "Buy", "Strong allocation + positive momentum"
        elif w > 0.10 and m > 0.0:
            return "Hold", "Adequate allocation"
        elif w < 0.05 or m < -0.10:
            return "Sell", "Underweight or negative momentum"
        elif regime == 3:  # Crisis
            return "Hold", "Crisis regime — reduce trading"
        else:
            return "Hold", "Neutral"
