"""
Market Regime Detection - Hidden Markov Model
States: Bull, Bear, High Vol, Crisis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from hmmlearn import hmm
except ImportError:
    hmm = None

from config import REGIME_LABELS, N_REGIMES, MODELS_DIR


class RegimeDetector:
    """Detect market regime using HMM on returns + volatility features."""

    def __init__(self, n_regimes: int = 4, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.regime_labels = REGIME_LABELS

    def _build_features(
        self,
        returns: pd.DataFrame,
        vix: pd.Series,
        drawdown: Optional[pd.DataFrame] = None,
    ) -> tuple[np.ndarray, pd.Index]:
        """Features: return, vol, vix (scaled), drawdown. Returns (values, index)."""
        ret = returns.mean(axis=1) if returns.ndim > 1 and returns.shape[1] > 1 else (returns.iloc[:, 0] if returns.ndim > 1 else returns)
        vol = returns.std(axis=1) if returns.ndim > 1 and returns.shape[1] > 1 else ret.rolling(21).std()
        vol = vol.fillna(vol.median()).replace(0, np.nan).ffill().bfill().fillna(0.01)
        vix_aligned = vix.reindex(ret.index).ffill().bfill().fillna(20)
        features = pd.DataFrame({
            "return": ret,
            "volatility": vol,
            "vix": vix_aligned,
        }, index=ret.index).fillna(0)
        if drawdown is not None and not drawdown.empty:
            dd = drawdown.mean(axis=1) if drawdown.ndim > 1 and drawdown.shape[1] > 1 else drawdown.iloc[:, 0]
            features["drawdown"] = dd.reindex(features.index).ffill().bfill().fillna(0)
        # Standardize
        roll_mean = features.rolling(252, min_periods=21).mean()
        roll_std = features.rolling(252, min_periods=21).std().replace(0, np.nan)
        features = (features - roll_mean) / roll_std.ffill().bfill().replace(0, 1)
        features = features.fillna(0)
        # Drop rows with all-zero for fit, but predict needs full index
        return features.values, features.index

    def fit(
        self,
        returns: pd.DataFrame,
        vix: pd.Series,
        drawdown: Optional[pd.DataFrame] = None,
    ) -> "RegimeDetector":
        """Fit HMM on historical features."""
        if hmm is None:
            raise ImportError("hmmlearn required. pip install hmmlearn")
        X, _ = self._build_features(returns, vix, drawdown)
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=self.random_state,
        )
        self.model.fit(X)
        return self

    def predict(
        self,
        returns: pd.DataFrame,
        vix: pd.Series,
        drawdown: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Predict regime for each date."""
        X, idx = self._build_features(returns, vix, drawdown)
        labels = self.model.predict(X)
        return pd.Series(labels, index=idx)

    def save(self, path: Optional[Path] = None) -> Path:
        """Save fitted model."""
        import pickle
        path = path or MODELS_DIR / "regime_hmm.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        return path

    def load(self, path: Optional[Path] = None) -> "RegimeDetector":
        """Load fitted model."""
        import pickle
        path = path or MODELS_DIR / "regime_hmm.pkl"
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        return self
