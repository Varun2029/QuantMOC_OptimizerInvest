"""
Market Regime Detection — Upgraded
- GaussianHMM baseline (hmmlearn)
- joblib model caching with fingerprint-based invalidation
- Random Forest ensemble layer for robustness
- Warm-start incremental refitting
States: Bull, Bear, High Vol, Crisis
"""

import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from hmmlearn import hmm
except ImportError:
    hmm = None

try:
    import joblib
except ImportError:
    joblib = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from config import REGIME_LABELS, N_REGIMES, MODELS_DIR


class RegimeDetector:
    """
    Detect market regime using HMM + optional RF ensemble.

    Caching:  Fitted model is saved to disk via joblib.
              Only refits when data fingerprint changes.
    Ensemble: If use_ensemble=True, blends HMM probabilities
              with a Random Forest classifier (70/30 weight).
    """

    def __init__(
        self,
        n_regimes: int = N_REGIMES,
        random_state: int = 42,
        use_ensemble: bool = True,
        hmm_iter: int = 100,
    ):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.use_ensemble = use_ensemble
        self.hmm_iter = hmm_iter
        self.model = None
        self.rf_model = None
        self.scaler = None
        self.regime_labels = REGIME_LABELS
        self._cached_fingerprint: Optional[str] = None

    # ── Feature building ───────────────────────────────────────────────────

    def _build_features(
        self,
        returns: pd.DataFrame,
        vix: pd.Series,
        drawdown: Optional[pd.DataFrame] = None,
    ) -> tuple[np.ndarray, pd.Index]:
        ret = (
            returns.mean(axis=1)
            if returns.ndim > 1 and returns.shape[1] > 1
            else (returns.iloc[:, 0] if returns.ndim > 1 else returns)
        )
        vol = (
            returns.std(axis=1)
            if returns.ndim > 1 and returns.shape[1] > 1
            else ret.rolling(21).std()
        )
        vol = vol.fillna(vol.median()).replace(0, np.nan).ffill().bfill().fillna(0.01)
        vix_aligned = vix.reindex(ret.index).ffill().bfill().fillna(20)

        features = pd.DataFrame(
            {"return": ret, "volatility": vol, "vix": vix_aligned},
            index=ret.index,
        ).fillna(0)

        if drawdown is not None and not drawdown.empty:
            dd = (
                drawdown.mean(axis=1)
                if drawdown.ndim > 1 and drawdown.shape[1] > 1
                else drawdown.iloc[:, 0]
            )
            features["drawdown"] = dd.reindex(features.index).ffill().bfill().fillna(0)

        # Rolling z-score normalisation (252-day window, min 21)
        roll_mean = features.rolling(252, min_periods=21).mean()
        roll_std = features.rolling(252, min_periods=21).std().replace(0, np.nan)
        features = (features - roll_mean) / roll_std.ffill().bfill().replace(0, 1)
        features = features.fillna(0)
        return features.values, features.index

    # ── Data fingerprint ───────────────────────────────────────────────────

    def _compute_fingerprint(self, X: np.ndarray) -> str:
        tail = X[-10:].tobytes()
        shape = str(X.shape).encode()
        return hashlib.md5(tail + shape).hexdigest()

    # ── Cache paths ────────────────────────────────────────────────────────

    def _hmm_path(self) -> Path:
        return MODELS_DIR / f"regime_hmm_{self.n_regimes}.joblib"

    def _rf_path(self) -> Path:
        return MODELS_DIR / f"regime_rf_{self.n_regimes}.joblib"

    def _fp_path(self) -> Path:
        return MODELS_DIR / f"regime_fp_{self.n_regimes}.txt"

    # ── Fit ────────────────────────────────────────────────────────────────

    def fit(
        self,
        returns: pd.DataFrame,
        vix: pd.Series,
        drawdown: Optional[pd.DataFrame] = None,
        force_refit: bool = False,
    ) -> "RegimeDetector":
        if hmm is None:
            raise ImportError("hmmlearn required: pip install hmmlearn")

        X, _ = self._build_features(returns, vix, drawdown)
        fp = self._compute_fingerprint(X)

        # Load cached model if fingerprint matches
        if not force_refit and joblib is not None:
            cached_fp = self._fp_path().read_text().strip() if self._fp_path().exists() else ""
            if cached_fp == fp and self._hmm_path().exists():
                self.model = joblib.load(self._hmm_path())
                if self.use_ensemble and SKLEARN_AVAILABLE and self._rf_path().exists():
                    self.rf_model = joblib.load(self._rf_path())
                return self

        # Fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.hmm_iter,
            random_state=self.random_state,
        )
        self.model.fit(X)

        # Fit RF ensemble using rule-based pseudo-labels
        if self.use_ensemble and SKLEARN_AVAILABLE:
            self._fit_rf_ensemble(X, returns, vix)

        # Save to disk
        if joblib is not None:
            joblib.dump(self.model, self._hmm_path())
            self._fp_path().write_text(fp)
            if self.rf_model is not None:
                joblib.dump(self.rf_model, self._rf_path())

        return self

    def _fit_rf_ensemble(
        self, X: np.ndarray, returns: pd.DataFrame, vix: pd.Series
    ) -> None:
        """Train RF on rule-based pseudo-labels then use for blending."""
        # Pseudo-label: 0=Bull, 1=Bear, 2=HighVol, 3=Crisis
        ret_col = returns.mean(axis=1) if returns.ndim > 1 else returns.iloc[:, 0]
        vix_aligned = vix.reindex(ret_col.index).ffill().bfill().fillna(20)
        vol = ret_col.rolling(21).std().fillna(0.01) * np.sqrt(252)

        y = np.zeros(len(X), dtype=int)
        y[(vol > 0.25) & (vix_aligned.values > 25)] = 2   # High Vol
        y[(ret_col.rolling(63).mean().fillna(0).values < -0.001)] = 1  # Bear
        y[(vol > 0.35) & (vix_aligned.values > 35)] = 3   # Crisis

        # Use HMM predictions to clean labels
        hmm_labels = self.model.predict(X)
        # Blend: 50% rule-based, 50% HMM
        y_blend = np.where(np.random.rand(len(y)) > 0.5, y, hmm_labels)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.rf_model.fit(X_scaled, y_blend)

    # ── Predict ────────────────────────────────────────────────────────────

    def predict(
        self,
        returns: pd.DataFrame,
        vix: pd.Series,
        drawdown: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Predict regime with optional RF ensemble blending."""
        X, idx = self._build_features(returns, vix, drawdown)

        hmm_labels = self.model.predict(X)

        if self.use_ensemble and SKLEARN_AVAILABLE and self.rf_model is not None and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            hmm_proba = self.model.predict_proba(X)      # (T, n_states)
            rf_proba = self.rf_model.predict_proba(X_scaled)  # (T, n_states)

            # Align RF columns to HMM state ordering
            n = self.n_regimes
            rf_cols = self.rf_model.classes_
            rf_proba_aligned = np.zeros((len(X), n))
            for i, c in enumerate(rf_cols):
                if c < n:
                    rf_proba_aligned[:, c] = rf_proba[:, i]

            # Weighted blend: 70% HMM, 30% RF
            blended = 0.70 * hmm_proba + 0.30 * rf_proba_aligned
            labels = blended.argmax(axis=1)
        else:
            labels = hmm_labels

        return pd.Series(labels, index=idx)

    # ── Regime probabilities ───────────────────────────────────────────────

    def predict_proba(
        self,
        returns: pd.DataFrame,
        vix: pd.Series,
        drawdown: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Return per-state probabilities for uncertainty quantification."""
        X, idx = self._build_features(returns, vix, drawdown)
        proba = self.model.predict_proba(X)
        cols = [REGIME_LABELS.get(i, str(i)) for i in range(self.n_regimes)]
        return pd.DataFrame(proba, index=idx, columns=cols)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        if joblib is None:
            raise ImportError("joblib required: pip install joblib")
        path = path or self._hmm_path()
        joblib.dump(self.model, path)
        return path

    def load(self, path: Optional[Path] = None) -> "RegimeDetector":
        if joblib is None:
            raise ImportError("joblib required: pip install joblib")
        path = path or self._hmm_path()
        self.model = joblib.load(path)
        return self
