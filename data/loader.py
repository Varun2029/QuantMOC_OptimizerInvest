"""
Data Layer - Multi-market (India, USA, UK) via yfinance
"""

import pandas as pd
from pathlib import Path
from typing import Optional

try:
    import yfinance as yf
except ImportError:
    yf = None

from config import (
    DATA_DIR,
    MARKETS,
    MARKET_TICKERS,
    FALLBACK_TICKERS,
    MACRO_TICKERS,
    DEFAULT_START,
    DEFAULT_END,
)


class DataLoader:
    """Multi-market data loader."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir or DATA_DIR)
        self.cache_dir.mkdir(exist_ok=True)

    def _fetch_single(self, ticker: str, start: str, end: str) -> Optional[pd.Series]:
        try:
            if yf is None:
                raise ImportError("yfinance required")
            t = yf.Ticker(ticker)
            df = t.history(start=start, end=end, auto_adjust=True)
            if df.empty or "Close" not in df.columns:
                return None
            return df["Close"].rename(ticker)
        except Exception:
            return None

    def _fetch_batch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        series_list = []
        for t in tickers:
            s = self._fetch_single(t, start, end)
            if s is not None and len(s) > 0:
                series_list.append(s)
        if not series_list:
            return pd.DataFrame()
        return pd.concat(series_list, axis=1).sort_index().ffill().bfill()

    def load_market(
        self,
        market: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load data for a specific market (india, usa, uk)."""
        start = start or DEFAULT_START
        end = end or DEFAULT_END
        tickers = MARKET_TICKERS.get(market, FALLBACK_TICKERS)
        cache_file = self.cache_dir / f"market_{market}.parquet"
        if use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                df = df.loc[start:end] if start and end else df
                if not df.empty and len(df) >= 50:
                    return df
            except Exception:
                pass
        df = self._fetch_batch(tickers, start, end)
        if df.empty:
            df = self._fetch_batch(FALLBACK_TICKERS, start, end)
        if not df.empty and use_cache:
            try:
                df.to_parquet(cache_file)
            except Exception:
                pass
        return df

    def load_volatility(self, market: str, start: str, end: str) -> pd.Series:
        vix_ticker = MARKETS.get(market, {}).get("volatility", "^VIX")
        s = self._fetch_single(vix_ticker, start, end)
        if s is not None and len(s) > 0:
            return s.ffill().bfill()
        return pd.Series(20, index=pd.DatetimeIndex([]))

    def load_macro(self, start: str, end: str) -> pd.DataFrame:
        return self._fetch_batch(list(MACRO_TICKERS.values()), start, end)

    def get_universe(self, market: str, start: str, end: str):
        """Market + vol + macro for pipeline."""
        market_df = self.load_market(market, start, end, use_cache=False)
        if market_df.empty or len(market_df) < 100:
            return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()
        vol = self.load_volatility(market, start, end)
        macro = self.load_macro(start, end)
        idx = market_df.index
        vol = vol.reindex(idx).ffill().bfill() if not vol.empty else pd.Series(20, index=idx)
        macro = macro.reindex(idx).ffill().bfill() if not macro.empty else pd.DataFrame(index=idx)
        return market_df, vol, macro

    def get_quote(self, ticker: str) -> Optional[dict]:
        """Current quote for a ticker."""
        try:
            if yf is None:
                return None
            t = yf.Ticker(ticker)
            info = t.info
            return {
                "price": info.get("regularMarketPrice") or info.get("previousClose"),
                "change": info.get("regularMarketChange"),
                "change_pct": info.get("regularMarketChangePercent"),
            }
        except Exception:
            return None
