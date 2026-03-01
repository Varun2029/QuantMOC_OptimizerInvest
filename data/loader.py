"""
Data Layer - Multi-market (India, USA, UK) via yfinance
UPGRADED: Parallel fetching, TTL-based cache, bulk download
"""

import hashlib
import time
import pandas as pd
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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

CACHE_TTL_SECONDS = 4 * 3600  # 4 hours


class DataLoader:
    """Multi-market data loader with parallel fetching and TTL cache."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir or DATA_DIR)
        self.cache_dir.mkdir(exist_ok=True)

    # ── Cache helpers ──────────────────────────────────────────────────────

    def _cache_key(self, market: str, start: str, end: str) -> Path:
        slug = hashlib.md5(f"{market}_{start}_{end}".encode()).hexdigest()[:10]
        return self.cache_dir / f"market_{market}_{slug}.parquet"

    def _cache_valid(self, path: Path) -> bool:
        return path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS

    # ── Single ticker ──────────────────────────────────────────────────────

    def _fetch_single(self, ticker: str, start: str, end: str) -> Optional[pd.Series]:
        """Fetch single ticker with exponential-backoff retry for rate limits."""
        for attempt in range(3):
            try:
                if yf is None:
                    raise ImportError("yfinance required")
                t = yf.Ticker(ticker)
                df = t.history(start=start, end=end, auto_adjust=True)
                if df.empty or "Close" not in df.columns:
                    return None
                s = df["Close"].rename(ticker)
                s.index = s.index.tz_localize(None)
                return s
            except Exception as e:
                err = str(e).lower()
                # Rate limit or server error — back off and retry
                if attempt < 2 and any(x in err for x in ["429", "too many", "timeout", "connection"]):
                    import time
                    time.sleep(2 ** attempt)
                    continue
                return None
        return None

    # ── Parallel batch ─────────────────────────────────────────────────────

    def _fetch_batch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Fetch tickers in parallel using ThreadPoolExecutor."""
        results: list[pd.Series] = []
        with ThreadPoolExecutor(max_workers=min(10, len(tickers))) as executor:
            futures = {
                executor.submit(self._fetch_single, t, start, end): t
                for t in tickers
            }
            for future in as_completed(futures):
                s = future.result()
                if s is not None and len(s) > 0:
                    results.append(s)
        if not results:
            return pd.DataFrame()
        return pd.concat(results, axis=1).sort_index().ffill().bfill()

    # ── Bulk download (faster for many tickers) ────────────────────────────

    def _fetch_bulk(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Use yfinance bulk download — up to 5x faster than per-ticker for large lists."""
        try:
            if yf is None:
                raise ImportError("yfinance required")
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                return pd.DataFrame()
            # Extract Close prices
            if isinstance(raw.columns, pd.MultiIndex):
                if "Close" in raw.columns.get_level_values(0):
                    df = raw["Close"]
                else:
                    df = raw.xs("Close", axis=1, level=0) if "Close" in raw.columns.get_level_values(0) else pd.DataFrame()
            else:
                df = raw[["Close"]] if "Close" in raw.columns else raw

            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.dropna(how="all").ffill().bfill()
            # Keep only requested tickers
            valid = [c for c in df.columns if c in tickers]
            return df[valid] if valid else df
        except Exception:
            # Fall back to parallel single-fetch
            return self._fetch_batch(tickers, start, end)

    # ── Market load ────────────────────────────────────────────────────────

    def load_market(
        self,
        market: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load price data for a market with TTL-based caching."""
        start = start or DEFAULT_START
        end = end or DEFAULT_END
        tickers = MARKET_TICKERS.get(market, FALLBACK_TICKERS)
        cache_file = self._cache_key(market, start, end)

        if use_cache and self._cache_valid(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty and len(df) >= 50:
                    return df
            except Exception:
                pass

        # Try bulk first, fall back to parallel single
        df = self._fetch_bulk(tickers, start, end)
        if df.empty:
            df = self._fetch_batch(tickers, start, end)
        if df.empty:
            df = self._fetch_bulk(FALLBACK_TICKERS, start, end)
        if df.empty:
            df = self._fetch_batch(FALLBACK_TICKERS, start, end)

        if not df.empty and use_cache:
            try:
                df.to_parquet(cache_file)
            except Exception:
                pass
        return df

    # ── Volatility & macro ─────────────────────────────────────────────────

    def load_volatility(self, market: str, start: str, end: str) -> pd.Series:
        vix_ticker = MARKETS.get(market, {}).get("volatility", "^VIX")
        s = self._fetch_single(vix_ticker, start, end)
        if s is not None and len(s) > 0:
            return s.ffill().bfill()
        return pd.Series(20.0, index=pd.DatetimeIndex([]))

    def load_macro(self, start: str, end: str) -> pd.DataFrame:
        return self._fetch_batch(list(MACRO_TICKERS.values()), start, end)

    def get_universe(self, market: str, start: str, end: str):
        """Market + vol + macro. Parallel vol+macro fetch."""
        # Kick off market load and vol load in parallel
        with ThreadPoolExecutor(max_workers=3) as ex:
            fut_market = ex.submit(self.load_market, market, start, end, False)
            fut_vol = ex.submit(self.load_volatility, market, start, end)
            fut_macro = ex.submit(self.load_macro, start, end)
            market_df = fut_market.result()
            vol = fut_vol.result()
            macro = fut_macro.result()

        if market_df.empty or len(market_df) < 100:
            return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

        idx = market_df.index
        vol = vol.reindex(idx).ffill().bfill() if not vol.empty else pd.Series(20.0, index=idx)
        macro = macro.reindex(idx).ffill().bfill() if not macro.empty else pd.DataFrame(index=idx)
        return market_df, vol, macro

    def get_quote(self, ticker: str) -> Optional[dict]:
        """Current quote for a single ticker."""
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
