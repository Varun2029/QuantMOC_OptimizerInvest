"""
Configuration for Hedge Fund Style Quantitative System
Multi-market: India, USA, UK
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ============ MULTI-MARKET TICKERS (yfinance) ============
# India: NSE symbols use .NS suffix, BSE use .BO
# USA: direct symbols
# UK: LSE use .L suffix

MARKETS = {
    "india": {
        "name": "India",
        "currency": "₹",
        "index": "NIFTY 50",
        "volatility": "^VIX",
        "equity": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS"],
        "etfs": ["NIFTYBEES.NS", "BANKBEES.NS"],
        "bonds": ["LIQUIDBEES.NS"],   # ICICIGILT.NS delisted → use LIQUIDBEES
        "gold": ["GOLDBEES.NS"],
    },
    "usa": {
        "name": "USA",
        "currency": "$",
        "index": "S&P 500",
        "volatility": "^VIX",
        "equity": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"],
        "etfs": ["SPY", "QQQ", "XLK", "XLF"],
        "bonds": ["TLT", "BND"],
        "gold": ["GLD"],
    },
    "uk": {
        "name": "UK",
        "currency": "£",
        "index": "FTSE 100",
        "volatility": "^VIX",
        "equity": ["HSBA.L", "SHEL.L", "AZN.L", "ULVR.L", "GSK.L", "BP.L"],
        "etfs": ["ISF.L", "VUSA.L"],
        "bonds": ["VGOV.L"],
        "gold": ["SGLN.L"],
    },
}

# Default tickers per market for optimizer (smaller set for speed)
MARKET_TICKERS = {
    "india": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "NIFTYBEES.NS", "GOLDBEES.NS", "LIQUIDBEES.NS"],
    "usa": ["SPY", "QQQ", "TLT", "GLD", "XLK", "XLF"],
    "uk": ["ISF.L", "VUSA.L", "VGOV.L", "SGLN.L", "HSBA.L", "AZN.L"],
}

# Fallback if market-specific fetch fails
FALLBACK_TICKERS = ["SPY", "QQQ", "TLT", "GLD"]

MACRO_TICKERS = {"interest_rates": "^TNX", "crude_oil": "CL=F", "inflation_proxy": "TIP"}

REGIME_LABELS = {0: "Bull", 1: "Bear", 2: "High Vol", 3: "Crisis"}
N_REGIMES = 4

# India SIP
SIP_MIN_AMOUNT_INR = 500
SIP_MAX_AMOUNT_INR = 100_000
SIP_FREQUENCIES = ["Monthly", "Weekly", "Quarterly"]

# Backtest
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2024-12-31"
INITIAL_CAPITAL = 1_000_000
RISK_FREE_RATE = 0.05
TRANSACTION_COST_BPS = 10
SLIPPAGE_BPS = 5
TURNOVER_TARGET = 0.3
N_SIMULATIONS = 5_000
VAR_CONFIDENCE = 0.95
