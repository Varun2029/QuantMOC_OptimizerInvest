# Quant Invest — Multi-Market Investment Platform

<p align="center">
  <strong>Hedge Fund Style Quantitative System</strong><br>
  India • USA • UK | Portfolio • SIP • Trading • Risk • Payments
</p>

---

## Description

**Quant Invest** is a quantitative investment platform that combines hedge-fund-style analytics with multi-market support across India, USA, and UK. It uses regime detection (HMM), factor-based optimization, and Monte Carlo risk analysis to generate buy/sell signals and optimal portfolio weights.

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Market** | India (NSE), USA (NYSE/NASDAQ), UK (LSE) with live data via yfinance |
| **Portfolio** | Track holdings, bonds, SIP across markets with real-time P&L |
| **Trading** | Buy/Sell with optimizer signals (Buy / Hold / Sell) based on regime & momentum |
| **SIP (India)** | Start Systematic Investment Plans in ETFs (NIFTYBEES, BANKBEES, GOLDBEES) |
| **Risk Engine** | VaR, CVaR, Monte Carlo simulation, regime detection (Bull/Bear/High Vol/Crisis) |
| **Payments** | Add funds / withdraw (demo UI — integrate Razorpay/Stripe for production) |

### Tech Stack

- **Data**: yfinance, pandas
- **Regime**: hmmlearn (Hidden Markov Model)
- **Optimization**: cvxpy (Mean-Variance, Risk Parity, Black-Litterman, CVaR)
- **UI**: Streamlit
- **Risk**: Monte Carlo VaR/CVaR

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run
streamlit run app.py
```

**Important:** Use `streamlit run app.py`, not `python app.py`.

---

## Project Structure

```
QUANT/
├── app.py              # Main Streamlit app (Blue/Red/Black/White theme)
├── config.py           # Markets, tickers (India/USA/UK)
├── run_pipeline.py     # Orchestrates data → factors → regime → optimizer → risk
├── portfolio/          # Holdings, SIP, bonds model
├── data/               # Multi-market loader (yfinance)
├── feature_engine/     # Momentum, volatility, value factors
├── regime_model/       # HMM regime detection
├── optimizer/          # Portfolio weights + Buy/Sell signals
├── risk_engine/        # Monte Carlo VaR/CVaR
└── backtester/         # Sharpe, CAGR, transaction costs
```

---

##  UI

- **Color scheme**: Blue, Red, Black, White
- **Tabs**: Dashboard, Portfolio, Trade, SIP (India), Risk, Payments
- Dark theme with accent highlights

---

## Disclaimer

This is a **simulated** trading platform. No real orders or payments are executed. For live trading, integrate broker APIs (e.g. Zerodha, Upstox, Alpaca). Not financial advice.

---

## License

MIT
