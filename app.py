"""
Quant Invest — Hedge Fund Style Investment Platform
Multi-market (India, USA, UK) | Portfolio | SIP | Trading | Risk | Payments
"""

import sys
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        print("\n  Run with: streamlit run app.py\n")
        sys.exit(1)
except Exception:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from run_pipeline import main
from config import MARKETS, REGIME_LABELS, SIP_MIN_AMOUNT_INR, SIP_FREQUENCIES
from portfolio.model import Portfolio, Holding

# ============ THEME: Blue | Red | Black | White ============
COLORS = {
    "bg_dark": "#0a0a0a",
    "bg_card": "#111111",
    "bg_panel": "#1a1a1a",
    "border": "#2a2a2a",
    "blue": "#2563eb",
    "blue_light": "#3b82f6",
    "blue_dim": "#1e40af",
    "red": "#dc2626",
    "red_light": "#ef4444",
    "white": "#ffffff",
    "gray_light": "#e5e5e5",
    "gray": "#a3a3a3",
    "gray_dim": "#525252",
}

st.set_page_config(
    page_title="Quant Invest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global */
    .stApp {{ background: {COLORS['bg_dark']}; }}
    .main .block-container {{ 
        padding: 2rem 3rem 3rem; 
        max-width: 1500px; 
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3 {{ font-family: 'Inter', sans-serif; color: {COLORS['white']} !important; }}
    p, span, label {{ color: {COLORS['gray_light']} !important; }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0f0f0f 0%, #050505 100%) !important;
        border-right: 1px solid {COLORS['border']} !important;
    }}
    [data-testid="stSidebar"] .stMarkdown {{ color: {COLORS['white']} !important; }}
    
    /* Cards */
    .metric-card {{
        background: linear-gradient(145deg, {COLORS['bg_card']} 0%, {COLORS['bg_panel']} 100%);
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        border: 1px solid {COLORS['border']};
        margin: 0.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        transition: transform 0.2s, border-color 0.2s;
    }}
    .metric-card:hover {{ border-color: {COLORS['blue_dim']}; }}
    .metric-card .label {{ color: {COLORS['gray']} !important; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; }}
    .metric-card .value {{ color: {COLORS['white']} !important; font-size: 1.6rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }}
    .metric-card.accent-blue {{ border-left: 4px solid {COLORS['blue']}; }}
    .metric-card.accent-red {{ border-left: 4px solid {COLORS['red']}; }}
    
    /* Panels */
    .panel {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }}
    .panel-header {{
        color: {COLORS['white']};
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid {COLORS['border']};
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: {COLORS['bg_panel']} !important;
        padding: 0.5rem;
        border-radius: 10px;
        gap: 0.25rem;
        border: 1px solid {COLORS['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {COLORS['gray']} !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: {COLORS['blue']} !important;
        color: {COLORS['white']} !important;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['blue']} 0%, {COLORS['blue_dim']} 100%) !important;
        color: {COLORS['white']} !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: opacity 0.2s, transform 0.2s !important;
    }}
    .stButton > button:hover {{
        opacity: 0.9 !important;
        transform: translateY(-1px) !important;
    }}
    
    /* Signal badges */
    .signal-buy {{
        background: {COLORS['blue']} !important;
        color: white !important;
        padding: 0.25rem 0.75rem !important;
        border-radius: 6px !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
    }}
    .signal-sell {{
        background: {COLORS['red']} !important;
        color: white !important;
        padding: 0.25rem 0.75rem !important;
        border-radius: 6px !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
    }}
    .signal-hold {{
        background: {COLORS['gray_dim']} !important;
        color: white !important;
        padding: 0.25rem 0.75rem !important;
        border-radius: 6px !important;
        font-size: 0.75rem !important;
    }}
    
    /* Tables */
    .stDataFrame {{ border: 1px solid {COLORS['border']}; border-radius: 8px; overflow: hidden; }}
    .stDataFrame th {{ background: {COLORS['bg_panel']} !important; color: {COLORS['blue']} !important; }}
    .stDataFrame td {{ color: {COLORS['gray_light']} !important; }}
    
    /* Dividers */
    hr {{ border-color: {COLORS['border']} !important; }}
    
    /* Footer */
    .footer {{ color: {COLORS['gray']}; font-size: 0.75rem; margin-top: 3rem; text-align: center; }}
</style>
""", unsafe_allow_html=True)

# Session state
if "results" not in st.session_state:
    st.session_state.results = None
if "portfolio" not in st.session_state:
    st.session_state.portfolio = Portfolio()
    for m, tickers in [("india", ["RELIANCE.NS"]), ("usa", ["AAPL"]), ("uk", ["HSBA.L"])]:
        st.session_state.portfolio.add_holding(
            Holding(ticker=tickers[0], market=m, asset_type="stock", quantity=10, avg_cost=100, current_price=105)
        )
if "sips" not in st.session_state:
    st.session_state.sips = []

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### <span style='color:#2563eb'>Quant Invest</span>", unsafe_allow_html=True)
    st.caption("Multi-Market Portfolio Platform")
    st.divider()
    
    market = st.selectbox(
        "Market",
        ["india", "usa", "uk"],
        format_func=lambda x: f"{MARKETS[x]['currency']} {MARKETS[x]['name']}",
    )
    start = st.text_input("Start Date", "2020-01-01")
    end = st.text_input("End Date", "2024-12-31")
    
    if st.button("▶ Run Optimizer", use_container_width=True):
        with st.spinner("Running..."):
            try:
                st.session_state.results = main(market=market, start=start, end=end)
                st.success("Done!")
            except Exception as e:
                st.error(str(e))

if st.session_state.results is None:
    st.markdown("## Welcome to Quant Invest")
    st.markdown("Select a **market** and click **Run Optimizer** to analyze.")
    st.stop()

r = st.session_state.results
portfolio = st.session_state.portfolio
metrics = r["metrics"]

# Tab navigation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard", "💼 Portfolio", "📈 Trade", "🔄 SIP (India)", "⚠️ Risk", "💳 Payments"
])

# ============ TAB 1: DASHBOARD ============
with tab1:
    st.markdown(f"### {MARKETS[r['market']]['currency']} {MARKETS[r['market']]['name']} — Optimizer")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Sharpe", f"{metrics.sharpe:.2f}")
    with m2:
        st.metric("CAGR", f"{metrics.cagr:.1%}")
    with m3:
        st.metric("Max DD", f"{metrics.max_drawdown:.1%}")
    with m4:
        st.metric("Total Return", f"{metrics.total_return:.1%}")
    with m5:
        regime_name = REGIME_LABELS.get(r["current_regime"], "—")
        st.metric("Regime", regime_name)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(10, 4), facecolor=COLORS["bg_dark"])
        ax.set_facecolor(COLORS["bg_dark"])
        ax.fill_between(r["equity"].index, r["equity"].values, alpha=0.3, color=COLORS["blue"])
        ax.plot(r["equity"].index, r["equity"].values, color=COLORS["blue_light"], linewidth=2)
        ax.set_title("Backtest Equity", color=COLORS["white"])
        ax.set_ylabel("Value", color=COLORS["gray_light"])
        ax.tick_params(colors=COLORS["gray"])
        ax.spines["bottom"].set_color(COLORS["border"])
        ax.spines["left"].set_color(COLORS["border"])
        ax.grid(True, alpha=0.2, color=COLORS["gray_dim"])
        st.pyplot(fig)
        plt.close()
    with col2:
        if r.get("signals") is not None and not r["signals"].empty:
            st.markdown("**Optimizer Signals**")
            for _, row in r["signals"].iterrows():
                sig = row["signal"]
                cls = "signal-buy" if sig == "Buy" else ("signal-sell" if sig == "Sell" else "signal-hold")
                st.markdown(f"**{row['ticker'][:18]}** <span class='{cls}'>{sig}</span>", unsafe_allow_html=True)

# ============ TAB 2: PORTFOLIO ============
with tab2:
    st.markdown("### Portfolio Overview")
    total = portfolio.total_value()
    by_market = portfolio.by_market()
    
    p1, p2, p3 = st.columns(3)
    p1.metric("Total Value", f"₹{total:,.0f}" if total else "—")
    p2.metric("Markets", len([m for m, v in by_market.items() if v > 0]))
    p3.metric("Holdings", len(portfolio.holdings))
    
    mcols = st.columns(len(by_market))
    for i, (m, val) in enumerate(by_market.items()):
        curr = MARKETS.get(m, {}).get("currency", "")
        with mcols[i]:
            st.metric(m.upper(), f"{curr}{val:,.0f}")
    
    for m in ["india", "usa", "uk"]:
        hld = [h for h in portfolio.holdings if h.market == m]
        if hld:
            df = pd.DataFrame([
                {"Ticker": h.ticker, "Qty": h.quantity, "Avg Cost": h.avg_cost, "Value": h.value, "P&L %": f"{h.pnl_pct:.1f}%"}
                for h in hld
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

# ============ TAB 3: TRADE ============
with tab3:
    st.markdown("### Buy / Sell")
    st.caption("Optimizer signals guide Buy / Hold / Sell.")
    
    if r.get("signals") is not None and not r["signals"].empty:
        sig_df = r["signals"]
        st.dataframe(
            sig_df[["ticker", "signal", "reason", "weight", "momentum"]].rename(
                columns={"weight": "Optimal Weight", "momentum": "12m Mom"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.selectbox("Ticker", sig_df["ticker"].tolist())
            qty = st.number_input("Quantity", min_value=1, value=10)
            order_type = st.radio("Order Type", ["Market", "Limit"])
        with col2:
            side = st.radio("Side", ["Buy", "Sell"])
            if st.button("Place Order (Simulated)"):
                st.success(f"Simulated {side}: {qty} {ticker} @ Market")

# ============ TAB 4: SIP (India) ============
with tab4:
    st.markdown("### Start SIP — India")
    sip_col1, sip_col2 = st.columns(2)
    with sip_col1:
        amount = st.number_input("Monthly (₹)", min_value=SIP_MIN_AMOUNT_INR, value=5000, step=500)
        freq = st.selectbox("Frequency", SIP_FREQUENCIES)
        ticker = st.selectbox("Fund / ETF", ["NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"])
    with sip_col2:
        start_d = st.date_input("Start Date")
        tenure_y = st.number_input("Tenure (years)", min_value=1, value=5)
        if st.button("Start SIP"):
            st.session_state.sips.append({"ticker": ticker, "amount": amount, "frequency": freq, "start_date": str(start_d), "market": "india"})
            st.success("SIP started (simulated)")
    
    with st.expander("SIP Calculator"):
        amt = st.slider("Monthly (₹)", 1000, 50000, 10000)
        yrs = st.slider("Years", 5, 30, 10)
        rate = st.slider("Expected CAGR %", 8, 18, 12) / 100
        fv = amt * 12 * (((1 + rate) ** yrs - 1) / rate) * (1 + rate / 12)
        st.metric("Estimated Maturity", f"₹{fv:,.0f}")
    
    if st.session_state.sips:
        for s in st.session_state.sips:
            st.info(f"₹{s['amount']}/mo — {s['ticker']} ({s['frequency']})")

# ============ TAB 5: RISK ============
with tab5:
    st.markdown("### Risk Analysis")
    rm = r.get("risk_metrics")
    if rm:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("VaR 95%", f"{rm.var_95:.2%}")
        r2.metric("CVaR 95%", f"{rm.cvar_95:.2%}")
        r3.metric("Prob Ruin", f"{rm.prob_ruin:.2%}")
        r4.metric("Sharpe", f"{rm.sharpe:.2f}")
        
        if r.get("mc_paths") is not None:
            fig, ax = plt.subplots(figsize=(10, 4), facecolor=COLORS["bg_dark"])
            ax.set_facecolor(COLORS["bg_dark"])
            paths = r["mc_paths"]
            n = min(80, paths.shape[0])
            for i in np.random.choice(paths.shape[0], n, replace=False):
                ax.plot((1 + paths[i]) * 100, alpha=0.1, color=COLORS["blue"])
            ax.plot((1 + paths.mean(axis=0)) * 100, color=COLORS["red"], lw=2)
            ax.set_title("Monte Carlo Paths", color=COLORS["white"])
            ax.set_ylabel("Portfolio %", color=COLORS["gray_light"])
            ax.tick_params(colors=COLORS["gray"])
            ax.spines["bottom"].set_color(COLORS["border"])
            ax.spines["left"].set_color(COLORS["border"])
            ax.grid(True, alpha=0.2)
            st.pyplot(fig)
            plt.close()

# ============ TAB 6: PAYMENTS ============
with tab6:
    st.markdown("### Payments")
    st.caption("*Demo — connect payment gateway for production.*")
    pay_col1, pay_col2 = st.columns(2)
    with pay_col1:
        market_pay = st.selectbox("Market", ["india", "usa", "uk"], key="pay_market")
        amount_pay = st.number_input("Amount", min_value=100, value=10000, key="pay_amt")
        if st.button("Add Funds (Demo)"):
            st.success(f"Added {MARKETS[market_pay]['currency']}{amount_pay:,}")
    with pay_col2:
        wd_amount = st.number_input("Amount", min_value=100, value=5000, key="wd_amt")
        if st.button("Withdraw (Demo)"):
            st.success(f"Withdrawal initiated: {wd_amount}")

st.markdown('<p class="footer">Quant Invest — Multi-market optimizer. Not financial advice. Simulated trading.</p>', unsafe_allow_html=True)
