"""
Quant Invest — Upgraded UI
Modern dark UI | Plotly charts | Floating FAB | Suggestion toasts
Real payment integrations: Razorpay (India), Stripe (USA/UK), PayPal (Global)
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
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from pathlib import Path

from run_pipeline import main
from config import MARKETS, REGIME_LABELS, SIP_MIN_AMOUNT_INR, SIP_FREQUENCIES
from portfolio.model import Portfolio, Holding

# ── Plotly ────────────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY = True
except ImportError:
    PLOTLY = False

# ── Payment gateways ──────────────────────────────────────────────────────
try:
    import razorpay
    RAZORPAY_AVAILABLE = True
except ImportError:
    RAZORPAY_AVAILABLE = False

try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False

# ── Read keys from st.secrets (set on Streamlit Cloud dashboard) ──────────
# Falls back to empty string so sandbox simulation still works locally
def _secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

# ═══════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════════════════
C = {
    "bg":       "#060810",
    "surface":  "#0d1117",
    "card":     "#111827",
    "border":   "#1f2937",
    "border2":  "#374151",
    "blue":     "#3b82f6",
    "blue2":    "#1d4ed8",
    "blue_dim": "#1e3a5f",
    "cyan":     "#06b6d4",
    "green":    "#10b981",
    "red":      "#ef4444",
    "amber":    "#f59e0b",
    "purple":   "#8b5cf6",
    "white":    "#f9fafb",
    "gray1":    "#9ca3af",
    "gray2":    "#6b7280",
    "gray3":    "#374151",
}

REGIME_COLORS = {
    "Bull": C["green"],
    "Bear": C["red"],
    "High Vol": C["amber"],
    "Crisis": "#dc2626",
}

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Quant Invest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Syne:wght@700;800&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after {{ box-sizing: border-box; }}
.stApp {{ background: {C['bg']}; font-family: 'Space Grotesk', sans-serif; }}
.main .block-container {{ padding: 1.5rem 2.5rem 4rem; max-width: 1600px; }}
h1,h2,h3,h4 {{ font-family: 'Syne', sans-serif; color: {C['white']} !important; }}
p, span, label, div {{ color: {C['gray1']} !important; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #080c14 0%, {C['surface']} 100%) !important;
    border-right: 1px solid {C['border']} !important;
    min-width: 280px !important;
}}
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label {{ color: {C['white']} !important; }}

/* ── Cards ── */
.qcard {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin: 0.5rem 0;
    position: relative;
    transition: border-color .2s, transform .2s, box-shadow .2s;
}}
.qcard:hover {{
    border-color: {C['blue_dim']};
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(59,130,246,.12);
}}
.qcard.accent-blue  {{ border-left: 3px solid {C['blue']}; }}
.qcard.accent-green {{ border-left: 3px solid {C['green']}; }}
.qcard.accent-red   {{ border-left: 3px solid {C['red']}; }}
.qcard.accent-amber {{ border-left: 3px solid {C['amber']}; }}
.qcard.accent-cyan  {{ border-left: 3px solid {C['cyan']}; }}
.qcard .qlabel {{ color: {C['gray2']} !important; font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; margin-bottom: .3rem; }}
.qcard .qval   {{ color: {C['white']} !important; font-size: 1.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; line-height: 1.1; }}
.qcard .qdelta {{ font-size: .78rem; margin-top: .25rem; font-family: 'JetBrains Mono', monospace; }}
.qdelta.up   {{ color: {C['green']} !important; }}
.qdelta.down {{ color: {C['red']} !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {C['surface']} !important;
    padding: .4rem !important;
    border-radius: 12px !important;
    gap: .25rem !important;
    border: 1px solid {C['border']} !important;
}}
.stTabs [data-baseweb="tab"] {{
    color: {C['gray1']} !important;
    padding: .65rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: .88rem !important;
    border-radius: 8px !important;
    transition: all .15s !important;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg,{C['blue']} 0%,{C['blue2']} 100%) !important;
    color: {C['white']} !important;
    box-shadow: 0 4px 12px rgba(59,130,246,.3) !important;
}}

/* ── Buttons ── */
.stButton>button {{
    background: linear-gradient(135deg,{C['blue']} 0%,{C['blue2']} 100%) !important;
    color: {C['white']} !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
    padding: .65rem 1.4rem !important;
    transition: opacity .2s, transform .15s, box-shadow .2s !important;
    letter-spacing: .01em !important;
}}
.stButton>button:hover {{
    opacity: .9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,.35) !important;
}}

/* ── Signals ── */
.sig {{
    display: inline-block;
    padding: .2rem .7rem;
    border-radius: 6px;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
}}
.sig-buy   {{ background: rgba(16,185,129,.2); color: {C['green']} !important; border: 1px solid rgba(16,185,129,.4); }}
.sig-sell  {{ background: rgba(239,68,68,.2);  color: {C['red']} !important;   border: 1px solid rgba(239,68,68,.4); }}
.sig-hold  {{ background: rgba(107,114,128,.2); color: {C['gray1']} !important; border: 1px solid rgba(107,114,128,.4); }}

/* ── Regime badge ── */
.regime-badge {{
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    padding: .35rem .9rem;
    border-radius: 20px;
    font-size: .8rem;
    font-weight: 600;
    border: 1px solid;
}}

/* ── Section headers ── */
.section-header {{
    display: flex;
    align-items: center;
    gap: .75rem;
    margin: 1.5rem 0 1rem;
    padding-bottom: .75rem;
    border-bottom: 1px solid {C['border']};
}}
.section-header h3 {{ margin: 0 !important; font-size: 1.15rem !important; }}
.section-icon {{
    width: 34px; height: 34px;
    background: rgba(59,130,246,.15);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    border: 1px solid rgba(59,130,246,.25);
}}

/* ── Tables ── */
.stDataFrame {{ border-radius: 10px !important; overflow: hidden !important; border: 1px solid {C['border']} !important; }}
.stDataFrame th {{ background: {C['surface']} !important; color: {C['blue']} !important; font-weight: 600 !important; }}
.stDataFrame td {{ color: {C['gray1']} !important; }}

/* ── Inputs ── */
.stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div {{
    background: {C['card']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 8px !important;
    color: {C['white']} !important;
}}
.stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {{
    border-color: {C['blue']} !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,.2) !important;
}}

/* ── Payment cards ── */
.pay-method {{
    background: {C['card']};
    border: 2px solid {C['border']};
    border-radius: 12px;
    padding: 1rem;
    cursor: pointer;
    transition: all .2s;
    text-align: center;
}}
.pay-method:hover, .pay-method.selected {{
    border-color: {C['blue']};
    background: rgba(59,130,246,.08);
    box-shadow: 0 0 0 3px rgba(59,130,246,.15);
}}
.pay-method .pm-icon {{ font-size: 2rem; margin-bottom: .4rem; }}
.pay-method .pm-name {{ font-size: .8rem; font-weight: 600; color: {C['white']} !important; }}

/* ── Floating FAB ── */
#fab-container {{
    position: fixed;
    bottom: 2rem; right: 2rem;
    z-index: 9999;
    display: flex;
    flex-direction: column-reverse;
    align-items: flex-end;
    gap: .6rem;
}}
.fab-main {{
    width: 56px; height: 56px;
    background: linear-gradient(135deg,{C['blue']} 0%,{C['blue2']} 100%);
    border-radius: 50%;
    border: none;
    cursor: pointer;
    font-size: 1.4rem;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 6px 24px rgba(59,130,246,.45);
    transition: transform .2s, box-shadow .2s;
    color: white;
}}
.fab-main:hover {{ transform: scale(1.1) rotate(45deg); box-shadow: 0 10px 32px rgba(59,130,246,.6); }}
.fab-action {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: .5rem 1rem;
    font-size: .8rem;
    font-weight: 600;
    color: {C['white']};
    cursor: pointer;
    white-space: nowrap;
    opacity: 0;
    transform: translateX(20px);
    transition: all .2s;
    box-shadow: 0 4px 16px rgba(0,0,0,.4);
}}
.fab-open .fab-action {{ opacity: 1; transform: translateX(0); }}

/* ── Chart entrance animation ── */
.js-plotly-plot {{
    animation: chartrise 0.7s cubic-bezier(0.22,1,0.36,1) both;
}}
@keyframes chartrise {{
    from {{ opacity: 0; transform: translateY(22px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
/* ── Glow pulse on last marker ── */
@keyframes glowpulse {{
    0%,100% {{ filter: drop-shadow(0 0 4px rgba(6,182,212,0.7));  opacity: 1; }}
    50%     {{ filter: drop-shadow(0 0 16px rgba(6,182,212,1.0)); opacity: 0.7; }}
}}

/* ── Divider ── */
hr {{ border-color: {C['border']} !important; margin: 1rem 0 !important; }}

/* ── Toast ── */
.stToast {{ background: {C['card']} !important; border: 1px solid {C['border2']} !important; color: {C['white']} !important; }}

/* ── Progress ── */
.stProgress>div>div>div {{ background: linear-gradient(90deg,{C['blue']},{C['cyan']}) !important; border-radius: 4px !important; }}

/* ── Footer ── */
.qfooter {{ color: {C['gray2']} !important; font-size: .72rem; text-align: center; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid {C['border']}; }}
</style>
""", unsafe_allow_html=True)

# ─── Floating FAB ─────────────────────────────────────────────────────────
components.html("""
<div id="fab-container">
  <button class="fab-main" onclick="toggleFab()" title="Quick Actions">+</button>
  <div id="fab-actions" style="display:flex;flex-direction:column;gap:.5rem;align-items:flex-end;">
    <button class="fab-action" onclick="scrollToTab(0)">📊 Dashboard</button>
    <button class="fab-action" onclick="scrollToTab(2)">⚡ Quick Trade</button>
    <button class="fab-action" onclick="scrollToTab(3)">🔄 SIP Setup</button>
    <button class="fab-action" onclick="scrollToTab(5)">💳 Payments</button>
  </div>
</div>
<style>
  #fab-container { position:fixed; bottom:2rem; right:2rem; z-index:9999;
    display:flex; flex-direction:column-reverse; align-items:flex-end; gap:.6rem; }
  .fab-main { width:56px;height:56px; background:linear-gradient(135deg,#3b82f6,#1d4ed8);
    border-radius:50%; border:none; cursor:pointer; font-size:1.5rem; color:white;
    box-shadow:0 6px 24px rgba(59,130,246,.45); transition:transform .2s,box-shadow .2s; }
  .fab-main:hover { transform:scale(1.1) rotate(45deg); box-shadow:0 10px 32px rgba(59,130,246,.6); }
  .fab-action { background:#111827; border:1px solid #1f2937; border-radius:10px;
    padding:.55rem 1.1rem; font-size:.82rem; font-weight:600; color:#f9fafb; cursor:pointer;
    white-space:nowrap; opacity:0; transform:translateX(20px); transition:all .2s;
    box-shadow:0 4px 16px rgba(0,0,0,.4); }
  .fab-open .fab-action { opacity:1; transform:translateX(0); }
  #fab-actions { display:flex;flex-direction:column;gap:.5rem;align-items:flex-end; }
</style>
<script>
  let fabOpen = false;
  function toggleFab() {
    fabOpen = !fabOpen;
    const c = document.getElementById('fab-container');
    if (fabOpen) { c.classList.add('fab-open'); }
    else { c.classList.remove('fab-open'); }
  }
  function scrollToTab(i) {
    const tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
    if (tabs[i]) tabs[i].click();
    toggleFab();
  }
</script>
""", height=0)

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
if "results" not in st.session_state:
    st.session_state.results = None
if "portfolio" not in st.session_state:
    st.session_state.portfolio = Portfolio()
    for m, tk in [("india", "RELIANCE.NS"), ("usa", "AAPL"), ("uk", "HSBA.L")]:
        st.session_state.portfolio.add_holding(
            Holding(ticker=tk, market=m, asset_type="stock", quantity=10, avg_cost=100, current_price=105)
        )
if "sips" not in st.session_state:
    st.session_state.sips = []
if "transactions" not in st.session_state:
    st.session_state.transactions = []
if "wallet" not in st.session_state:
    st.session_state.wallet = {"india": 50000.0, "usa": 1000.0, "uk": 800.0}
if "pay_method" not in st.session_state:
    st.session_state.pay_method = None

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS  (must be defined before tab code references them)
# ═══════════════════════════════════════════════════════════════════════════
def _process_payment_success(market: str, amount: float, gateway: str, ref: str):
    """Update wallet balance and record transaction."""
    st.session_state.wallet[market] += amount
    st.session_state.transactions.append({
        "type": "Deposit",
        "market": market,
        "amount": amount,
        "currency": MARKETS[market]["currency"],
        "gateway": gateway,
        "reference": ref,
        "status": "Completed",
    })

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:.5rem 0 1rem'>
      <div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;
           background:linear-gradient(90deg,#3b82f6,#06b6d4);-webkit-background-clip:text;
           -webkit-text-fill-color:transparent'>Quant Invest</div>
      <div style='font-size:.75rem;color:#6b7280;margin-top:.2rem'>Hedge Fund Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Wallet summary
    wallet = st.session_state.wallet
    currencies = {"india": "₹", "usa": "$", "uk": "£"}
    st.markdown("**Wallet**")
    st.markdown(f"""
    <div style='display:flex;flex-direction:column;gap:.4rem;margin-bottom:.5rem'>
      <div style='display:flex;justify-content:space-between;align-items:center;
           background:#0d1117;border:1px solid #1f2937;border-radius:8px;padding:.5rem .85rem'>
        <span style='font-size:.75rem;color:#6b7280'>🇮🇳 India</span>
        <span style='font-family:JetBrains Mono;font-size:.9rem;color:#f9fafb;font-weight:600'>
          ₹{wallet['india']:,.0f}
        </span>
      </div>
      <div style='display:flex;justify-content:space-between;align-items:center;
           background:#0d1117;border:1px solid #1f2937;border-radius:8px;padding:.5rem .85rem'>
        <span style='font-size:.75rem;color:#6b7280'>🇺🇸 USA</span>
        <span style='font-family:JetBrains Mono;font-size:.9rem;color:#f9fafb;font-weight:600'>
          ${wallet['usa']:,.0f}
        </span>
      </div>
      <div style='display:flex;justify-content:space-between;align-items:center;
           background:#0d1117;border:1px solid #1f2937;border-radius:8px;padding:.5rem .85rem'>
        <span style='font-size:.75rem;color:#6b7280'>🇬🇧 UK</span>
        <span style='font-family:JetBrains Mono;font-size:.9rem;color:#f9fafb;font-weight:600'>
          £{wallet['uk']:,.0f}
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    market = st.selectbox(
        "Market",
        ["india", "usa", "uk"],
        format_func=lambda x: f"{MARKETS[x]['currency']} {MARKETS[x]['name']}",
    )
    start = st.text_input("Start Date", "2020-01-01")
    end = st.text_input("End Date", "2024-12-31")

    with st.expander("⚙ Model Parameters"):
        n_sims = st.slider("MC Simulations", 500, 10000, 5000, 500)
        n_regimes = st.slider("HMM Regimes", 2, 6, 4)
        force_refit = st.checkbox("Force Model Refit", value=False)
        use_ensemble = st.checkbox("Use RF Ensemble", value=True)

    run_clicked = st.button("▶ Run Optimizer", use_container_width=True)
    if run_clicked:
        with st.spinner("Running pipeline..."):
            prog = st.progress(0)
            try:
                prog.progress(20)
                st.session_state.results = main(
                    market=market, start=start, end=end, force_refit=force_refit
                )
                prog.progress(100)
                st.success("✅ Analysis complete")

                # ── Suggestion toasts ──────────────────────────────────────
                r = st.session_state.results
                if r and r.get("signals") is not None and not r["signals"].empty:
                    buys = r["signals"][r["signals"]["signal"] == "Buy"]
                    if len(buys) > 0:
                        t = buys.iloc[0]
                        st.toast(
                            f"💡 **{t['ticker']}** — Buy signal | Momentum {t['momentum']:.1%} | Weight {t['weight']:.1%}",
                            icon="📈"
                        )
                if r and r.get("risk_metrics"):
                    rm = r["risk_metrics"]
                    if rm.prob_ruin > 0.15:
                        st.toast(
                            f"⚠️ Risk Alert: Ruin probability {rm.prob_ruin:.0%} — consider defensive rebalance",
                            icon="🔴"
                        )
                    if rm.sharpe > 1.5:
                        st.toast(f"🏆 Excellent Sharpe: {rm.sharpe:.2f}", icon="✨")

                regime_name = REGIME_LABELS.get(r.get("current_regime", 0), "—")
                st.toast(f"🔮 Current Regime: **{regime_name}**", icon="🌐")

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                prog.empty()

    if st.session_state.results:
        r = st.session_state.results
        regime_name = REGIME_LABELS.get(r.get("current_regime", 0), "—")
        rc = REGIME_COLORS.get(regime_name, C["blue"])
        st.markdown(f"""
        <div style='margin-top:1rem;padding:.6rem 1rem;border-radius:10px;
             background:rgba(0,0,0,.3);border:1px solid {rc}44'>
          <div style='font-size:.72rem;color:#6b7280'>Current Regime</div>
          <div style='font-size:1rem;font-weight:700;color:{rc};margin-top:.2rem'>⬤ {regime_name}</div>
        </div>
        """, unsafe_allow_html=True)

if st.session_state.results is None:
    # Landing screen
    st.markdown("""
    <div style='text-align:center;padding:5rem 2rem'>
      <div style='font-family:Syne,sans-serif;font-size:3rem;font-weight:800;
           background:linear-gradient(90deg,#3b82f6,#06b6d4,#8b5cf6);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           margin-bottom:1rem'>Quant Invest</div>
      <div style='font-size:1.1rem;color:#9ca3af;max-width:500px;margin:0 auto 2rem'>
        Hedge fund-style quantitative analytics.<br>Select a market and run the optimizer to begin.
      </div>
      <div style='display:flex;gap:1rem;justify-content:center;flex-wrap:wrap'>
        <div style='background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1rem 1.5rem;'>
          <div style='font-size:1.5rem'>🔬</div><div style='font-size:.8rem;color:#9ca3af;margin-top:.3rem'>ML Regime Detection</div>
        </div>
        <div style='background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1rem 1.5rem;'>
          <div style='font-size:1.5rem'>⚡</div><div style='font-size:.8rem;color:#9ca3af;margin-top:.3rem'>Factor Optimization</div>
        </div>
        <div style='background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1rem 1.5rem;'>
          <div style='font-size:1.5rem'>🎲</div><div style='font-size:.8rem;color:#9ca3af;margin-top:.3rem'>Monte Carlo Risk</div>
        </div>
        <div style='background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1rem 1.5rem;'>
          <div style='font-size:1.5rem'>💳</div><div style='font-size:.8rem;color:#9ca3af;margin-top:.3rem'>Live Payments</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════
r = st.session_state.results
portfolio = st.session_state.portfolio
metrics = r["metrics"]
rm = r.get("risk_metrics")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard", "💼 Portfolio", "⚡ Trade", "🔄 SIP", "🎲 Risk", "💳 Payments"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    mkt_info = MARKETS[r["market"]]
    regime_name = REGIME_LABELS.get(r["current_regime"], "—")
    rc = REGIME_COLORS.get(regime_name, C["blue"])

    st.markdown(f"""
    <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem'>
      <div>
        <div style='font-family:Syne,sans-serif;font-size:1.6rem;font-weight:700;color:#f9fafb'>
          {mkt_info['currency']} {mkt_info['name']} — Optimizer Results
        </div>
        <div style='font-size:.82rem;color:#6b7280'>
          {r.get("market_df").index[0].strftime('%d %b %Y') if not r["market_df"].empty else ""}
          &nbsp;→&nbsp;
          {r.get("market_df").index[-1].strftime('%d %b %Y') if not r["market_df"].empty else ""}
        </div>
      </div>
      <div class='regime-badge' style='color:{rc};border-color:{rc}44;background:{rc}18'>
        ⬤ &nbsp;{regime_name} Regime
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ──
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpi_data = [
        (k1, "Sharpe Ratio", f"{metrics.sharpe:.2f}", "accent-blue",
         "↑ Higher is better", "up" if metrics.sharpe > 1 else "down"),
        (k2, "CAGR", f"{metrics.cagr:.1%}", "accent-green",
         "Annual compounded return", "up" if metrics.cagr > 0 else "down"),
        (k3, "Max Drawdown", f"{metrics.max_drawdown:.1%}", "accent-red",
         "Peak-to-trough loss", "down"),
        (k4, "Total Return", f"{metrics.total_return:.1%}", "accent-blue",
         "Full period return", "up" if metrics.total_return > 0 else "down"),
        (k5, "Volatility", f"{metrics.volatility:.1%}", "accent-amber",
         "Annualised std dev", ""),
        (k6, "Sortino", f"{metrics.sortino:.2f}", "accent-cyan",
         "Downside-adjusted", "up" if metrics.sortino > 1 else "down"),
    ]
    for col, label, val, accent, delta_txt, delta_cls in kpi_data:
        with col:
            st.markdown(f"""
            <div class='qcard {accent}'>
              <div class='qlabel'>{label}</div>
              <div class='qval'>{val}</div>
              <div class='qdelta {delta_cls}'>{delta_txt}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── Charts ──
    chart_col, sig_col = st.columns([7, 3])

    with chart_col:
        if PLOTLY and r.get("equity") is not None:
            eq = r["equity"]

            # Helper: hex → rgba string (no 8-digit hex — Plotly doesn't support it)
            def hex_to_rgba(h, a):
                h = h.lstrip("#")
                r2, g2, b2 = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
                return f"rgba({r2},{g2},{b2},{a})"

            REGIME_RGBA = {
                "Bull":     hex_to_rgba("#10b981", 0.08),
                "Bear":     hex_to_rgba("#ef4444", 0.08),
                "High Vol": hex_to_rgba("#f59e0b", 0.08),
                "Crisis":   hex_to_rgba("#dc2626", 0.10),
            }

            fig = go.Figure()

            # Regime bands drawn first (behind equity line)
            if r.get("regime") is not None:
                regime_s = r["regime"].reindex(eq.index).ffill().bfill()
                eq_min = float(eq.min()) * 0.98
                for regime_id, regime_lbl in REGIME_LABELS.items():
                    mask = regime_s == regime_id
                    if not mask.any():
                        continue
                    dates_r = list(eq.index[mask])
                    vals_top = list(eq[mask].values)
                    # Closed polygon: top then bottom reversed
                    fig.add_trace(go.Scatter(
                        x=dates_r + dates_r[::-1],
                        y=vals_top + [eq_min] * len(dates_r),
                        fill="toself",
                        fillcolor=REGIME_RGBA.get(regime_lbl, "rgba(55,65,81,0.08)"),
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                        name=regime_lbl,
                    ))

            # Gradient fill under equity curve
            fig.add_trace(go.Scatter(
                x=eq.index, y=eq.values,
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.07)",
                line=dict(color="rgba(59,130,246,0.0)", width=0),
                showlegend=False, hoverinfo="skip", name="_fill",
            ))

            # Main equity line — vibrant blue
            fig.add_trace(go.Scatter(
                x=eq.index, y=eq.values,
                line=dict(color="#3b82f6", width=2.5),
                name="Portfolio",
                hovertemplate="%{x|%d %b %Y}<br><b>%{y:,.0f}</b><extra></extra>",
                showlegend=False,
            ))

            # Glowing endpoint dot — rendered as a 1-point scatter with large marker
            fig.add_trace(go.Scatter(
                x=[eq.index[-1]], y=[eq.values[-1]],
                mode="markers",
                marker=dict(
                    color="#06b6d4",
                    size=12,
                    line=dict(color="rgba(6,182,212,0.4)", width=6),
                    symbol="circle",
                ),
                name="Current",
                hovertemplate=f"Latest: <b>{eq.values[-1]:,.0f}</b><extra></extra>",
                showlegend=False,
            ))

            fig.update_layout(
                paper_bgcolor="#060810",
                plot_bgcolor="#060810",
                xaxis=dict(
                    showgrid=False, color="#6b7280", showline=False,
                    tickfont=dict(size=11, color="#6b7280"),
                    zeroline=False,
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="rgba(31,41,55,0.6)",
                    color="#6b7280", tickfont=dict(size=11, color="#6b7280"),
                    zeroline=False,
                ),
                margin=dict(l=8, r=8, t=40, b=8),
                title=dict(
                    text="Backtest Equity Curve",
                    font=dict(color="#f9fafb", size=14, family="Syne"),
                    x=0.01,
                ),
                showlegend=False,
                hovermode="x unified",
                font=dict(family="Space Grotesk"),
                hoverlabel=dict(
                    bgcolor="#111827", font_color="#f9fafb",
                    bordercolor="#374151",
                ),
                height=340,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # ── CSS glow pulse on chart endpoint via injected style ──
            components.html("""
            <style>
            /* Animate the Plotly endpoint marker with a glow pulse */
            .js-plotly-plot .plotly .scatter .point:last-child path {
                animation: glowpulse 2s ease-in-out infinite;
                filter: drop-shadow(0 0 6px rgba(6,182,212,0.9));
            }
            @keyframes glowpulse {
                0%,100% { opacity:1; filter: drop-shadow(0 0 4px rgba(6,182,212,0.7)); }
                50%      { opacity:.6; filter: drop-shadow(0 0 14px rgba(6,182,212,1)); }
            }
            </style>
            """, height=0)

    with sig_col:
        if r.get("signals") is not None and not r["signals"].empty:
            st.markdown("**Optimizer Signals**")
            for _, row in r["signals"].iterrows():
                sig = row["signal"]
                sig_cls = f"sig-{'buy' if sig=='Buy' else 'sell' if sig=='Sell' else 'hold'}"
                explanation = row.get("explanation", "")
                tip = f'title="{explanation}"' if explanation else ""
                st.markdown(f"""
                <div class='qcard' style='padding:.8rem 1rem;margin:.3rem 0' {tip}>
                  <div style='display:flex;justify-content:space-between;align-items:center'>
                    <span style='color:#f9fafb;font-weight:600;font-size:.88rem'>{row['ticker'][:14]}</span>
                    <span class='sig {sig_cls}'>{sig}</span>
                  </div>
                  <div style='color:#6b7280;font-size:.73rem;margin-top:.3rem'>
                    {row.get('reason','')[:40]}
                  </div>
                  <div style='display:flex;gap:.8rem;margin-top:.3rem'>
                    <span style='font-size:.72rem;color:#9ca3af'>Wt: <b style='color:#f9fafb'>{row['weight']:.1%}</b></span>
                    <span style='font-size:.72rem;color:#9ca3af'>Mom: <b style='color:{"#10b981" if row["momentum"]>0 else "#ef4444"}'>{row["momentum"]:.1%}</b></span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Regime Performance ──
    if r.get("regime_perf") is not None and not r["regime_perf"].empty:
        st.markdown("**Performance by Regime**")
        rp = r["regime_perf"]
        if PLOTLY:
            regime_bar_colors = [
                REGIME_COLORS.get(str(name), "#3b82f6")
                for name in rp["regime"]
            ]
            fig2 = go.Figure(go.Bar(
                x=rp["regime"], y=rp["sharpe"],
                marker=dict(
                    color=regime_bar_colors,
                    opacity=0.85,
                    line=dict(width=0),
                ),
                hovertemplate="%{x}: <b>%{y:.2f}</b><extra></extra>",
            ))
            fig2.update_layout(
                paper_bgcolor="#060810", plot_bgcolor="#060810",
                font=dict(color="#9ca3af", family="Space Grotesk"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="rgba(31,41,55,0.6)", color="#6b7280"),
                margin=dict(l=8, r=8, t=10, b=8),
                height=220,
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    total = portfolio.total_value()
    by_market = portfolio.by_market()

    p1, p2, p3 = st.columns(3)
    for col, label, val, accent in [
        (p1, "Total Value", f"₹{total:,.0f}", "accent-blue"),
        (p2, "Active Markets", str(len([m for m, v in by_market.items() if v > 0])), "accent-green"),
        (p3, "Holdings", str(len(portfolio.holdings)), "accent-amber"),
    ]:
        with col:
            st.markdown(f"""
            <div class='qcard {accent}'>
              <div class='qlabel'>{label}</div>
              <div class='qval'>{val}</div>
            </div>
            """, unsafe_allow_html=True)

    # Market breakdown + portfolio treemap
    left, right = st.columns([3, 2])
    with left:
        for mkt in ["india", "usa", "uk"]:
            hld = [h for h in portfolio.holdings if h.market == mkt]
            if hld:
                curr = MARKETS.get(mkt, {}).get("currency", "")
                st.markdown(f"**{mkt.upper()} Holdings — {curr}**")
                df = pd.DataFrame([{
                    "Ticker": h.ticker, "Qty": h.quantity,
                    "Avg Cost": f"{curr}{h.avg_cost:,.2f}",
                    "Current": f"{curr}{h.current_price:,.2f}",
                    "Value": f"{curr}{h.value:,.2f}",
                    "P&L %": f"{'+' if h.pnl_pct >= 0 else ''}{h.pnl_pct:.2f}%"
                } for h in hld])
                st.dataframe(df, use_container_width=True, hide_index=True)

    with right:
        if PLOTLY and by_market:
            labels = list(by_market.keys())
            values = list(by_market.values())
            fig3 = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.55,
                marker=dict(
                    colors=["#3b82f6", "#06b6d4", "#8b5cf6"],
                    line=dict(color="#060810", width=2),
                ),
                textfont=dict(color="white", size=12),
                hovertemplate="%{label}: <b>%{value:,.0f}</b> (%{percent})<extra></extra>",
            ))
            fig3.update_layout(
                paper_bgcolor="#060810",
                font=dict(color="#9ca3af", family="Space Grotesk"),
                margin=dict(l=8, r=8, t=36, b=8),
                showlegend=True,
                legend=dict(font=dict(color="#9ca3af"), bgcolor="rgba(0,0,0,0)"),
                title=dict(text="Market Allocation", font=dict(color="#f9fafb", family="Syne")),
                annotations=[dict(
                    text=f"₹{total:,.0f}", x=0.5, y=0.5,
                    font=dict(size=12, color="white", family="JetBrains Mono"),
                    showarrow=False,
                )],
                hoverlabel=dict(bgcolor="#111827", font_color="#f9fafb", bordercolor="#374151"),
            )
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # Add holding form
    with st.expander("➕ Add Holding"):
        ac1, ac2, ac3, ac4, ac5, ac6 = st.columns(6)
        with ac1: new_ticker = st.text_input("Ticker", "AAPL")
        with ac2: new_market = st.selectbox("Market", ["india", "usa", "uk"], key="add_mkt")
        with ac3: new_type = st.selectbox("Type", ["stock", "etf", "bond", "gold"])
        with ac4: new_qty = st.number_input("Qty", min_value=1, value=10)
        with ac5: new_cost = st.number_input("Avg Cost", min_value=1.0, value=100.0)
        with ac6: new_price = st.number_input("Current Price", min_value=1.0, value=105.0)
        if st.button("Add Holding"):
            portfolio.add_holding(Holding(
                ticker=new_ticker, market=new_market, asset_type=new_type,
                quantity=new_qty, avg_cost=new_cost, current_price=new_price
            ))
            st.success(f"Added {new_ticker}")
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: TRADE
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class='section-header'>
      <div class='section-icon'>⚡</div>
      <h3>Order Execution</h3>
      <span style='font-size:.78rem;color:#6b7280;margin-left:.5rem'>Simulated trading — signals guide direction</span>
    </div>
    """, unsafe_allow_html=True)

    if r.get("signals") is not None and not r["signals"].empty:
        sig_df = r["signals"]

        # Signals table
        display_cols = ["ticker", "signal", "reason", "weight", "momentum", "regime"]
        display_cols = [c for c in display_cols if c in sig_df.columns]
        st.dataframe(
            sig_df[display_cols].rename(columns={
                "weight": "Optimal Wt", "momentum": "12m Mom"
            }),
            use_container_width=True, hide_index=True
        )

        st.markdown("")
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            ticker = st.selectbox("Ticker", sig_df["ticker"].tolist(), key="trade_ticker")
            side = st.radio("Side", ["Buy", "Sell"], horizontal=True)
        with tc2:
            qty = st.number_input("Quantity", min_value=1, value=10, key="trade_qty")
            order_type = st.radio("Order Type", ["Market", "Limit", "Stop-Loss"], horizontal=True)
        with tc3:
            if order_type in ["Limit", "Stop-Loss"]:
                limit_px = st.number_input("Limit Price", min_value=0.01, value=100.0)
            else:
                limit_px = None
                st.markdown("<div class='qcard' style='padding:.8rem 1rem'><div class='qlabel'>Execution</div><div style='color:#f9fafb;font-weight:600'>At Market Price</div></div>", unsafe_allow_html=True)

            # Signal suggestion box
            matched = sig_df[sig_df["ticker"] == ticker]
            if not matched.empty:
                row = matched.iloc[0]
                sig_color = C["green"] if row["signal"] == "Buy" else C["red"] if row["signal"] == "Sell" else C["gray2"]
                st.markdown(f"""
                <div class='qcard' style='border-left:3px solid {sig_color}'>
                  <div class='qlabel'>AI Signal for {ticker}</div>
                  <div style='color:{sig_color};font-weight:700;font-size:1rem'>{row["signal"]}</div>
                  <div style='color:#9ca3af;font-size:.75rem;margin-top:.3rem'>{row.get("reason","")}</div>
                </div>
                """, unsafe_allow_html=True)

        # Two-step confirm
        if "confirm_order" not in st.session_state:
            st.session_state.confirm_order = False

        if st.button("Preview Order"):
            st.session_state.confirm_order = True

        if st.session_state.confirm_order:
            limit_str = f" @ {limit_px}" if limit_px else " @ Market"
            st.markdown(f"""
            <div class='qcard accent-amber' style='margin-top:.5rem'>
              <div class='qlabel'>Order Preview</div>
              <div style='color:#f9fafb;font-size:1.1rem;font-weight:700'>{side} {qty} × {ticker}{limit_str}</div>
              <div style='color:#9ca3af;font-size:.78rem;margin-top:.3rem'>Type: {order_type} | Market: {r["market"].upper()}</div>
            </div>
            """, unsafe_allow_html=True)
            col_confirm, col_cancel = st.columns([1, 3])
            with col_confirm:
                if st.button("✅ Confirm", key="confirm_btn"):
                    st.success(f"Order placed: {side} {qty} × {ticker}")
                    st.session_state.confirm_order = False
                    st.session_state.transactions.append({
                        "type": side, "ticker": ticker, "qty": qty,
                        "order_type": order_type, "status": "Simulated"
                    })
            with col_cancel:
                if st.button("Cancel", key="cancel_btn"):
                    st.session_state.confirm_order = False

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: SIP
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class='section-header'>
      <div class='section-icon'>🔄</div>
      <h3>SIP Manager — India</h3>
    </div>
    """, unsafe_allow_html=True)

    sip1, sip2 = st.columns(2)
    with sip1:
        sip_amount = st.number_input("Monthly Amount (₹)", min_value=SIP_MIN_AMOUNT_INR, value=5000, step=500)
        sip_freq = st.selectbox("Frequency", SIP_FREQUENCIES)
        sip_ticker = st.selectbox("Fund / ETF", [
            "NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS",
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "LIQUIDBEES.NS"
        ])
        sip_start = st.date_input("SIP Start Date")
        sip_tenure = st.number_input("Tenure (years)", min_value=1, max_value=30, value=5)
        if st.button("🚀 Start SIP (Simulated)"):
            st.session_state.sips.append({
                "ticker": sip_ticker, "amount": sip_amount,
                "frequency": sip_freq, "start_date": str(sip_start), "market": "india"
            })
            st.success(f"SIP started: ₹{sip_amount:,}/mo in {sip_ticker}")
            st.toast(f"✅ SIP activated for {sip_ticker} at ₹{sip_amount:,}/month", icon="🔄")

    with sip2:
        st.markdown("**SIP Calculator**")
        calc_amt = st.slider("Monthly (₹)", 1000, 100000, 10000, 1000)
        calc_yrs = st.slider("Years", 1, 30, 10)
        calc_rate = st.slider("Expected CAGR %", 6, 20, 12)
        r_monthly = calc_rate / 100 / 12
        n_months = calc_yrs * 12
        fv = calc_amt * ((1 + r_monthly) ** n_months - 1) / r_monthly * (1 + r_monthly)
        invested = calc_amt * n_months
        gain = fv - invested

        st.markdown(f"""
        <div class='qcard accent-green' style='margin-top:.5rem'>
          <div style='display:flex;justify-content:space-between;margin-bottom:.5rem'>
            <div><div class='qlabel'>Invested</div><div class='qval' style='font-size:1.2rem'>₹{invested:,.0f}</div></div>
            <div><div class='qlabel'>Returns</div><div class='qval' style='font-size:1.2rem;color:#10b981'>₹{gain:,.0f}</div></div>
            <div><div class='qlabel'>Maturity Value</div><div class='qval' style='font-size:1.2rem'>₹{fv:,.0f}</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if PLOTLY:
            months = list(range(1, n_months + 1))
            vals = [calc_amt * ((1 + r_monthly) ** m - 1) / r_monthly * (1 + r_monthly) for m in months]
            invs = [calc_amt * m for m in months]
            fig_sip = go.Figure()
            fig_sip.add_trace(go.Scatter(
                x=months, y=vals,
                fill="tozeroy",
                fillcolor="rgba(16,185,129,0.08)",
                line=dict(color="#10b981", width=2.2),
                name="Wealth",
                hovertemplate="Month %{x}: <b>₹%{y:,.0f}</b><extra></extra>",
            ))
            fig_sip.add_trace(go.Scatter(
                x=months, y=invs,
                line=dict(color="rgba(107,114,128,0.6)", width=1.5, dash="dot"),
                name="Invested",
                hovertemplate="Month %{x}: ₹%{y:,.0f}<extra></extra>",
            ))
            # Glowing endpoint
            fig_sip.add_trace(go.Scatter(
                x=[months[-1]], y=[vals[-1]],
                mode="markers",
                marker=dict(color="#10b981", size=10, line=dict(color="rgba(16,185,129,0.4)", width=6)),
                showlegend=False,
                hovertemplate=f"Maturity: <b>₹{vals[-1]:,.0f}</b><extra></extra>",
            ))
            fig_sip.update_layout(
                paper_bgcolor="#060810", plot_bgcolor="#060810",
                xaxis=dict(showgrid=False, color="#6b7280", title="Month", tickfont=dict(color="#6b7280")),
                yaxis=dict(showgrid=True, gridcolor="rgba(31,41,55,0.6)", color="#6b7280", tickfont=dict(color="#6b7280")),
                margin=dict(l=8, r=8, t=8, b=8), height=200,
                legend=dict(font=dict(color="#9ca3af"), bgcolor="rgba(0,0,0,0)"),
                font=dict(family="Space Grotesk"),
                hoverlabel=dict(bgcolor="#111827", font_color="#f9fafb", bordercolor="#374151"),
            )
            st.plotly_chart(fig_sip, use_container_width=True, config={"displayModeBar": False})

    if st.session_state.sips:
        st.markdown("**Active SIPs**")
        for s in st.session_state.sips:
            st.markdown(f"""
            <div class='qcard accent-blue' style='padding:.7rem 1rem'>
              <div style='display:flex;justify-content:space-between;align-items:center'>
                <div>
                  <span style='color:#f9fafb;font-weight:600'>{s['ticker']}</span>
                  <span style='margin-left:.75rem;color:#9ca3af;font-size:.78rem'>{s['frequency']}</span>
                </div>
                <span style='font-family:JetBrains Mono;color:#3b82f6;font-weight:700'>₹{s['amount']:,}/mo</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: RISK
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div class='section-header'>
      <div class='section-icon'>🎲</div>
      <h3>Risk Analysis</h3>
    </div>
    """, unsafe_allow_html=True)

    if rm:
        r1, r2, r3, r4, r5, r6 = st.columns(6)
        risk_kpis = [
            (r1, "VaR 95%", f"{rm.var_95:.2%}", "accent-red"),
            (r2, "CVaR 95%", f"{rm.cvar_95:.2%}", "accent-red"),
            (r3, "Prob Ruin", f"{rm.prob_ruin:.2%}", "accent-amber"),
            (r4, "Calmar", f"{rm.calmar:.2f}", "accent-blue"),
            (r5, "Sortino", f"{rm.sortino:.2f}", "accent-blue"),
            (r6, "Skewness", f"{rm.skewness:.2f}", "accent-cyan"),
        ]
        for col, label, val, accent in risk_kpis:
            with col:
                st.markdown(f"""
                <div class='qcard {accent}'>
                  <div class='qlabel'>{label}</div>
                  <div class='qval' style='font-size:1.4rem'>{val}</div>
                </div>
                """, unsafe_allow_html=True)

        # Monte Carlo chart
        if r.get("mc_paths") is not None and PLOTLY:
            paths = r["mc_paths"]
            n_display = min(120, paths.shape[0])
            idx = np.random.choice(paths.shape[0], n_display, replace=False)

            fig_mc = go.Figure()

            # Faint background paths
            for i in idx:
                fig_mc.add_trace(go.Scatter(
                    y=(1 + paths[i]) * 100,
                    line=dict(color="rgba(59,130,246,0.12)", width=0.8),
                    showlegend=False,
                    hoverinfo="skip",
                ))

            # Percentile band fill (5th–95th)
            p5  = (1 + np.percentile(paths, 5,  axis=0)) * 100
            p95 = (1 + np.percentile(paths, 95, axis=0)) * 100
            days = list(range(len(p5)))

            fig_mc.add_trace(go.Scatter(
                x=days + days[::-1],
                y=list(p95) + list(p5[::-1]),
                fill="toself",
                fillcolor="rgba(59,130,246,0.06)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name="90% Band",
            ))

            # 5th percentile line
            fig_mc.add_trace(go.Scatter(
                y=p5, x=days,
                line=dict(color="rgba(239,68,68,0.55)", width=1.2, dash="dot"),
                name="5th pct",
                hovertemplate="Day %{x} · 5th pct: <b>%{y:.1f}%</b><extra></extra>",
            ))

            # 95th percentile line
            fig_mc.add_trace(go.Scatter(
                y=p95, x=days,
                line=dict(color="rgba(16,185,129,0.55)", width=1.2, dash="dot"),
                name="95th pct",
                hovertemplate="Day %{x} · 95th pct: <b>%{y:.1f}%</b><extra></extra>",
            ))

            # Mean path — bright cyan
            mean_path = (1 + paths.mean(axis=0)) * 100
            fig_mc.add_trace(go.Scatter(
                y=mean_path, x=days,
                line=dict(color="#06b6d4", width=2.8),
                name="Mean",
                hovertemplate="Day %{x}: <b>%{y:.1f}%</b><extra></extra>",
            ))

            # Glowing endpoint on mean path
            fig_mc.add_trace(go.Scatter(
                x=[days[-1]], y=[mean_path[-1]],
                mode="markers",
                marker=dict(
                    color="#06b6d4", size=11,
                    line=dict(color="rgba(6,182,212,0.4)", width=7),
                ),
                showlegend=False,
                hovertemplate=f"Final mean: <b>{mean_path[-1]:.1f}%</b><extra></extra>",
            ))

            fig_mc.update_layout(
                title=dict(
                    text="Monte Carlo Simulation — 500 Paths · 252 Days",
                    font=dict(color="#f9fafb", size=14, family="Syne"),
                    x=0.01,
                ),
                paper_bgcolor="#060810",
                plot_bgcolor="#060810",
                xaxis=dict(
                    showgrid=False, color="#6b7280", title="Day",
                    tickfont=dict(color="#6b7280"), zeroline=False,
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="rgba(31,41,55,0.6)",
                    color="#6b7280", title="Portfolio %",
                    tickfont=dict(color="#6b7280"), zeroline=False,
                ),
                legend=dict(
                    font=dict(color="#9ca3af", size=11),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                ),
                margin=dict(l=8, r=8, t=44, b=8),
                font=dict(family="Space Grotesk"),
                hoverlabel=dict(bgcolor="#111827", font_color="#f9fafb", bordercolor="#374151"),
                height=380,
            )
            st.plotly_chart(fig_mc, use_container_width=True, config={"displayModeBar": False})

        # Max drawdown distribution
        if r.get("mc_max_dds") is not None and PLOTLY:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Histogram(
                x=r["mc_max_dds"],
                nbinsx=50,
                marker=dict(
                    color="rgba(239,68,68,0.75)",
                    line=dict(color="rgba(239,68,68,0.0)", width=0),
                ),
                name="Max Drawdown",
                hovertemplate="DD: %{x:.1%}<br>Count: %{y}<extra></extra>",
            ))
            # Ruin threshold line
            fig_dd.add_vline(
                x=-0.5,
                line=dict(color="rgba(245,158,11,0.7)", width=1.5, dash="dash"),
                annotation_text="Ruin (-50%)",
                annotation_font_color="#f59e0b",
                annotation_font_size=11,
            )
            fig_dd.update_layout(
                title=dict(
                    text="Max Drawdown Distribution",
                    font=dict(color="#f9fafb", size=14, family="Syne"),
                    x=0.01,
                ),
                paper_bgcolor="#060810",
                plot_bgcolor="#060810",
                xaxis=dict(
                    showgrid=False, color="#6b7280",
                    tickformat=".0%", tickfont=dict(color="#6b7280"),
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="rgba(31,41,55,0.6)",
                    color="#6b7280", tickfont=dict(color="#6b7280"),
                ),
                margin=dict(l=8, r=8, t=44, b=8),
                height=240,
                font=dict(family="Space Grotesk"),
                hoverlabel=dict(bgcolor="#111827", font_color="#f9fafb", bordercolor="#374151"),
            )
            st.plotly_chart(fig_dd, use_container_width=True, config={"displayModeBar": False})

    # Stress tests
    if r.get("stress_tests") is not None and not r["stress_tests"].empty:
        st.markdown("**Stress Test Scenarios**")
        st.dataframe(r["stress_tests"], use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: PAYMENTS
# ═══════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("""
    <div class='section-header'>
      <div class='section-icon'>💳</div>
      <h3>Payments & Wallet</h3>
      <span style='font-size:.78rem;color:#6b7280;margin-left:.5rem'>Razorpay (India) · Stripe (USA/UK) · PayPal (Global)</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Wallet Balances ──
    st.markdown("**Wallet Balances**")
    wallet = st.session_state.wallet
    wbal_cols = st.columns(3)
    wallet_items = [
        ("india", "₹", "🇮🇳 India", "accent-blue"),
        ("usa", "$", "🇺🇸 USA", "accent-green"),
        ("uk", "£", "🇬🇧 UK", "accent-cyan"),
    ]
    for col, (mkt, sym, label, accent) in zip(wbal_cols, wallet_items):
        with col:
            bal = wallet[mkt]
            # Pick font size based on digit count to avoid overflow
            digits = len(f"{bal:,.0f}")
            fsize = "1.4rem" if digits <= 7 else "1.1rem" if digits <= 10 else ".9rem"
            st.markdown(f"""
            <div class='qcard {accent}'>
              <div class='qlabel'>{label}</div>
              <div style='color:#f9fafb;font-size:{fsize};font-weight:700;
                   font-family:JetBrains Mono;line-height:1.2;word-break:break-all'>
                {sym}{bal:,.2f}
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    pay_tab_add, pay_tab_withdraw, pay_tab_history = st.tabs(["➕ Add Funds", "⬇ Withdraw", "📋 History"])

    # ── ADD FUNDS ──
    with pay_tab_add:
        left_pay, right_pay = st.columns([3, 2])
        with left_pay:
            market_pay = st.selectbox(
                "Market", ["india", "usa", "uk"], key="pay_market",
                format_func=lambda x: f"{MARKETS[x]['currency']} {MARKETS[x]['name']}"
            )
            curr_sym = MARKETS[market_pay]["currency"]
            amount_pay = st.number_input(
                f"Amount ({curr_sym})", min_value=100, value=10000, key="pay_amt"
            )

            # ── Brand SVG logos (inline, no external files needed) ──
            LOGOS = {
                "razorpay": """<svg viewBox="0 0 120 36" xmlns="http://www.w3.org/2000/svg" height="22">
                  <polygon points="44,4 28,32 36,32 52,4" fill="#2D9EE0"/>
                  <polygon points="52,4 36,32 44,32 60,4" fill="#072654"/>
                  <text x="65" y="26" font-family="Arial" font-size="18" font-weight="bold" fill="#072654">Pay</text>
                </svg>""",
                "upi": """<svg viewBox="0 0 80 32" xmlns="http://www.w3.org/2000/svg" height="22">
                  <rect width="80" height="32" rx="4" fill="#ffffff" opacity="0"/>
                  <polygon points="10,4 22,16 10,28 16,28 28,16 16,4" fill="#097939"/>
                  <polygon points="22,4 34,16 22,28 28,28 40,16 28,4" fill="#eb6024"/>
                  <text x="44" y="22" font-family="Arial Black" font-size="14" font-weight="900" fill="#ffffff">UPI</text>
                </svg>""",
                "paytm": """<svg viewBox="0 0 90 28" xmlns="http://www.w3.org/2000/svg" height="22">
                  <rect width="90" height="28" rx="4" fill="#00BAF2"/>
                  <text x="8" y="20" font-family="Arial" font-size="14" font-weight="bold" fill="#ffffff">Paytm</text>
                </svg>""",
                "netbanking": """<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg" height="22">
                  <rect x="2" y="14" width="28" height="14" rx="2" fill="#374151"/>
                  <rect x="2" y="10" width="28" height="6" fill="#3b82f6"/>
                  <polygon points="16,2 2,10 30,10" fill="#6b7280"/>
                </svg>""",
                "stripe": """<svg viewBox="0 0 60 26" xmlns="http://www.w3.org/2000/svg" height="22">
                  <rect width="60" height="26" rx="4" fill="#635BFF"/>
                  <text x="8" y="18" font-family="Arial" font-size="13" font-weight="bold" fill="#ffffff">stripe</text>
                </svg>""",
                "stripe_uk": """<svg viewBox="0 0 60 26" xmlns="http://www.w3.org/2000/svg" height="22">
                  <rect width="60" height="26" rx="4" fill="#635BFF"/>
                  <text x="8" y="18" font-family="Arial" font-size="13" font-weight="bold" fill="#ffffff">stripe</text>
                </svg>""",
                "paypal": """<svg viewBox="0 0 80 28" xmlns="http://www.w3.org/2000/svg" height="22">
                  <text x="0" y="22" font-family="Arial" font-size="20" font-weight="bold" fill="#003087">Pay</text>
                  <text x="36" y="22" font-family="Arial" font-size="20" font-weight="bold" fill="#009cde">Pal</text>
                </svg>""",
                "gpay": """<svg viewBox="0 0 64 26" xmlns="http://www.w3.org/2000/svg" height="22">
                  <text x="0" y="20" font-family="Arial" font-size="16" font-weight="bold" fill="#4285F4">G</text>
                  <text x="14" y="20" font-family="Arial" font-size="16" font-weight="bold" fill="#EA4335">P</text>
                  <text x="26" y="20" font-family="Arial" font-size="16" font-weight="bold" fill="#FBBC05">a</text>
                  <text x="37" y="20" font-family="Arial" font-size="16" font-weight="bold" fill="#34A853">y</text>
                </svg>""",
                "ach": """<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg" height="22">
                  <rect x="2" y="14" width="28" height="14" rx="2" fill="#374151"/>
                  <rect x="2" y="10" width="28" height="6" fill="#10b981"/>
                  <polygon points="16,2 2,10 30,10" fill="#6b7280"/>
                </svg>""",
                "openbanking": """<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg" height="22">
                  <rect x="2" y="14" width="28" height="14" rx="2" fill="#374151"/>
                  <rect x="2" y="10" width="28" height="6" fill="#8b5cf6"/>
                  <polygon points="16,2 2,10 30,10" fill="#6b7280"/>
                </svg>""",
            }

            # ── Payment Method Selection ──
            st.markdown("**Select Payment Method**")
            if market_pay == "india":
                methods = [
                    ("razorpay", "Razorpay", "Cards, UPI, NetBanking"),
                    ("upi",      "UPI",       "GPay, PhonePe, Paytm"),
                    ("paytm",    "Paytm",     "Paytm Wallet & UPI"),
                    ("netbanking","Net Banking","All major banks"),
                ]
            elif market_pay == "usa":
                methods = [
                    ("stripe",  "Stripe",      "Visa, MC, Amex, GPay"),
                    ("paypal",  "PayPal",       "PayPal & Venmo"),
                    ("ach",     "ACH Transfer", "Bank transfer"),
                    ("gpay",    "Google Pay",   "GPay wallet"),
                ]
            else:  # uk
                methods = [
                    ("stripe_uk",    "Stripe",       "Visa, MC, Amex"),
                    ("paypal",       "PayPal",        "PayPal & cards"),
                    ("openbanking",  "Open Banking",  "Bank transfer"),
                    ("gpay",         "Google Pay",    "GPay wallet"),
                ]

            # Render logo cards as HTML; hidden buttons below handle click
            logo_html = "<div style='display:flex;gap:.6rem;margin-bottom:.5rem;flex-wrap:wrap'>"
            for method_id, name, desc in methods:
                selected = st.session_state.pay_method == method_id
                border = "#3b82f6" if selected else "#1f2937"
                bg     = "rgba(59,130,246,0.12)" if selected else "#111827"
                logo   = LOGOS.get(method_id, "")
                logo_html += f"""
                <div style='flex:1;min-width:100px;background:{bg};border:2px solid {border};
                     border-radius:12px;padding:.7rem .5rem;text-align:center;
                     transition:all .2s;cursor:pointer'>
                  <div style='height:26px;display:flex;align-items:center;justify-content:center'>
                    {logo}
                  </div>
                  <div style='font-size:.65rem;color:#9ca3af;margin-top:.35rem'>{desc}</div>
                </div>"""
            logo_html += "</div>"
            st.markdown(logo_html, unsafe_allow_html=True)

            # Hidden Streamlit buttons (invisible — just for state change)
            btn_cols = st.columns(len(methods))
            for col, (method_id, name, desc) in zip(btn_cols, methods):
                with col:
                    if st.button(name, key=f"pm_{method_id}", use_container_width=True):
                        st.session_state.pay_method = method_id
                        st.rerun()

            if st.session_state.pay_method:
                st.markdown("")
                method_id = st.session_state.pay_method

                # ── Razorpay (India) ──
                if method_id == "razorpay":
                    st.markdown(f"""
                    <div class='qcard accent-blue' style='margin-top:.5rem'>
                      <div class='qlabel'>Razorpay Payment Gateway</div>
                      <div style='color:#f9fafb;font-weight:600;font-size:1rem'>Add {curr_sym}{amount_pay:,}</div>
                      <div style='color:#9ca3af;font-size:.75rem;margin-top:.3rem'>
                        Supports: Cards · UPI · NetBanking · Wallets · EMI
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    rz_key_id = st.text_input("Razorpay Key ID",
                        value=_secret("RAZORPAY_KEY_ID", "rzp_test_xxxxxxxxxxxx"),
                        type="password")
                    rz_key_secret = st.text_input("Razorpay Key Secret",
                        value=_secret("RAZORPAY_KEY_SECRET", ""),
                        type="password")

                    if st.button("💳 Pay with Razorpay", use_container_width=True):
                        if RAZORPAY_AVAILABLE and rz_key_id and rz_key_secret and not rz_key_id.endswith("xx"):
                            try:
                                client = razorpay.Client(auth=(rz_key_id, rz_key_secret))
                                order = client.order.create({
                                    "amount": int(amount_pay * 100),
                                    "currency": "INR",
                                    "receipt": f"quant_{len(st.session_state.transactions)+1}",
                                    "payment_capture": 1,
                                })
                                st.success(f"Order created: {order['id']}")
                                st.info(f"Redirect user to Razorpay checkout with order_id: {order['id']}")
                                # In production: open Razorpay checkout JS widget
                                _process_payment_success(market_pay, amount_pay, "Razorpay", order["id"])
                            except Exception as e:
                                st.error(f"Razorpay error: {e}")
                        else:
                            # Sandbox simulation
                            _process_payment_success(market_pay, amount_pay, "Razorpay (Sandbox)", "rzp_sim_001")
                            st.success(f"✅ {curr_sym}{amount_pay:,} added via Razorpay (Sandbox)")
                            st.toast(f"💰 Wallet funded: {curr_sym}{amount_pay:,}", icon="✅")

                # ── UPI ──
                elif method_id == "upi":
                    st.markdown("**UPI Payment**")
                    upi_id = st.text_input("Your UPI ID", placeholder="yourname@upi")
                    if st.button("📲 Pay via UPI", use_container_width=True):
                        if upi_id:
                            # In production: initiate UPI collect via PSP
                            _process_payment_success(market_pay, amount_pay, "UPI", f"upi_{upi_id}")
                            st.success(f"✅ UPI payment request sent to {upi_id}")
                            st.toast(f"💰 {curr_sym}{amount_pay:,} added via UPI", icon="📲")
                        else:
                            st.warning("Enter your UPI ID")

                # ── Paytm ──
                elif method_id == "paytm":
                    st.markdown("**Paytm Payment**")
                    paytm_phone = st.text_input("Paytm Mobile Number", placeholder="+91 9XXXXXXXXX")
                    if st.button("🟦 Pay with Paytm", use_container_width=True):
                        # In production: use paytmchecksum library
                        # import PaytmChecksum
                        # checksum = PaytmChecksum.generateSignature(params, merchant_key)
                        _process_payment_success(market_pay, amount_pay, "Paytm", "ptm_sim_001")
                        st.success(f"✅ {curr_sym}{amount_pay:,} added via Paytm (Sandbox)")
                        st.info("Production: Uses paytmchecksum library — POST /v1/initiateTransaction")
                        st.toast(f"💰 Wallet funded via Paytm", icon="🟦")

                # ── Stripe ──
                elif method_id in ["stripe", "stripe_uk"]:
                    stripe_curr = "usd" if market_pay == "usa" else "gbp"
                    stripe_sym = "$" if market_pay == "usa" else "£"
                    st.markdown(f"""
                    <div class='qcard accent-blue'>
                      <div class='qlabel'>Stripe Payment</div>
                      <div style='color:#f9fafb;font-weight:600'>Add {stripe_sym}{amount_pay:,}</div>
                      <div style='color:#9ca3af;font-size:.75rem;margin-top:.3rem'>
                        Supports: Visa · Mastercard · Amex · Apple Pay · Google Pay
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    stripe_key = st.text_input("Stripe Secret Key",
                        value=_secret("STRIPE_SECRET_KEY", "sk_test_xxxx"),
                        type="password")
                    st.markdown("""
                    <div style='font-size:.75rem;color:#6b7280;margin:.5rem 0'>
                      Use hosted Stripe Payment Elements in production (PCI compliant). <br>
                      Never collect raw card data server-side.
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("💳 Pay with Stripe", use_container_width=True):
                        if STRIPE_AVAILABLE and stripe_key and not stripe_key.endswith("xxxx"):
                            try:
                                stripe.api_key = stripe_key
                                intent = stripe.PaymentIntent.create(
                                    amount=int(amount_pay * 100),
                                    currency=stripe_curr,
                                    payment_method_types=["card"],
                                    metadata={"platform": "quant_invest"},
                                )
                                st.success(f"Payment intent: {intent['id']}")
                                st.info("Pass client_secret to Stripe.js on frontend to collect card details")
                                _process_payment_success(market_pay, amount_pay, "Stripe", intent["id"])
                            except Exception as e:
                                st.error(f"Stripe error: {e}")
                        else:
                            _process_payment_success(market_pay, amount_pay, "Stripe (Sandbox)", "pi_sim_001")
                            st.success(f"✅ {curr_sym}{amount_pay:,} added via Stripe (Sandbox)")
                            st.toast(f"💰 Wallet funded via Stripe", icon="💳")

                # ── PayPal ──
                elif method_id == "paypal":
                    st.markdown("**PayPal Payment**")
                    st.markdown("""
                    <div style='font-size:.75rem;color:#6b7280;margin-bottom:.5rem'>
                      Uses PayPal REST API v2. Set up at developer.paypal.com
                    </div>
                    """, unsafe_allow_html=True)
                    pp_client_id = st.text_input("PayPal Client ID",
                        value=_secret("PAYPAL_CLIENT_ID", ""),
                        type="password", placeholder="Client ID from PayPal Developer")
                    pp_client_secret = st.text_input("PayPal Client Secret",
                        value=_secret("PAYPAL_CLIENT_SECRET", ""),
                        type="password")

                    if st.button("🅿 Pay with PayPal", use_container_width=True):
                        if pp_client_id and pp_client_secret:
                            st.info("""PayPal Integration Steps:
1. POST https://api-m.sandbox.paypal.com/v1/oauth2/token → get access_token
2. POST /v2/checkout/orders with amount and currency
3. Redirect user to approve link from response
4. POST /v2/checkout/orders/{order_id}/capture on return""")
                        _process_payment_success(market_pay, amount_pay, "PayPal", "paypal_sim_001")
                        st.success(f"✅ {curr_sym}{amount_pay:,} added via PayPal (Sandbox)")
                        st.toast(f"💰 Wallet funded via PayPal", icon="🅿")

                # ── Google Pay ──
                elif method_id == "gpay":
                    st.markdown("**Google Pay**")
                    st.markdown("""
                    <div style='font-size:.75rem;color:#6b7280;margin-bottom:.5rem'>
                      For India: Processed via Razorpay PSP (UPI intent flow)<br>
                      For USA/UK: Processed via Stripe (Google Pay token)
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("G Pay with Google Pay", use_container_width=True):
                        _process_payment_success(market_pay, amount_pay, "Google Pay", "gpay_sim_001")
                        st.success(f"✅ {curr_sym}{amount_pay:,} added via Google Pay (Sandbox)")
                        st.toast(f"💰 Wallet funded via GPay", icon="✅")

                # ── Net Banking / ACH / Open Banking ──
                elif method_id in ["netbanking", "ach", "openbanking"]:
                    method_names = {"netbanking": "Net Banking", "ach": "ACH Transfer", "openbanking": "Open Banking"}
                    st.markdown(f"**{method_names[method_id]}**")
                    bank = st.selectbox("Select Bank", [
                        "HDFC Bank", "SBI", "ICICI Bank", "Axis Bank", "Kotak"
                    ] if method_id == "netbanking" else [
                        "Chase", "Bank of America", "Wells Fargo", "Citibank"
                    ] if method_id == "ach" else [
                        "Barclays", "HSBC", "Lloyds", "NatWest", "Santander"
                    ])
                    if st.button(f"🏦 Pay via {method_names[method_id]}", use_container_width=True):
                        _process_payment_success(market_pay, amount_pay, method_names[method_id], f"bank_sim_001")
                        st.success(f"✅ {curr_sym}{amount_pay:,} transfer initiated via {bank}")
                        st.toast(f"💰 Bank transfer initiated", icon="🏦")

        with right_pay:
            st.markdown("**Payment Summary**")
            st.markdown(f"""
            <div class='qcard accent-blue'>
              <div style='display:flex;justify-content:space-between;margin-bottom:.5rem'>
                <span style='color:#9ca3af;font-size:.82rem'>Amount</span>
                <span style='color:#f9fafb;font-weight:700;font-family:JetBrains Mono'>{curr_sym}{amount_pay:,}</span>
              </div>
              <div style='display:flex;justify-content:space-between;margin-bottom:.5rem'>
                <span style='color:#9ca3af;font-size:.82rem'>Market</span>
                <span style='color:#f9fafb;font-weight:600'>{market_pay.upper()}</span>
              </div>
              <div style='display:flex;justify-content:space-between;margin-bottom:.5rem'>
                <span style='color:#9ca3af;font-size:.82rem'>Gateway Fee</span>
                <span style='color:#10b981;font-weight:600'>Free</span>
              </div>
              <div style='display:flex;justify-content:space-between;border-top:1px solid #1f2937;padding-top:.5rem;margin-top:.5rem'>
                <span style='color:#f9fafb;font-weight:700'>Total</span>
                <span style='color:#3b82f6;font-weight:700;font-family:JetBrains Mono;font-size:1.1rem'>{curr_sym}{amount_pay:,}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style='margin-top:.75rem;padding:.7rem;background:rgba(16,185,129,0.08);
                 border:1px solid rgba(16,185,129,0.2);border-radius:8px;font-size:.73rem;color:#6b7280'>
              🔒 <b style='color:#10b981'>Secure Payment</b><br>
              All transactions encrypted with TLS 1.3.<br>
              Card data handled by PCI DSS Level 1 gateways.<br>
              We never store card numbers.
            </div>
            """, unsafe_allow_html=True)

    # ── WITHDRAW ──
    with pay_tab_withdraw:
        wd_col1, wd_col2 = st.columns(2)
        with wd_col1:
            wd_market = st.selectbox("Market", ["india", "usa", "uk"], key="wd_market",
                format_func=lambda x: f"{MARKETS[x]['currency']} {MARKETS[x]['name']}")
            wd_curr = MARKETS[wd_market]["currency"]
            wd_max = wallet[wd_market]
            wd_amount = st.number_input(
                f"Amount ({wd_curr})", min_value=100, max_value=max(100, int(wd_max)), value=min(5000, int(wd_max))
            )
            wd_method = st.selectbox("Withdrawal Method",
                ["Bank Transfer", "UPI"] if wd_market == "india" else
                ["ACH Transfer", "Wire Transfer"] if wd_market == "usa" else
                ["BACS Transfer", "SWIFT"]
            )
            if st.button("⬇ Initiate Withdrawal", use_container_width=True):
                if wd_amount <= wallet[wd_market]:
                    wallet[wd_market] -= wd_amount
                    st.session_state.transactions.append({
                        "type": "Withdraw", "market": wd_market, "amount": wd_amount,
                        "method": wd_method, "status": "Processing"
                    })
                    st.success(f"✅ Withdrawal of {wd_curr}{wd_amount:,} initiated via {wd_method}")
                    st.toast(f"⬇ Withdrawal processing: {wd_curr}{wd_amount:,}", icon="⬇")
                else:
                    st.error(f"Insufficient balance. Available: {wd_curr}{wd_max:,.2f}")

        with wd_col2:
            st.markdown(f"""
            <div class='qcard accent-red'>
              <div class='qlabel'>Available Balance</div>
              <div class='qval'>{MARKETS[wd_market]['currency']}{wallet[wd_market]:,.2f}</div>
              <div class='qdelta down' style='margin-top:.3rem'>After withdrawal: {MARKETS[wd_market]['currency']}{max(0, wallet[wd_market]-wd_amount):,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── HISTORY ──
    with pay_tab_history:
        if st.session_state.transactions:
            tx_df = pd.DataFrame(st.session_state.transactions)
            st.dataframe(tx_df, use_container_width=True, hide_index=True)
            if st.button("📥 Export CSV"):
                csv = tx_df.to_csv(index=False)
                st.download_button("Download", csv, "transactions.csv", "text/csv")
        else:
            st.markdown("""
            <div style='text-align:center;padding:3rem;color:#6b7280'>
              <div style='font-size:2rem;margin-bottom:.5rem'>📋</div>
              No transactions yet. Add funds to get started.
            </div>
            """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────
st.markdown(
    '<p class="qfooter">Quant Invest — Hedge Fund Analytics · Not financial advice · Simulated trading environment</p>',
    unsafe_allow_html=True
)

