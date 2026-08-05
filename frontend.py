import sys
import time
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from backend import HorizonPredictor

# Ensure root folder is in Python Path for direct imports
sys.path.insert(0, str(Path(__file__).resolve().parent))



# --- Page Configuration ---
st.set_page_config(
    page_title="Indian Market Live AI Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("🇮🇳 Live Indian Market AI Real-Time & Horizon Prediction Dashboard")

# --- Initialize Backend Predictor Instance ---
@st.cache_resource
def get_predictor():
    return HorizonPredictor()

predictor = get_predictor()

# --- Top 20 Indian Market Companies (NSE) ---
TOP_20_INDIAN_STOCKS = {
    "RELIANCE.NS": "Reliance Industries Ltd.",
    "TCS.NS": "Tata Consultancy Services Ltd.",
    "HDFCBANK.NS": "HDFC Bank Ltd.",
    "INFY.NS": "Infosys Ltd.",
    "ICICIBANK.NS": "ICICI Bank Ltd.",
    "HINDUNILVR.NS": "Hindustan Unilever Ltd.",
    "ITC.NS": "ITC Ltd.",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel Ltd.",
    "LTIM.NS": "LTIMindtree Ltd.",
    "TATAMOTORS.NS": "Tata Motors Ltd.",
    "TATASTEEL.NS": "Tata Steel Ltd.",
    "AXISBANK.NS": "Axis Bank Ltd.",
    "KOTAKBANK.NS": "Kotak Mahindra Bank Ltd.",
    "LT.NS": "Larsen & Toubro Ltd.",
    "M&M.NS": "Mahindra & Mahindra Ltd.",
    "MARUTI.NS": "Maruti Suzuki India Ltd.",
    "SUNPHARMA.NS": "Sun Pharmaceutical Industries Ltd.",
    "ULTRACEMCO.NS": "UltraTech Cement Ltd.",
    "^NSEI": "NIFTY 50 Index"
}

# --- Sidebar Inputs ---
st.sidebar.header("🔄 Live Data Stream Controls")
auto_refresh = st.sidebar.checkbox("Enable 10-Second Auto-Refresh", value=True)
if auto_refresh:
    st.sidebar.caption("🟢 Live Fragment Stream Active (Zero Screen Blur)")

st.sidebar.header("Indian Stock Selection")

selected_company = st.sidebar.selectbox(
    "Select Indian Stock / Index:",
    options=list(TOP_20_INDIAN_STOCKS.keys()),
    format_func=lambda ticker: f"{TOP_20_INDIAN_STOCKS[ticker]} ({ticker})"
)

use_custom = st.sidebar.checkbox("Or enter another NSE Symbol")
if use_custom:
    custom_input = st.sidebar.text_input("Enter NSE Ticker (e.g. WIPRO, ZOMATO)", value="WIPRO").strip().upper()
    ticker_symbol = custom_input if custom_input.endswith(".NS") or custom_input.startswith("^") else f"{custom_input}.NS"
else:
    ticker_symbol = selected_company

period = st.sidebar.selectbox("History Period", ["1d", "5d", "1mo", "3mo", "1y"], index=1)
interval = st.sidebar.selectbox("Candle Interval", ["1m", "5m", "15m", "1h", "1d"], index=1)


# --- Feature & Options Chain Calculation Engine ---
class MarketFeatureEngine:

    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # True Range & ATR (14)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean().bfill()

        # RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50.0)

        # Exponential Moving Averages
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        return df

    @staticmethod
    def fetch_options_analytics(symbol: str, spot_price: float) -> dict:
        """Deep Options Chain Analysis: Calculates PCR, OI Bias, and Volatility Skew."""
        default_res = {"pcr": 1.0, "oi_bias": 0.0, "max_pain": spot_price, "mean_iv": 0.20}
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            if not expirations:
                return default_res

            chain = ticker.option_chain(expirations[0])
            calls, puts = chain.calls, chain.puts

            if calls.empty or puts.empty:
                return default_res

            total_call_oi = calls['openInterest'].fillna(0).sum()
            total_put_oi = puts['openInterest'].fillna(0).sum()
            pcr = (total_put_oi / total_call_oi) if total_call_oi > 0 else 1.0

            # Filter Near-the-Money (NTM) Options for OI & IV Sentiment Analysis
            calls_ntm = calls[abs(calls['strike'] - spot_price) / spot_price <= 0.05]
            puts_ntm = puts[abs(puts['strike'] - spot_price) / spot_price <= 0.05]

            call_oi_ntm = calls_ntm['openInterest'].fillna(0).sum()
            put_oi_ntm = puts_ntm['openInterest'].fillna(0).sum()
            total_ntm_oi = call_oi_ntm + put_oi_ntm

            # Directional bias based on options positioning (-1.0 to +1.0)
            oi_bias = (put_oi_ntm - call_oi_ntm) / (total_ntm_oi + 1e-9) if total_ntm_oi > 0 else 0.0

            # Average Implied Volatility
            iv_calls = calls_ntm['impliedVolatility'].fillna(0.20).mean()
            iv_puts = puts_ntm['impliedVolatility'].fillna(0.20).mean()
            mean_iv = (iv_calls + iv_puts) / 2.0 if not np.isnan(iv_calls) else 0.20

            return {
                "pcr": float(pcr),
                "oi_bias": float(oi_bias),
                "mean_iv": float(mean_iv)
            }
        except Exception:
            return default_res


# --- Options-Integrated Multi-Hour Real-Time Horizon Predictor UI Component ---

def render_horizon_cards(predictions):
    st.subheader("⏱️ Live Horizon Targets (1H, 2H, 3H)")
    
    # Guard against empty or invalid prediction payload
    if not predictions or not isinstance(predictions, dict):
        st.warning("⚠️ No horizon target data available.")
        return

    cols = st.columns(3)
    horizons = ["1H", "2H", "3H"]
    
    for idx, h in enumerate(horizons):
        data = predictions.get(h, {})
        if not data:
            with cols[idx]:
                st.info(f"### {h} Target\n*Data unavailable*")
            continue
            
        with cols[idx]:
            st.markdown(f"### {h} Target Range")
            
            # Extract values safely with fallback defaults
            center = data.get('center', 0.0)
            spread = data.get('spread', 0.0)
            lower = data.get('lower', 0.0)
            upper = data.get('upper', 0.0)
            
            # Display Target Metrics
            st.metric(label=f"{h} Center Target", value=f"₹{center:,.2f}", delta=f"±₹{spread:,.2f} Range")
            st.info(f"**Expected Range:** ₹{lower:,.2f} — ₹{upper:,.2f}")
            
            # Pattern Triggers Display
            active_patterns = data.get('active_patterns', [])
            if active_patterns:
                st.caption(f"🎯 **Triggers:** {', '.join(active_patterns)}")
            else:
                st.caption("🎯 **Triggers:** Neutral / None")
                
            # Options Sentiment Display
            pcr_sentiment = data.get('options_sentiment', None)
            if pcr_sentiment:
                st.caption(f"📊 **Options Bias:** {pcr_sentiment}")


# --- Fetch Market Data ---
@st.cache_data(ttl=5, show_spinner=False)
def get_market_data(symbol: str, p: str, i: str):
    for attempt in range(2):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=p, interval=i).reset_index()
            if not df.empty:
                df.rename(columns={
                    "Date": "timestamp", "Datetime": "timestamp",
                    "Open": "open", "High": "high", 
                    "Low": "low", "Close": "close", "Volume": "volume"
                }, inplace=True)
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%H:%M:%S (%d %b)')
                info = stock.info
                long_name = info.get("longName", TOP_20_INDIAN_STOCKS.get(symbol, symbol))
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']], long_name
        except Exception:
            time.sleep(0.5)
    return pd.DataFrame(), symbol


# --- ISOLATED LIVE STREAM FRAGMENT (Zero Screen Blur) ---
@st.fragment(run_every="10s" if auto_refresh else None)
def render_live_dashboard(ticker: str, p: str, i: str):
    df_raw, company_name = get_market_data(ticker, p, i)

    if df_raw.empty or len(df_raw) < 10:
        st.error(f"⚠️ Unable to fetch live data for **{ticker}**. Please check market hours or select a larger history period.")
        return

    df_feat = MarketFeatureEngine.calculate_technical_indicators(df_raw)
    latest_row = df_feat.iloc[-1]
    
    # Try fetching fast live price, fallback to historical close
    try:
        spot_price = float(yf.Ticker(ticker).fast_info['lastPrice'])
    except Exception:
        spot_price = float(latest_row['close'])

    atr_val = float(latest_row['atr'])
    rsi_val = float(latest_row['rsi'])

    # Options Chain Analysis
    options_data = MarketFeatureEngine.fetch_options_analytics(ticker, spot_price)
    pcr = options_data['pcr']
    oi_bias = options_data['oi_bias']

    # Signal Logic
    score = 0.50
    if pcr > 1.2: score += 0.15
    elif pcr < 0.8: score -= 0.15

    if oi_bias > 0.2: score += 0.10
    elif oi_bias < -0.2: score -= 0.10

    if rsi_val < 35: score += 0.15
    elif rsi_val > 65: score -= 0.15

    prob = float(np.clip(score, 0.05, 0.95))
    signal = 100 if prob > 0.55 else (-100 if prob < 0.45 else 0)
    signal_label = "BULLISH" if signal > 0 else ("BEARISH" if signal < 0 else "NEUTRAL")

    # --- Predict Hourly Horizons Safely Using Class Instance ---
    try:
        if hasattr(predictor, 'predict_hourly_horizons'):
            horizons = predictor.predict_hourly_horizons(spot_price, atr_val, signal, options_data)
        elif hasattr(predictor, 'predict_live_ranges'):
            horizons = predictor.predict_live_ranges(df_feat)
        else:
            horizons = {}
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        horizons = {}

    # --- Header Banner ---
    st.subheader(f"📌 {company_name} (`{ticker}`)")
    st.caption(f"⚡ **LIVE STREAM** | Last Updated: **{latest_row['timestamp']}** | Period: **{p}** | Timeframe: **{i}**")

    # Metrics Display
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Live Spot Price", f"₹{spot_price:,.2f}")
    c2.metric("Predicted Signal", f"{'🟢' if signal_label=='BULLISH' else ('🔴' if signal_label=='BEARISH' else '⚪')} {signal_label}")
    c3.metric("RSI (14)", f"{rsi_val:.1f}")
    c4.metric("Put-Call Ratio (PCR)", f"{pcr:.2f}")

    st.markdown("---")

    # --- Render Horizon Prediction Cards ---
    render_horizon_cards(horizons)

    # --- Interactive Chart ---
    st.markdown("---")
    st.subheader("Interactive Candlestick Chart (Auto-Refreshing)")

    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df_feat['timestamp'],
        open=df_feat['open'], high=df_feat['high'],
        low=df_feat['low'], close=df_feat['close'],
        name=ticker
    ))

    # Moving Averages
    fig.add_trace(go.Scatter(x=df_feat['timestamp'], y=df_feat['ema_20'], line=dict(color='orange', width=1.5), name='EMA 20'))
    fig.add_trace(go.Scatter(x=df_feat['timestamp'], y=df_feat['ema_50'], line=dict(color='blue', width=1.5), name='EMA 50'))

    # Overlay Horizontal Horizon Lines
    if horizons and signal_label != "NEUTRAL":
        h1_target = horizons.get('1H', {}).get('center', None)
        h2_target = horizons.get('2H', {}).get('center', None)
        h3_target = horizons.get('3H', {}).get('center', None)

        if h1_target:
            fig.add_hline(y=h1_target, line_dash="dot", line_color="#00FF00", line_width=1.5,
                          annotation_text=f"1H Target: ₹{h1_target:,.2f}", annotation_position="top right")
        if h2_target:
            fig.add_hline(y=h2_target, line_dash="dash", line_color="#00DD88", line_width=2,
                          annotation_text=f"2H Target: ₹{h2_target:,.2f}", annotation_position="top right")
        if h3_target:
            fig.add_hline(y=h3_target, line_dash="solid", line_color="#00AAFF", line_width=2.5,
                          annotation_text=f"3H Target: ₹{h3_target:,.2f}", annotation_position="top right")

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        margin=dict(l=20, r=20, t=30, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


# --- EXECUTE THE LIVE FRAGMENT ---
render_live_dashboard(ticker_symbol, period, interval)
