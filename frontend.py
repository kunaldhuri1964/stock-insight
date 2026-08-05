import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# --- Page Configuration ---
st.set_page_config(
    page_title="Indian Market Live AI Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("🇮🇳 Live Indian Market AI Real-Time Dashboard")

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


# --- Options Analytics UI Boxes ---
def render_options_boxes(options_data: dict):
    st.subheader("📊 Options Chain Analytics")
    cols = st.columns(3)
    
    pcr = options_data.get('pcr', 1.0)
    oi_bias = options_data.get('oi_bias', 0.0)
    mean_iv = options_data.get('mean_iv', 0.20) * 100

    # PCR Box
    with cols[0]:
        st.metric(
            label="Put-Call Ratio (PCR)",
            value=f"{pcr:.2f}",
            delta="Bullish sentiment" if pcr > 1.0 else "Bearish sentiment"
        )
        st.caption("PCR > 1.0 indicates strong put writing (bullish market floor).")

    # OI Bias Box
    with cols[1]:
        st.metric(
            label="NTM Open Interest Bias",
            value=f"{oi_bias:+.2f}",
            delta="Call Heavy" if oi_bias < 0 else "Put Heavy"
        )
        st.caption("Measures Near-The-Money option open interest distribution.")

    # Implied Volatility Box
    with cols[2]:
        st.metric(
            label="Mean Implied Volatility (IV)",
            value=f"{mean_iv:.1f}%"
        )
        st.caption("Average implied volatility across near-the-money option contracts.")


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
    
    # Direct live spot price fetch to avoid historical candle delay
    try:
        live_ticker = yf.Ticker(ticker)
        fast_price = live_ticker.fast_info.get('lastPrice', None)
        if fast_price and not np.isnan(fast_price):
            spot_price = float(fast_price)
        else:
            spot_price = float(latest_row['close'])
    except Exception:
        spot_price = float(latest_row['close'])

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

    # --- Header Banner ---
    st.subheader(f"📌 {company_name} (`{ticker}`)")
    st.caption(f"⚡ **LIVE STREAM** | Last Updated: **{latest_row['timestamp']}** | Period: **{p}** | Timeframe: **{i}**")

    # Primary Metrics Display
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Live Spot Price", f"₹{spot_price:,.2f}")
    c2.metric("Predicted Signal", f"{'🟢' if signal_label=='BULLISH' else ('🔴' if signal_label=='BEARISH' else '⚪')} {signal_label}")
    c3.metric("RSI (14)", f"{rsi_val:.1f}")
    c4.metric("Put-Call Ratio (PCR)", f"{pcr:.2f}")

    st.markdown("---")

    # --- RESTORED OPTIONS ANALYTICS 3 BOXES ---
    render_options_boxes(options_data)

    st.markdown("---")

    # --- Interactive Chart ---
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

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        margin=dict(l=20, r=20, t=30, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


# --- EXECUTE THE LIVE FRAGMENT ---
render_live_dashboard(ticker_symbol, period, interval)
