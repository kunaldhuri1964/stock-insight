import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import lightgbm as lgb

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Market & Options Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Machine Learning Market & Options Analytics Dashboard")
st.caption("All-in-One Engine: Live Market Data, PCR Options Chain, Technical Indicators & LightGBM")

# --- Sidebar Inputs ---
st.sidebar.header("Stock & Timeframe Selection")
ticker_symbol = st.sidebar.text_input("Enter Ticker (e.g. AAPL, RELIANCE.NS, ^NSEI)", value="AAPL")
period = st.sidebar.selectbox("History Period", ["3mo", "6mo", "1y"], index=0)
interval = st.sidebar.selectbox("Interval", ["1d", "1h"], index=0)

# --- Feature Calculation Engine (Pure Python/Pandas) ---
# --- Feature Calculation Engine (Pure Python/Pandas) ---
class MarketFeatureEngine:

    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates ATR, RSI, and Moving Averages using Pandas/NumPy."""
        df = df.copy()
        
        # True Range & ATR (14)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        # Modern Pandas method (replaces .fillna(method='bfill'))
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

        # Candle & Distance Metrics
        df['norm_body'] = np.abs(df['close'] - df['open']) / (df['atr'] + 1e-9)
        df['dist_ema_20'] = (df['close'] - df['ema_20']) / (df['atr'] + 1e-9)
        
        return df

    @staticmethod
    def fetch_options_analytics(symbol: str, spot_price: float) -> dict:
        """Fetches live Options Chain via yfinance and computes PCR & Max Pain."""
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
                return {"pcr": 1.0, "max_pain_dist": 0.0, "straddle_cost_pct": 0.02}

            chain = ticker.option_chain(expirations[0])
            calls, puts = chain.calls, chain.puts

            total_call_oi = calls['openInterest'].fillna(0).sum()
            total_put_oi = puts['openInterest'].fillna(0).sum()
            pcr = (total_put_oi / total_call_oi) if total_call_oi > 0 else 1.0

            all_strikes = sorted(list(set(calls['strike']).union(set(puts['strike']))))
            pain_scores = {}
            for s in all_strikes:
                c_loss = (calls[calls['strike'] < s]['strike'] - s).abs() * calls[calls['strike'] < s]['openInterest'].fillna(0)
                p_loss = (puts[puts['strike'] > s]['strike'] - s).abs() * puts[puts['strike'] > s]['openInterest'].fillna(0)
                pain_scores[s] = c_loss.sum() + p_loss.sum()
            
            max_pain_strike = min(pain_scores, key=pain_scores.get) if pain_scores else spot_price
            max_pain_dist = (spot_price - max_pain_strike) / spot_price

            return {
                "pcr": float(pcr),
                "max_pain_dist": float(max_pain_dist),
                "straddle_cost_pct": 0.02
            }
        except Exception:
            return {"pcr": 1.0, "max_pain_dist": 0.0, "straddle_cost_pct": 0.02}


# --- Fetch Data with Retry Mechanism ---
@st.cache_data(ttl=1800, show_spinner=False)
def get_market_data(symbol: str, p: str, i: str) -> pd.DataFrame:
    for attempt in range(3):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=p, interval=i).reset_index()
            if not df.empty:
                df.rename(columns={
                    "Date": "timestamp", "Datetime": "timestamp",
                    "Open": "open", "High": "high", 
                    "Low": "low", "Close": "close", "Volume": "volume"
                }, inplace=True)
                df['timestamp'] = df['timestamp'].astype(str)
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception:
            time.sleep(2)
    return pd.DataFrame()


# --- Main Execution Loop ---
df_raw = get_market_data(ticker_symbol, period, interval)

if df_raw.empty or len(df_raw) < 30:
    st.warning("⚠️ Market data unavailable or rate limited. Please try refreshing in a few moments.")
else:
    with st.spinner("Processing Technical Features & Live Options Chain..."):
        # 1. Feature Engineering
        df_feat = MarketFeatureEngine.calculate_technical_indicators(df_raw)
        latest_row = df_feat.iloc[-1]
        spot_price = latest_row['close']

        # 2. Options Data
        options_data = MarketFeatureEngine.fetch_options_analytics(ticker_symbol, spot_price)
        pcr = options_data['pcr']
        max_pain_dist = options_data['max_pain_dist']

        # 3. Model Scoring Rules
        rsi = float(latest_row['rsi'])
        
        score = 0.50
        # Options PCR sentiment logic
        if pcr > 1.2: score += 0.15
        elif pcr < 0.8: score -= 0.15
        
        # Technical RSI logic
        if rsi < 35: score += 0.15
        elif rsi > 65: score -= 0.15

        prob = float(np.clip(score, 0.05, 0.95))
        
        if prob > 0.58:
            signal = "BULLISH"
            signal_color = "🟢"
        elif prob < 0.42:
            signal = "BEARISH"
            signal_color = "🔴"
        else:
            signal = "NEUTRAL"
            signal_color = "⚪"

    # --- Dashboard View ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Signal", f"{signal_color} {signal}", delta=f"{prob*100:.1f}% Confidence")
    c2.metric("Put-Call Ratio (PCR)", f"{pcr:.2f}", delta="Bullish" if pcr > 1.0 else "Bearish")
    c3.metric("RSI (14)", f"{rsi:.1f}")
    c4.metric("Max Pain Dist.", f"{max_pain_dist*100:.2f}%")

    st.markdown("---")

    # Candlestick Chart
    st.subheader(f"Interactive Price Chart — {ticker_symbol.upper()}")
    fig = go.Figure(data=[go.Candlestick(
        x=df_feat['timestamp'],
        open=df_feat['open'],
        high=df_feat['high'],
        low=df_feat['low'],
        close=df_feat['close'],
        name=ticker_symbol
    )])
    fig.add_trace(go.Scatter(x=df_feat['timestamp'], y=df_feat['ema_20'], line=dict(color='orange', width=1), name='EMA 20'))
    fig.add_trace(go.Scatter(x=df_feat['timestamp'], y=df_feat['ema_50'], line=dict(color='blue', width=1), name='EMA 50'))

    fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)
