import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Market Prediction & Position Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Market Prediction & Position Dashboard")

# --- Sidebar Inputs ---
st.sidebar.header("Stock & Timeframe Options")
ticker_symbol = st.sidebar.text_input("Enter Ticker (e.g. AAPL, RELIANCE.NS, ^NSEI)", value="AAPL").strip().upper()

# Flexible History & Interval Selections
period = st.sidebar.selectbox("History Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=3)
interval = st.sidebar.selectbox("Candle Interval", ["5m", "15m", "1h", "1d"], index=2)

account_balance = st.sidebar.number_input("Account Capital ($)", value=10000.0, step=1000.0)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=2.0) / 100.0

# --- Feature Calculation Engine ---
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
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            if not expirations:
                return {"pcr": 1.0, "max_pain_dist": 0.0}

            chain = ticker.option_chain(expirations[0])
            calls, puts = chain.calls, chain.puts

            total_call_oi = calls['openInterest'].fillna(0).sum()
            total_put_oi = puts['openInterest'].fillna(0).sum()
            pcr = (total_put_oi / total_call_oi) if total_call_oi > 0 else 1.0

            return {"pcr": float(pcr)}
        except Exception:
            return {"pcr": 1.0}

# --- Position Calculator Engine ---
def calculate_trade_position(spot_price: float, atr: float, signal: str, balance: float, risk: float):
    sl_distance = max(atr * 1.5, spot_price * 0.005)
    tp_distance = sl_distance * 2.0  # 1:2 Risk-Reward Ratio
    risk_amount = balance * risk

    if signal == "BULLISH":
        action = "BUY / LONG"
        stop_loss = spot_price - sl_distance
        target = spot_price + tp_distance
    elif signal == "BEARISH":
        action = "SELL / SHORT"
        stop_loss = spot_price + sl_distance
        target = spot_price - tp_distance
    else:
        action = "NO TRADE (NEUTRAL)"
        stop_loss = spot_price
        target = spot_price

    quantity = int(risk_amount / sl_distance) if sl_distance > 0 else 0
    return {
        "action": action,
        "entry": round(spot_price, 2),
        "stop_loss": round(stop_loss, 2),
        "target": round(target, 2),
        "quantity": max(1, quantity) if signal != "NEUTRAL" else 0
    }

# --- Fetch Market Data ---
@st.cache_data(ttl=300, show_spinner=False)
def get_market_data(symbol: str, p: str, i: str):
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
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                info = stock.info
                long_name = info.get("longName", symbol)
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']], long_name
        except Exception:
            time.sleep(1)
    return pd.DataFrame(), symbol

# --- Execution ---
df_raw, company_name = get_market_data(ticker_symbol, period, interval)

if df_raw.empty or len(df_raw) < 15:
    st.error(f"⚠️ Unable to fetch data for **{ticker_symbol}** with period `{period}` and interval `{interval}`. Please check the symbol or try a longer period.")
else:
    df_feat = MarketFeatureEngine.calculate_technical_indicators(df_raw)
    latest_row = df_feat.iloc[-1]
    spot_price = float(latest_row['close'])
    atr_val = float(latest_row['atr'])
    rsi_val = float(latest_row['rsi'])

    # Options Analysis
    options_data = MarketFeatureEngine.fetch_options_analytics(ticker_symbol, spot_price)
    pcr = options_data['pcr']

    # Signal Logic
    score = 0.50
    if pcr > 1.2 or rsi_val < 35: score += 0.20
    elif pcr < 0.8 or rsi_val > 65: score -= 0.20

    prob = float(np.clip(score, 0.05, 0.95))
    signal = "BULLISH" if prob > 0.55 else ("BEARISH" if prob < 0.45 else "NEUTRAL")

    # Trade Position Details
    pos = calculate_trade_position(spot_price, atr_val, signal, account_balance, risk_pct)

    # --- Header Information ---
    st.subheader(f"📌 {company_name} (`{ticker_symbol}`)")
    st.caption(f"Showing **{len(df_raw)}** candles | Period: **{period}** | Timeframe: **{interval}**")

    # Metrics Display
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${spot_price:.2f}")
    c2.metric("Predicted Signal", f"{'🟢' if signal=='BULLISH' else ('🔴' if signal=='BEARISH' else '⚪')} {signal}")
    c3.metric("RSI (14)", f"{rsi_val:.1f}")
    c4.metric("Put-Call Ratio (PCR)", f"{pcr:.2f}")

    # Position Info Box
    st.info(f"🎯 **Trade Position Plan:** Action: **{pos['action']}** | Entry: **${pos['entry']}** | Target (TP): **${pos['target']}** | Stop Loss (SL): **${pos['stop_loss']}** | Rec. Size: **{pos['quantity']} shares**")

    # --- Plotly Candlestick Chart with Prediction Overlay Lines ---
    fig = go.Figure()

    # 1. Candlestick Trace
    fig.add_trace(go.Candlestick(
        x=df_feat['timestamp'],
        open=df_feat['open'], high=df_feat['high'],
        low=df_feat['low'], close=df_feat['close'],
        name="Price"
    ))

    # 2. Indicators Traces
    fig.add_trace(go.Scatter(x=df_feat['timestamp'], y=df_feat['ema_20'], line=dict(color='orange', width=1.5), name='EMA 20'))
    fig.add_trace(go.Scatter(x=df_feat['timestamp'], y=df_feat['ema_50'], line=dict(color='blue', width=1.5), name='EMA 50'))

    # 3. Horizontal Predicted Lines (Target, Entry, Stop Loss)
    if signal != "NEUTRAL":
        fig.add_hline(y=pos['target'], line_dash="dash", line_color="green", line_width=2,
                      annotation_text=f"Predicted Target (TP): ${pos['target']}", annotation_position="top right")
        
        fig.add_hline(y=pos['stop_loss'], line_dash="dash", line_color="red", line_width=2,
                      annotation_text=f"Stop Loss (SL): ${pos['stop_loss']}", annotation_position="bottom right")

        fig.add_hline(y=pos['entry'], line_dash="solid", line_color="yellow", line_width=1,
                      annotation_text=f"Entry: ${pos['entry']}", annotation_position="top left")

    fig.update_layout(
        title=f"{ticker_symbol} Price Chart with Target & SL Overlay",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)
