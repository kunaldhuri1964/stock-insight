import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="AI Market Prediction Dashboard", layout="wide")
st.title("📈 Machine Learning Market Prediction Dashboard")

# Replace this URL with your actual deployed FastAPI backend URL (e.g., on Render)
BACKEND_URL = "https://your-fastapi-backend.onrender.com/api/v1/predict"

st.sidebar.header("Stock & Timeframe Selection")
ticker_symbol = st.sidebar.text_input("Enter Ticker (e.g. AAPL, RELIANCE.NS, ^NSEI)", value="AAPL")
period = st.sidebar.selectbox("History Period", ["3mo", "6mo", "1y"], index=0)
interval = st.sidebar.selectbox("Timeframe Interval", ["1d", "1h"], index=0)

@st.cache_data(ttl=300)
def fetch_ohlc_data(symbol, p, i):
    stock = yf.Ticker(symbol)
    df = stock.history(period=p, interval=i).reset_index()
    df.rename(columns={
        "Date": "timestamp", "Datetime": "timestamp",
        "Open": "open", "High": "high", 
        "Low": "low", "Close": "close", "Volume": "volume"
    }, inplace=True)
    df['timestamp'] = df['timestamp'].astype(str)
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

try:
    df_ohlc = fetch_ohlc_data(ticker_symbol, period, interval)

    if len(df_ohlc) < 30:
        st.error("Insufficient market bars returned. Select a longer period.")
    else:
        bars_payload = df_ohlc.to_dict(orient="records")
        payload = {
            "symbol": ticker_symbol,
            "timeframe": interval,
            "bars": bars_payload
        }

        with st.spinner("Connecting to FastAPI Prediction Engine..."):
            try:
                response = requests.post(BACKEND_URL, json=payload, timeout=10)
                
                if response.status_code == 200:
                    prediction = response.json()
                    
                    # Metrics Display
                    col1, col2, col3 = st.columns(3)
                    
                    signal = prediction["signal"]
                    prob = prediction["probability"]
                    
                    if signal == "BULLISH":
                        col1.metric("Predicted Signal", "🟢 BULLISH", delta=f"{prob*100:.1f}% Score")
                    elif signal == "BEARISH":
                        col1.metric("Predicted Signal", "🔴 BEARISH", delta=f"{(1-prob)*100:.1f}% Score", delta_color="inverse")
                    else:
                        col1.metric("Predicted Signal", "⚪ NEUTRAL", delta=f"{prob*100:.1f}% Score")

                    col2.metric("RSI (14)", f"{prediction['market_context']['rsi']:.2f}")
                    col3.metric("ADX (Trend Strength)", f"{prediction['market_context']['adx']:.2f}")

                    # Pattern Findings
                    st.subheader("🔍 Active Candlestick Patterns Detected")
                    if prediction["detected_patterns"]:
                        for pattern in prediction["detected_patterns"]:
                            st.info(f"Pattern Detected: **{pattern}**")
                    else:
                        st.write("No major candlestick patterns active on current candle.")

                    # Plotly Candlestick Chart
                    st.subheader("Interactive Price Chart")
                    fig = go.Figure(data=[go.Candlestick(
                        x=df_ohlc['timestamp'],
                        open=df_ohlc['open'],
                        high=df_ohlc['high'],
                        low=df_ohlc['low'],
                        close=df_ohlc['close'],
                        name=ticker_symbol
                    )])
                    fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=500)
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(f"Backend API Error: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach FastAPI server. Please make sure BACKEND_URL is configured correctly.")

except Exception as e:
    st.error(f"Error loading market data: {e}")
