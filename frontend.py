import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import time

st.set_page_config(page_title="AI Market Prediction Dashboard", layout="wide")
st.title("📈 Machine Learning Market Prediction Dashboard")

# Update this URL once your backend is deployed on Render
BACKEND_URL = "https://your-backend-name.onrender.com/api/v1/predict"

st.sidebar.header("Stock Selection")
ticker_symbol = st.sidebar.text_input("Enter Ticker (e.g. AAPL, RELIANCE.NS, ^NSEI)", value="AAPL")
period = st.sidebar.selectbox("History Period", ["3mo", "6mo", "1y"], index=0)
interval = st.sidebar.selectbox("Interval", ["1d", "1h"], index=0)

@st.cache_data(ttl=3600)
def fetch_ohlc(symbol, p, i):
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

df_ohlc = fetch_ohlc(ticker_symbol, period, interval)

if df_ohlc.empty or len(df_ohlc) < 30:
    st.warning("Could not fetch enough market data. Please wait a moment or change the ticker.")
else:
    bars_payload = df_ohlc.to_dict(orient="records")
    payload = {"symbol": ticker_symbol, "timeframe": interval, "bars": bars_payload}

    with st.spinner("Analyzing market structure..."):
        try:
            res = requests.post(BACKEND_URL, json=payload, timeout=15)
            if res.status_code == 200:
                pred = res.json()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Signal", pred["signal"], delta=f"{pred['probability']*100:.1f}% Score")
                col2.metric("RSI (14)", f"{pred['market_context']['rsi']:.2f}")
                col3.metric("ADX", f"{pred['market_context']['adx']:.2f}")
                col4.metric("Put-Call Ratio (PCR)", f"{pred['market_context']['pcr']:.2f}")

                st.subheader("Interactive Price Chart")
                fig = go.Figure(data=[go.Candlestick(
                    x=df_ohlc['timestamp'], open=df_ohlc['open'],
                    high=df_ohlc['high'], low=df_ohlc['low'], close=df_ohlc['close']
                )])
                fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(f"Backend API Response Error: {res.text}")
        except Exception as e:
            st.error(f"Backend connection error: {e}")
