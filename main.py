import numpy as np
import pandas as pd
import lightgbm as lgb
import talib
import yfinance as yf
from backend import HorizonPredictor
from frontend import render_horizon_cards
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

st.set_page_config(page_title="Multi-Asset Pattern Predictor", layout="wide")

# Initialize Backend Predictor
predictor = HorizonPredictor()

# Popular Default Watchlist (Indian & Global Indices / Assets)
DEFAULT_WATCHLIST = {
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK",
    "Sensex": "^BSESN",
    "S&P 500": "^GSPC",
    "Gold (USD)": "GC=F",
    "Crude Oil": "CL=F"
}

def fetch_live_ohlc(symbol, interval="15m", period="5d"):
    """Fetches real-time OHLC data dynamically for any ticker."""
    stock = yf.Ticker(symbol)
    df = stock.history(period=period, interval=interval)
    
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    # Keep standardized columns for TA-Lib
    df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df

# --- DYNAMIC CONTROLS IN SIDEBAR ---
st.sidebar.title("🔍 Market Selector")

# Option 1: Choose from Watchlist Presets
selected_preset = st.sidebar.selectbox("Quick Select Index / Asset:", ["Custom Search"] + list(DEFAULT_WATCHLIST.keys()))

# Option 2: Custom Ticker Input (Supports any NSE stock, Crypto, Forex, US Stock)
if selected_preset == "Custom Search":
    ticker_input = st.sidebar.text_input("Enter Any Stock Symbol (e.g. TATAMOTORS.NS, AAPL, BTC-USD):", "TATAMOTORS.NS")
    active_symbol = ticker_input.strip()
else:
    active_symbol = DEFAULT_WATCHLIST[selected_preset]

interval_choice = st.sidebar.selectbox("Candle Interval", ["5m", "15m", "1h"], index=1)

# --- MAIN DASHBOARD VIEW ---
st.title(f"📊 Live Pattern Horizon Predictor: `{active_symbol}`")

if st.sidebar.button("Run Prediction", type="primary"):
    with st.spinner(f"Scanning market patterns for {active_symbol}..."):
        live_df = fetch_live_ohlc(symbol=active_symbol, interval=interval_choice)

    if not live_df.empty:
        latest_row = live_df.iloc[-1]
        
        # Display Current Asset Banner
        col1, col2, col3 = st.columns(3)
        col1.metric("Selected Symbol", active_symbol)
        col2.metric("Last Price", f"₹{latest_row['Close']:.2f}")
        col3.metric("Last Candle Time", str(latest_row['Datetime']))
        
        st.divider()

        # Pass dynamic data to Backend for pattern detection & prediction
        predictions = predictor.predict_live_ranges(live_df)
        
        # Display dynamic range cards
        render_horizon_cards(predictions)
    else:
        st.error(f"No data found for `{active_symbol}`. For Indian stocks, add `.NS` (e.g., `RELIANCE.NS`, `INFY.NS`).")

# --- Schemas ---
class OHLCVBar(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class PredictionRequest(BaseModel):
    symbol: str
    timeframe: str
    bars: List[OHLCVBar]

# --- Helper Engine ---
def fetch_options_features(symbol: str, spot_price: float) -> Dict[str, float]:
    """Fetches Options Chain data and calculates PCR and Max Pain."""
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            return {"pcr": 1.0, "max_pain_dist": 0.0, "straddle_cost_pct": 0.02, "avg_iv": 0.20}

        chain = ticker.option_chain(expirations[0])
        calls, puts = chain.calls, chain.puts

        total_call_oi = calls['openInterest'].fillna(0).sum()
        total_put_oi = puts['openInterest'].fillna(0).sum()
        pcr = (total_put_oi / total_call_oi) if total_call_oi > 0 else 1.0

        all_strikes = sorted(list(set(calls['strike']).union(set(puts['strike']))))
        pain_scores = {}
        for s in all_strikes:
            call_loss = (calls[calls['strike'] < s]['strike'] - s).abs() * calls[calls['strike'] < s]['openInterest'].fillna(0)
            put_loss = (puts[puts['strike'] > s]['strike'] - s).abs() * puts[puts['strike'] > s]['openInterest'].fillna(0)
            pain_scores[s] = call_loss.sum() + put_loss.sum()
        
        max_pain_strike = min(pain_scores, key=pain_scores.get) if pain_scores else spot_price
        max_pain_dist = (spot_price - max_pain_strike) / spot_price

        return {"pcr": float(pcr), "max_pain_dist": float(max_pain_dist), "straddle_cost_pct": 0.02, "avg_iv": 0.20}
    except Exception:
        return {"pcr": 1.0, "max_pain_dist": 0.0, "straddle_cost_pct": 0.02, "avg_iv": 0.20}

# --- Routes ---
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Backend API is online"}

@app.post("/api/v1/predict")
def predict_stock(req: PredictionRequest):
    if len(req.bars) < 30:
        raise HTTPException(status_code=400, detail="At least 30 historical bars required.")

    df = pd.DataFrame([b.dict() for b in req.bars])
    
    # Calculate Features
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    
    latest_row = df.iloc[-1]
    spot_price = latest_row['close']
    options_info = fetch_options_features(req.symbol, spot_price)

    # Prediction Logic
    pcr = options_info['pcr']
    rsi = float(latest_row['rsi']) if not np.isnan(latest_row['rsi']) else 50.0
    adx = float(latest_row['adx']) if not np.isnan(latest_row['adx']) else 20.0

    score = 0.50
    if pcr > 1.2: score += 0.15
    elif pcr < 0.8: score -= 0.15
    if rsi < 35: score += 0.15
    elif rsi > 65: score -= 0.15

    prob = float(np.clip(score, 0.05, 0.95))
    signal = "BULLISH" if prob > 0.58 else ("BEARISH" if prob < 0.42 else "NEUTRAL")

    return {
        "symbol": req.symbol,
        "signal": signal,
        "probability": prob,
        "detected_patterns": ["PCR Sentiment Active"] if abs(pcr - 1.0) > 0.2 else [],
        "market_context": {
            "rsi": rsi,
            "adx": adx,
            "atr": float(latest_row['atr']) if not np.isnan(latest_row['atr']) else 1.0,
            "pcr": pcr,
            "max_pain_dist": options_info['max_pain_dist']
        }
    }
