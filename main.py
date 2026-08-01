import numpy as np
import pandas as pd
import lightgbm as lgb
import talib
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

app = FastAPI(title="Stock Prediction API")

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
