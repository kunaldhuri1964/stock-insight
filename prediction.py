import numpy as np
import pandas as pd
import lightgbm as lgb
import talib
import yfinance as yf
from pydantic import BaseModel
from typing import List, Dict, Optional

# --- API Schemas ---
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

class PredictionResponse(BaseModel):
    symbol: str
    signal: str
    probability: float
    detected_patterns: List[str]
    market_context: Dict[str, float]


# --- Feature Engineering with Options Data ---
class FeatureEngineer:

    @staticmethod
    def fetch_options_features(symbol: str, spot_price: float) -> Dict[str, float]:
        """Fetches live Options Chain via yfinance and computes PCR, Max Pain, and ATM metrics."""
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
                # Default fallback values if ticker has no options chain
                return {"pcr": 1.0, "max_pain_dist": 0.0, "straddle_cost_pct": 0.02, "avg_iv": 0.20}

            # Nearest Expiration Options Chain
            nearest_expiry = expirations[0]
            chain = ticker.option_chain(nearest_expiry)
            calls, puts = chain.calls, chain.puts

            # 1. Put-Call Ratio (PCR)
            total_call_oi = calls['openInterest'].fillna(0).sum()
            total_put_oi = puts['openInterest'].fillna(0).sum()
            pcr = (total_put_oi / total_call_oi) if total_call_oi > 0 else 1.0

            # 2. Max Pain Calculation
            all_strikes = sorted(list(set(calls['strike']).union(set(puts['strike']))))
            pain_scores = {}
            for s in all_strikes:
                call_loss = (calls[calls['strike'] < s]['strike'] - s).abs() * calls[calls['strike'] < s]['openInterest'].fillna(0)
                put_loss = (puts[puts['strike'] > s]['strike'] - s).abs() * puts[puts['strike'] > s]['openInterest'].fillna(0)
                pain_scores[s] = call_loss.sum() + put_loss.sum()
            
            max_pain_strike = min(pain_scores, key=pain_scores.get) if pain_scores else spot_price
            max_pain_dist = (spot_price - max_pain_strike) / spot_price

            # 3. ATM Straddle Price Percentage
            atm_call = calls.iloc[(calls['strike'] - spot_price).abs().argsort()[:1]]
            atm_put = puts.iloc[(puts['strike'] - spot_price).abs().argsort()[:1]]
            
            call_price = atm_call['lastPrice'].values[0] if not atm_call.empty else 0.0
            put_price = atm_put['lastPrice'].values[0] if not atm_put.empty else 0.0
            straddle_cost_pct = (call_price + put_price) / spot_price

            # 4. Average Implied Volatility (IV)
            call_iv = atm_call['impliedVolatility'].values[0] if not atm_call.empty else 0.20
            put_iv = atm_put['impliedVolatility'].values[0] if not atm_put.empty else 0.20
            avg_iv = (call_iv + put_iv) / 2.0

            return {
                "pcr": float(pcr),
                "max_pain_dist": float(max_pain_dist),
                "straddle_cost_pct": float(straddle_cost_pct),
                "avg_iv": float(avg_iv)
            }
        except Exception:
            # Safe Fallbacks
            return {"pcr": 1.0, "max_pain_dist": 0.0, "straddle_cost_pct": 0.02, "avg_iv": 0.20}

    @staticmethod
    def extract_features(df: pd.DataFrame, symbol: str = "AAPL") -> pd.DataFrame:
        """Extracts Technical, Candlestick, and Option Chain features."""
        if len(df) < 30:
            raise ValueError("Insufficient data bars. At least 30 periods required.")

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Standard Technical Indicators
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # Scale-Invariant Geometry Features
        real_body = (df['close'] - df['open']).abs()
        df['norm_body'] = real_body / df['atr']
        df['norm_upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['atr']
        df['norm_lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['atr']
        df['candle_direction'] = np.sign(df['close'] - df['open'])

        df['dist_ema_20'] = (df['close'] - df['ema_20']) / df['atr']
        df['dist_ema_50'] = (df['close'] - df['ema_50']) / df['atr']

        # TA-Lib Patterns
        df['pattern_engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        df['pattern_morningstar'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['pattern_eveningstar'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['pattern_hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['pattern_shootingstar'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])

        # Options Chain Integration
        spot = df['close'].iloc[-1]
        options_data = FeatureEngineer.fetch_options_features(symbol, spot)
        
        df['pcr'] = options_data['pcr']
        df['max_pain_dist'] = options_data['max_pain_dist']
        df['straddle_cost_pct'] = options_data['straddle_cost_pct']
        df['avg_iv'] = options_data['avg_iv']

        return df


# --- Model Inference Service ---
class ModelInferenceService:
    def __init__(self, model_path: Optional[str] = None):
        if model_path:
            self.model = lgb.Booster(model_file=model_path)
        else:
            self.model = None

    def predict(self, features_df: pd.DataFrame) -> Dict:
        latest_row = features_df.iloc[-1]
        
        # Detect active candlestick patterns
        pattern_cols = [c for c in features_df.columns if c.startswith('pattern_')]
        active_patterns = []
        for col in pattern_cols:
            val = latest_row[col]
            if val != 0:
                direction = "Bullish" if val > 0 else "Bearish"
                clean_name = col.replace('pattern_', '').capitalize()
                active_patterns.append(f"{direction} {clean_name}")

        feature_cols = [
            'norm_body', 'norm_upper_shadow', 'norm_lower_shadow', 'candle_direction',
            'dist_ema_20', 'dist_ema_50', 'rsi', 'adx',
            'pattern_engulfing', 'pattern_morningstar', 'pattern_eveningstar', 
            'pattern_hammer', 'pattern_shootingstar',
            'pcr', 'max_pain_dist', 'straddle_cost_pct', 'avg_iv'  # New Options Features
        ]
        
        X = latest_row[feature_cols].values.reshape(1, -1)

        if self.model:
            prob_bullish = float(self.model.predict(X)[0])
        else:
            # Fallback combining Technical + PCR heuristic
            base = 0.5
            if latest_row['pcr'] > 1.2: base += 0.12
            elif latest_row['pcr'] < 0.8: base -= 0.12
            if latest_row['rsi'] < 30: base += 0.10
            if latest_row['rsi'] > 70: base -= 0.10
            prob_bullish = float(np.clip(base, 0.05, 0.95))

        if prob_bullish > 0.60:
            signal = "BULLISH"
        elif prob_bullish < 0.40:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        return {
            "signal": signal,
            "probability": prob_bullish,
            "patterns": active_patterns,
            "context": {
                "rsi": float(latest_row['rsi']),
                "adx": float(latest_row['adx']),
                "atr": float(latest_row['atr']),
                "pcr": float(latest_row['pcr']),
                "max_pain_dist": float(latest_row['max_pain_dist']),
                "avg_iv": float(latest_row['avg_iv'])
            }
        }
