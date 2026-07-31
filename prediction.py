import numpy as np
import pandas as pd
import lightgbm as lgb
import talib
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

# --- Shared Feature Extractor ---
class FeatureEngineer:
    @staticmethod
    def extract_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extracts normalized geometric features, candlestick patterns, and indicators."""
        if len(df) < 30:
            raise ValueError("Insufficient data bars. At least 30 periods required.")

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Indicators
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # Scale-Invariant Geometric Features (Normalized by ATR)
        real_body = (df['close'] - df['open']).abs()
        df['norm_body'] = real_body / df['atr']
        df['norm_upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['atr']
        df['norm_lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['atr']
        df['candle_direction'] = np.sign(df['close'] - df['open'])

        # Context Features
        df['dist_ema_20'] = (df['close'] - df['ema_20']) / df['atr']
        df['dist_ema_50'] = (df['close'] - df['ema_50']) / df['atr']

        # TA-Lib Patterns
        df['pattern_engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        df['pattern_morningstar'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['pattern_eveningstar'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['pattern_hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['pattern_shootingstar'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['pattern_doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        df['pattern_kicker'] = talib.CDLKICKER(df['open'], df['high'], df['low'], df['close'])

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
            'pattern_hammer', 'pattern_shootingstar', 'pattern_doji', 'pattern_kicker'
        ]
        
        X = latest_row[feature_cols].values.reshape(1, -1)

        if self.model:
            prob_bullish = float(self.model.predict(X)[0])
        else:
            # Fallback heuristic if model is not trained yet
            base = 0.5
            if latest_row['rsi'] < 30: base += 0.15
            if latest_row['rsi'] > 70: base -= 0.15
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
                "dist_ema_20": float(latest_row['dist_ema_20'])
            }
        }
