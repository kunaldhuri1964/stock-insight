import sqlite3
import requests
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Graceful optional imports
try:
    from pipeline import extract_58_candlestick_patterns
    HAS_PIPELINE = True
except Exception:
    HAS_PIPELINE = False

try:
    from nsepython import nse_get_index_quote, nse_eq
    HAS_NSEPYTHON = True
except Exception as e:
    HAS_NSEPYTHON = False
    print(f"nsepython import warning: {e}")

INDEX_MAP = {
    "NIFTY 50": {"nse_symbol": "NIFTY 50", "yf_symbol": "^NSEI"},
    "NIFTY": {"nse_symbol": "NIFTY 50", "yf_symbol": "^NSEI"},
    "BANKNIFTY": {"nse_symbol": "NIFTY BANK", "yf_symbol": "^NSEBANK"},
    "FINNIFTY": {"nse_symbol": "NIFTY FINANCIAL SERVICES", "yf_symbol": "NIFTY_FIN_SERVICE.NS"},
    "MIDCAP NIFTY": {"nse_symbol": "NIFTY MIDCAP 100", "yf_symbol": "^NSEMDCP50"},
    "SENSEX": {"nse_symbol": "SENSEX", "yf_symbol": "^BSESN"},
}


class HorizonPredictor:
    def __init__(self):
        # Attempt loading trained quantile models; fallback gracefully if models directory is missing
        self.models = {}
        try:
            self.models = {
                "1H": (joblib.load('models/1H_lower.pkl'), joblib.load('models/1H_upper.pkl')),
                "2H": (joblib.load('models/2H_lower.pkl'), joblib.load('models/2H_upper.pkl')),
                "3H": (joblib.load('models/3H_lower.pkl'), joblib.load('models/3H_upper.pkl'))
            }
        except Exception:
            self.models = {}

    def predict_hourly_horizons(self, spot_price: float, atr_val: float = None, signal: float = 0, options_data: dict = None) -> dict:
        """
        Calculates dynamic 1H, 2H, and 3H ranges using spot price, ATR volatility,
        directional signal score, and options chain sentiment.
        """
        if spot_price is None or spot_price <= 0:
            return {}

        # Default ATR fallback if zero or missing (0.5% default)
        if atr_val is None or atr_val <= 0:
            atr_val = spot_price * 0.005

        pcr_bias = 0.0
        options_sentiment = "Neutral"
        if isinstance(options_data, dict) and options_data:
            pcr = options_data.get('pcr', 1.0)
            if pcr is not None:
                pcr_bias = (pcr - 1.0) * 0.001
                if pcr > 1.2:
                    options_sentiment = "Bullish (PCR > 1.2)"
                elif pcr < 0.8:
                    options_sentiment = "Bearish (PCR < 0.8)"

        time_scales = {"1H": 1.0, "2H": 1.414, "3H": 1.732}
        directional_shift = (float(signal) / 100.0) * (atr_val * 0.2) + (spot_price * pcr_bias)

        horizons = {}
        for h_name, scale in time_scales.items():
            expected_center = spot_price + directional_shift
            half_spread = atr_val * scale

            lower_bound = expected_center - half_spread
            upper_bound = expected_center + half_spread

            horizons[h_name] = {
                "center": round(expected_center, 2),
                "lower": round(lower_bound, 2),
                "upper": round(upper_bound, 2),
                "spread": round(half_spread, 2),
                "range_text": f"₹{lower_bound:,.2f} — ₹{upper_bound:,.2f}",
                "options_sentiment": options_sentiment,
                "active_patterns": []
            }

        return horizons

    def predict_live_ranges(self, live_df: pd.DataFrame) -> dict:
        """
        Inference routine accepting live price DataFrames.
        Integrates pattern extraction pipeline if present or falls back to ATR calculation.
        """
        if live_df.empty:
            return {}

        df_cols = {col.lower(): col for col in live_df.columns}
        close_col = df_cols.get('close', 'close')
        current_price = float(live_df[close_col].iloc[-1])

        # Feature-based Model Pipeline Execution
        if HAS_PIPELINE and self.models:
            try:
                df_processed = extract_58_candlestick_patterns(live_df)
                feature_cols = [c for c in df_processed.columns if c.startswith('pattern_')] + ['net_pattern_score']
                latest_features = df_processed[feature_cols].iloc[[-1]]
                active_cols = [col.replace('pattern_', '') for col in feature_cols if latest_features[col].values[0] != 0]

                results = {}
                for horizon, (model_lower, model_upper) in self.models.items():
                    pred_low_ret = model_lower.predict(latest_features)[0]
                    pred_high_ret = model_upper.predict(latest_features)[0]

                    low_price = current_price * (1 + pred_low_ret)
                    high_price = current_price * (1 + pred_high_ret)
                    center = (low_price + high_price) / 2
                    spread = (high_price - low_price) / 2

                    results[horizon] = {
                        "center": round(center, 2),
                        "lower": round(low_price, 2),
                        "upper": round(high_price, 2),
                        "spread": round(spread, 2),
                        "active_patterns": active_cols
                    }
                return results
            except Exception:
                pass

        # Fallback to Mathematical Volatility Scale if models or pipeline are uninitialized
        atr_col = df_cols.get('atr', None)
        atr_val = float(live_df[atr_col].iloc[-1]) if atr_col and atr_col in live_df.columns else current_price * 0.005
        return self.predict_hourly_horizons(spot_price=current_price, atr_val=atr_val)


# --- SQLite Search Helpers ---

def search_stocks_in_db(search_term: str = "") -> list:
    clean_term = search_term.strip()
    if not clean_term:
        query = "SELECT symbol, name FROM stocks ORDER BY symbol ASC LIMIT 500"
        params = ()
    else:
        query = "SELECT symbol, name FROM stocks WHERE symbol LIKE ? OR name LIKE ? ORDER BY symbol ASC"
        params = (f"%{clean_term}%", f"%{clean_term}%")

    try:
        conn = sqlite3.connect("stocks.db")
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        results = [f"{row['symbol']} - {row['name']}" for _, row in df.iterrows()]
        if not results and clean_term:
            results = [f"{clean_term.upper()} - Custom/Direct Symbol Input"]
        return results
    except Exception:
        return [f"{clean_term.upper() if clean_term else 'RELIANCE'} - Direct Entry"]


# --- Live Quote Processing Helpers ---

def fetch_live_nse_quote(symbol: str) -> dict:
    clean_symbol = symbol.replace(".NS", "").replace(".BO", "").strip().upper()

    if HAS_NSEPYTHON and clean_symbol in INDEX_MAP:
        try:
            index_name = INDEX_MAP[clean_symbol]["nse_symbol"]
            quote = nse_get_index_quote(index_name)
            return {
                "symbol": clean_symbol,
                "company_name": index_name,
                "last_price": float(quote.get('lastPrice', 0)),
                "change": float(quote.get('change', 0)),
                "pChange": float(quote.get('pChange', 0)),
                "day_high": float(quote.get('high', 0)),
                "day_low": float(quote.get('low', 0)),
                "open": float(quote.get('open', 0)),
                "prev_close": float(quote.get('previousClose', 0))
            }
        except Exception:
            pass

    if HAS_NSEPYTHON:
        try:
            quote = nse_eq(clean_symbol)
            price_info = quote.get('priceInfo', {})
            return {
                "symbol": clean_symbol,
                "company_name": quote.get('info', {}).get('companyName', clean_symbol),
                "last_price": float(price_info.get('lastPrice', 0)),
                "change": float(price_info.get('change', 0)),
                "pChange": float(price_info.get('pChange', 0)),
                "day_high": float(price_info.get('intraDayHighLow', {}).get('max', 0)),
                "day_low": float(price_info.get('intraDayHighLow', {}).get('min', 0)),
                "open": float(price_info.get('open', 0)),
                "prev_close": float(price_info.get('previousClose', 0))
            }
        except Exception:
            pass

    # Yahoo Finance Fallback
    try:
        ticker = INDEX_MAP[clean_symbol]["yf_symbol"] if clean_symbol in INDEX_MAP else (
            f"{clean_symbol}.NS" if not clean_symbol.startswith("^") else clean_symbol
        )
        y_stock = yf.Ticker(ticker)
        fast_info = y_stock.fast_info

        last_price = float(fast_info['lastPrice'])
        prev_close = float(fast_info['previousClose'])
        change = last_price - prev_close
        p_change = (change / prev_close) * 100 if prev_close != 0 else 0

        return {
            "symbol": clean_symbol,
            "company_name": clean_symbol,
            "last_price": last_price,
            "change": float(change),
            "pChange": float(p_change),
            "day_high": float(fast_info['dayHigh']),
            "day_low": float(fast_info['dayLow']),
            "open": float(fast_info['open']),
            "prev_close": prev_close
        }
    except Exception:
        return None


# --- Option Chain Processing Helpers ---

def fetch_option_chain(symbol: str) -> pd.DataFrame:
    clean_symbol = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
    try:
        indices = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
        if clean_symbol in indices:
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={clean_symbol}"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={clean_symbol}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/option-chain",
        }

        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        response = session.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            raw_records = data.get("records", {}).get("data", [])
            processed_data = []
            for row in raw_records:
                strike = row.get("strikePrice")
                ce = row.get("CE", {})
                pe = row.get("PE", {})
                if ce or pe:
                    processed_data.append({
                        "Call OI": ce.get("openInterest", 0),
                        "Call LTP": ce.get("lastPrice", 0),
                        "Strike Price": strike,
                        "Put LTP": pe.get("lastPrice", 0),
                        "Put OI": pe.get("openInterest", 0),
                    })
            df = pd.DataFrame(processed_data)
            if not df.empty:
                return df
    except Exception as e:
        print(f"NSE Option Chain API Error: {e}")

    # Yahoo Finance Option Chain Fallback
    try:
        ticker_symbol = INDEX_MAP[clean_symbol]["yf_symbol"] if clean_symbol in INDEX_MAP else f"{clean_symbol}.NS"
        ticker = yf.Ticker(ticker_symbol)
        expirations = ticker.options
        if not expirations:
            return pd.DataFrame()

        opt = ticker.option_chain(expirations[0])
        calls = opt.calls[['strike', 'lastPrice', 'openInterest']].copy()
        puts = opt.puts[['strike', 'lastPrice', 'openInterest']].copy()

        merged = pd.merge(calls, puts, on='strike', how='outer', suffixes=(' Call', ' Put')).fillna(0)
        merged.rename(columns={
            'strike': 'Strike Price',
            'openInterest Call': 'Call OI',
            'lastPrice Call': 'Call LTP',
            'lastPrice Put': 'Put LTP',
            'openInterest Put': 'Put OI'
        }, inplace=True)

        return merged
    except Exception as e:
        print(f"yfinance Option Chain Fallback Error: {e}")
        return pd.DataFrame()


# --- Training & Data Pipeline Helpers ---

def load_and_prepare_data(symbol: str, interval: str = "1h", period: str = "730d") -> pd.DataFrame:
    clean_symbol = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
    yf_ticker = INDEX_MAP[clean_symbol]["yf_symbol"] if clean_symbol in INDEX_MAP else f"{clean_symbol}.NS"

    df = yf.download(yf_ticker, period=period, interval=interval, progress=False)
    if df.empty and not clean_symbol.startswith("^"):
        df = yf.download(clean_symbol, period=period, interval=interval, progress=False)

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardize column headers to lowercase
    df.columns = [c.lower() for c in df.columns]

    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        return None

    if len(df) < 30:
        return None

    df['body'] = abs(df['close'] - df['open'])
    df['total_range'] = df['high'] - df['low']

    max_oc = df[['open', 'close']].max(axis=1)
    min_oc = df[['open', 'close']].min(axis=1)

    df['upper_wick'] = df['high'] - max_oc
    df['lower_wick'] = min_oc - df['low']

    df['body_ratio'] = df['body'] / (df['total_range'] + 1e-6)
    df['upper_wick_ratio'] = df['upper_wick'] / (df['total_range'] + 1e-6)
    df['lower_wick_ratio'] = df['lower_wick'] / (df['total_range'] + 1e-6)

    df['pattern_hammer'] = ((df['lower_wick_ratio'] > 0.5) & (df['body_ratio'] < 0.3)).astype(int)
    df['pattern_doji'] = (df['body_ratio'] < 0.1).astype(int)

    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['dist_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['atr'] = df['total_range'].rolling(window=14).mean()

    df.dropna(inplace=True)
    return df


def train_and_predict(data: pd.DataFrame, prediction_horizon: int = 1):
    feature_cols = [
        'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
        'pattern_hammer', 'pattern_doji', 'dist_sma20'
    ]

    data['target'] = (data['close'].shift(-prediction_horizon) > data['close']).astype(int)

    latest_row = data.iloc[[-1]][feature_cols]
    model_df = data.iloc[:-prediction_horizon].dropna()

    X = model_df[feature_cols]
    y = model_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    latest_pred = model.predict(latest_row)[0]
    latest_prob = model.predict_proba(latest_row)[0][1]

    last_time = data.index[-1]
    last_price = float(data['close'].iloc[-1])
    avg_volatility = float(data['atr'].iloc[-1]) if 'atr' in data.columns else (last_price * 0.002)

    direction = 1 if latest_pred == 1 else -1
    future_times = [last_time]
    future_prices = [last_price]

    for step in range(1, prediction_horizon + 1):
        next_time = last_time + timedelta(hours=step)
        expected_move = (avg_volatility * 0.5 * step) * direction * (0.5 + abs(latest_prob - 0.5))
        next_price = last_price + expected_move
        future_times.append(next_time)
        future_prices.append(next_price)

    forecast_data = {
        "times": future_times,
        "prices": future_prices,
        "direction": latest_pred
    }

    return latest_pred, latest_prob, model, forecast_data
from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_model_accuracy(df_feat):
    X_test, y_test = get_test_data(df_feat)   # however you already split test data
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }
    return metrics
