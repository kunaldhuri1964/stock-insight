import sqlite3
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
# backend.py
import joblib
import numpy as np
import pandas as pd
from pipeline import extract_58_candlestick_patterns

class HorizonPredictor:
    def __init__(self):
        # Load trained quantile models for 1H, 2H, 3H
        self.models = {
            "1H": (joblib.load('models/1H_lower.pkl'), joblib.load('models/1H_upper.pkl')),
            "2H": (joblib.load('models/2H_lower.pkl'), joblib.load('models/2H_upper.pkl')),
            "3H": (joblib.load('models/3H_lower.pkl'), joblib.load('models/3H_upper.pkl'))
        }

    def predict_live_ranges(self, live_df):
        # Apply pattern extraction on live data
        df_processed = extract_58_candlestick_patterns(live_df)
        feature_cols = [c for c in df_processed.columns if c.startswith('pattern_')] + ['net_pattern_score']
        
        latest_features = df_processed[feature_cols].iloc[[-1]]
        current_price = float(live_df['Close'].iloc[-1])
        
        # Identify active patterns
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
try:
    from nsepython import nse_get_index_quote, nse_eq, nse_optionchain_scrip
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

def search_stocks_in_db(search_term=""):
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

def fetch_live_nse_quote(symbol):
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
        
    try:
        ticker = INDEX_MAP[clean_symbol]["yf_symbol"] if clean_symbol in INDEX_MAP else (f"{clean_symbol}.NS" if not clean_symbol.startswith("^") else clean_symbol)
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

# --- FETCH OPTION CHAIN DATA ---
import requests
import pandas as pd

def fetch_option_chain(symbol):
    try:
        symbol = symbol.upper().strip()
        
        # 1. Differentiate indices vs stock equities
        indices = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
        if symbol in indices:
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"

        # 2. Add realistic browser headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.nseindia.com/option-chain",
        }

        session = requests.Session()
        
        # 3. First hit home page to establish session cookies (crucial for NSE)
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        # 4. Fetch the option chain JSON data
        response = session.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None
            
        data = response.json()
        
        # 5. Process raw JSON into a clean DataFrame
        raw_records = data.get("records", {}).get("data", [])
        
        processed_data = []
        for row in raw_records:
            strike = row.get("strikePrice")
            ce = row.get("CE", {})
            pe = row.get("PE", {})
            
            if ce or pe:
                processed_data.append({
                    "CE_OI": ce.get("openInterest", 0),
                    "CE_LTP": ce.get("lastPrice", 0),
                    "Strike": strike,
                    "PE_LTP": pe.get("lastPrice", 0),
                    "PE_OI": pe.get("openInterest", 0),
                })
                
        df = pd.DataFrame(processed_data)
        return df if not df.empty else None

    except Exception as e:
        print(f"Error fetching option chain: {e}")
        return None

    # Fallback: Yahoo Finance Option Chain (works reliably on Streamlit Cloud)
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
        print(f"yfinance Option Chain Error: {e}")
        return pd.DataFrame()

def load_and_prepare_data(symbol, interval="1h", period="730d"):
    clean_symbol = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
    yf_ticker = INDEX_MAP[clean_symbol]["yf_symbol"] if clean_symbol in INDEX_MAP else f"{clean_symbol}.NS"

    df = yf.download(yf_ticker, period=period, interval=interval, progress=False)
    if df.empty and not clean_symbol.startswith("^"):
        df = yf.download(clean_symbol, period=period, interval=interval, progress=False)
        
    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns and isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    if len(df) < 30:
        return None

    df['Body'] = abs(df['Close'] - df['Open'])
    df['Total_Range'] = df['High'] - df['Low']
    
    max_oc = df[['Open', 'Close']].max(axis=1)
    min_oc = df[['Open', 'Close']].min(axis=1)

    df['Upper_Wick'] = df['High'] - max_oc
    df['Lower_Wick'] = min_oc - df['Low']

    df['Body_Ratio'] = df['Body'] / (df['Total_Range'] + 1e-6)
    df['Upper_Wick_Ratio'] = df['Upper_Wick'] / (df['Total_Range'] + 1e-6)
    df['Lower_Wick_Ratio'] = df['Lower_Wick'] / (df['Total_Range'] + 1e-6)

    df['Pattern_Hammer'] = ((df['Lower_Wick_Ratio'] > 0.5) & (df['Body_Ratio'] < 0.3)).astype(int)
    df['Pattern_Doji'] = (df['Body_Ratio'] < 0.1).astype(int)

    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Dist_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    df['ATR'] = df['Total_Range'].rolling(window=14).mean()

    df.dropna(inplace=True)
    return df

def train_and_predict(data, prediction_horizon=1):
    feature_cols = [
        'Body_Ratio', 'Upper_Wick_Ratio', 'Lower_Wick_Ratio',
        'Pattern_Hammer', 'Pattern_Doji', 'Dist_SMA20'
    ]

    data['Target'] = (data['Close'].shift(-prediction_horizon) > data['Close']).astype(int)

    latest_row = data.iloc[[-1]][feature_cols]
    model_df = data.iloc[:-prediction_horizon].dropna()

    X = model_df[feature_cols]
    y = model_df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    latest_pred = model.predict(latest_row)[0]
    latest_prob = model.predict_proba(latest_row)[0][1]

    last_time = data.index[-1]
    last_price = float(data['Close'].iloc[-1])
    avg_volatility = float(data['ATR'].iloc[-1]) if 'ATR' in data.columns else (last_price * 0.002)

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
