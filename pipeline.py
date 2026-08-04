import time
import json
import pandas as pd
import yfinance as yf
import talib
import matplotlib.pyplot as plt
import mplfinance as mpf

def load_pdf_rules(json_path="pdf_58_patterns.json"):
    """Loads pattern definitions extracted from your 58 Candlestick PDF."""
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def fetch_live_data_and_chart(ticker="BTC-USD", interval="1m", period="1d"):
    """
    Fetches real-time price tick and intraday candlestick data,
    saves an updated chart image, and returns the dataframe.
    """
    ticker_obj = yf.Ticker(ticker)
    
    # 1. Get exact current live price
    try:
        live_price = ticker_obj.fast_info['lastPrice']
    except Exception:
        live_price = None

    # 2. Download historical OHLC data for pattern detection
    df = ticker_obj.history(period=period, interval=interval)

    if df.empty:
        print("Error: Could not retrieve live data.")
        return None, None

    # Clean multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 3. Update & save live candlestick chart
    chart_filename = "live_chart.png"
    mpf.plot(
        df.tail(40),  # Show last 40 candles on chart
        type='candle',
        style='charles',
        title=f"Live Web Chart: {ticker} (Last Price: ${live_price:.2f if live_price else df['Close'].iloc[-1]:.2f})",
        ylabel='Price ($)',
        volume=True,
        savefig=chart_filename
    )
    
    return df, live_price
    def extract_58_candlestick_patterns(df):
    pattern_names = talib.get_function_groups()['Pattern Recognition']
    for pattern in pattern_names:
        pattern_func = getattr(talib, pattern)
        df[f'pattern_{pattern}'] = pattern_func(
            df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
        )
    pattern_cols = [c for c in df.columns if c.startswith('pattern_')]
    df['net_pattern_score'] = df[pattern_cols].sum(axis=1)
    return df

def train_and_save_horizon_models(df, horizon_name, horizon_steps):
    df = extract_58_candlestick_patterns(df)
    feature_cols = [c for c in df.columns if c.startswith('pattern_')] + ['net_pattern_score']
    
    df['target_return'] = (df['Close'].shift(-horizon_steps) - df['Close']) / df['Close']
    clean_df = df.dropna()
    
    X = clean_df[feature_cols]
    y = clean_df['target_return']
    
    model_lower = lgb.LGBMRegressor(objective='quantile', alpha=0.10, n_estimators=100)
    model_upper = lgb.LGBMRegressor(objective='quantile', alpha=0.90, n_estimators=100)
    
    model_lower.fit(X, y)
    model_upper.fit(X, y)
    
    # Save models to disk for Backend to use
    joblib.dump(model_lower, f'models/{horizon_name}_lower.pkl')
    joblib.dump(model_upper, f'models/{horizon_name}_upper.pkl')

def run_live_pipeline(ticker="BTC-USD", refresh_seconds=15):
    """
    Continuous pipeline that updates live prices, updates chart image,
    and runs 58 pattern detection math on fresh market ticks.
    """
    pdf_rules = load_pdf_rules("pdf_58_patterns.json")
    pattern_functions = talib.get_function_groups()['Pattern Recognition']

    print(f"Starting Live Non-AI Engine for {ticker} (Refreshing every {refresh_seconds}s)...")
    
    while True:
        df, live_price = fetch_live_data_and_chart(ticker=ticker)
        
        if df is None:
            time.sleep(refresh_seconds)
            continue

        open_p = df['Open'].values
        high_p = df['High'].values
        low_p = df['Low'].values
        close_p = df['Close'].values

        detected_patterns = []

        # Run TA-Lib mathematical pattern scans
        for func_name in pattern_functions:
            pattern_func = getattr(talib, func_name)
            result = pattern_func(open_p, high_p, low_p, close_p)
            latest_signal = result[-1]
            
            if latest_signal != 0:
                sentiment = "BULLISH" if latest_signal > 0 else "BEARISH"
                detected_patterns.append({
                    "function": func_name,
                    "sentiment": sentiment
                })

        timestamp = df.index[-1]
        current_display_price = live_price if live_price else close_p[-1]

        # Terminal Display
        print("\n" + "="*50)
        print(f" LIVE TICKER REPORT [{timestamp.strftime('%H:%M:%S')}]")
        print(f" Ticker: {ticker} | REAL-TIME PRICE: ${current_display_price:.2f}")
        print(f" Live Chart Updated: Saved to 'live_chart.png'")
        print("="*50)

        if not detected_patterns:
            print("Patterns Detected: None on current candle.")
            print("PREDICTION: NEUTRAL")
        else:
            print(f"Detected {len(detected_patterns)} Pattern(s):")
            for match in detected_patterns:
                func = match["function"]
                sentiment = match["sentiment"]
                pdf_info = next((v for k, v in pdf_rules.items() if v.get("id") == func), None)
                
                print(f" ► {func} ({sentiment})")
                if pdf_info:
                    print(f"   PDF Rule: {pdf_info.get('rules', 'N/A')}")

        print(f"\nWaiting {refresh_seconds} seconds for next tick update...\n")
        time.sleep(refresh_seconds)

if __name__ == "__main__":
    # Runs continuous scanner and chart updater
    run_live_pipeline("BTC-USD", refresh_seconds=15)
