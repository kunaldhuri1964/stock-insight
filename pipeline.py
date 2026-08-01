import json
import pandas as pd
import yfinance as yf
import talib

def load_pdf_rules(json_path="pdf_58_patterns.json"):
    with open(json_path, "r") as f:
        return json.load(f)

def run_pattern_scan(ticker="BTC-USD", interval="15m", period="1d"):
    print(f"Fetching live data for {ticker}...")
    df = yf.download(tickers=ticker, period=period, interval=interval)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    open_p, high_p = df['Open'].values, df['High'].values
    low_p, close_p = df['Low'].values, df['Close'].values

    # Scan TA-Lib patterns
    pattern_functions = talib.get_function_groups()['Pattern Recognition']
    detected = []

    for func_name in pattern_functions:
        pattern_func = getattr(talib, func_name)
        result = pattern_func(open_p, high_p, low_p, close_p)
        latest_signal = result[-1]
        
        if latest_signal != 0:
            detected.append((func_name, "BULLISH" if latest_signal > 0 else "BEARISH"))

    pdf_rules = load_pdf_rules()
    print(f"\nScan complete. Detected {len(detected)} pattern(s).")
    for func, sentiment in detected:
        print(f"-> Pattern: {func} ({sentiment})")

if __name__ == "__main__":
    run_pattern_scan("BTC-USD")
