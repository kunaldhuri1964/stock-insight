# init_db.py
import sqlite3
import pandas as pd
import requests

def setup_stock_database():
    print("Fetching official NSE listed stocks master list...")
    
    # Official NSE Equity Master CSV URL
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    
    # Standard headers to prevent blocking
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }


try:
    # 1. Read CSV
    df = pd.read_csv("nse_stocks.csv")
    print(f"Original CSV Rows: {len(df)}")

    # 2. Clean column names (removes extra spaces)
    df.columns = df.columns.str.strip()

    # Automatically identify symbol and company name columns
    symbol_col = [c for c in df.columns if 'SYMBOL' in c.upper()][0]
    company_col = [
        c for c in df.columns if 'NAME' in c.upper() or 'COMPANY' in c.upper()
    ][0]

    # Select and rename
    clean_df = df[[symbol_col, company_col]].copy()
    clean_df.columns = ['symbol', 'company_name']

    # 3. Remove blanks & duplicates
    clean_df = clean_df.dropna()
    clean_df = clean_df.drop_duplicates(subset=['symbol'])

    # 4. Save to Database
    conn = sqlite3.connect("stocks.db")
    clean_df.to_sql("stocks_table", conn, if_exists="replace", index=False)

    # 5. Check actual DB count
    count = conn.execute("SELECT COUNT(*) FROM stocks_table").fetchone()[0]
    conn.close()

    print(
        f"SUCCESS! Stored {count} unique companies into 'stocks_table' in"
        " stocks.db"
    )

except Exception as e:
    print(f"Error processing CSV: {e}")

    try:
        response = requests.get(url, headers=headers)
        with open("EQUITY_L.csv", "wb") as f:
            f.write(response.content)
            
        df = pd.read_csv("EQUITY_L.csv")
        df = df[['SYMBOL', 'NAME OF COMPANY']].dropna()
        df['SYMBOL'] = df['SYMBOL'].str.strip()
        df['NAME OF COMPANY'] = df['NAME OF COMPANY'].str.strip()
        
    except Exception as e:
        print(f"Could not download fresh list from NSE: {e}. Using fallback defaults.")
        # Fallback dataset if offline
        df = pd.DataFrame([
            {"SYMBOL": "RELIANCE", "NAME OF COMPANY": "Reliance Industries Limited"},
            {"SYMBOL": "TCS", "NAME OF COMPANY": "Tata Consultancy Services Limited"},
            {"SYMBOL": "INFY", "NAME OF COMPANY": "Infosys Limited"},
            {"SYMBOL": "HDFCBANK", "NAME OF COMPANY": "HDFC Bank Limited"},
            {"SYMBOL": "ICICIBANK", "NAME OF COMPANY": "ICICI Bank Limited"},
            {"SYMBOL": "SBIN", "NAME OF COMPANY": "State Bank of India"},
            {"SYMBOL": "BHARTIARTL", "NAME OF COMPANY": "Bharti Airtel Limited"},
            {"SYMBOL": "ITC", "NAME OF COMPANY": "ITC Limited"},
            {"SYMBOL": "LTIM", "NAME OF COMPANY": "LTIMindtree Limited"},
            {"SYMBOL": "TATAMOTORS", "NAME OF COMPANY": "Tata Motors Limited"}
        ])

    # Add Major Indices into Database
    indices = [
        {"SYMBOL": "NIFTY 50", "NAME OF COMPANY": "NIFTY 50 INDEX"},
        {"SYMBOL": "BANKNIFTY", "NAME OF COMPANY": "NIFTY BANK INDEX"},
        {"SYMBOL": "SENSEX", "NAME OF COMPANY": "BSE SENSEX INDEX"},
        {"SYMBOL": "FINNIFTY", "NAME OF COMPANY": "NIFTY FINANCIAL SERVICES INDEX"},
        {"SYMBOL": "MIDCAP NIFTY", "NAME OF COMPANY": "NIFTY MIDCAP 100 INDEX"}
    ]
    
    df_indices = pd.DataFrame(indices)
    df_all = pd.concat([df_indices, df], ignore_index=True)

    # Save into SQLite Database
    conn = sqlite3.connect("stocks.db")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS stocks")
    cursor.execute("""
        CREATE TABLE stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL
        )
    """)
    
    for _, row in df_all.iterrows():
        cursor.execute("INSERT OR IGNORE INTO stocks (symbol, name) VALUES (?, ?)", (row['SYMBOL'], row['NAME OF COMPANY']))
        
    conn.commit()
    conn.close()
    print(f"✅ Successfully database updated with {len(df_all)} stocks & indices in 'stocks.db'!")

if __name__ == "__main__":
    setup_stock_database()