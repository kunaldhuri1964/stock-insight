import sqlite3
import pandas as pd
from typing import Dict

DB_NAME = "market_data.db"

def init_db():
    """Initializes local SQLite database tables."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
            signal TEXT NOT NULL,
            probability REAL NOT NULL,
            patterns TEXT,
            rsi REAL,
            adx REAL,
            atr REAL
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(symbol: str, prediction_result: Dict):
    """Saves live predictions into local database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    patterns_str = ", ".join(prediction_result.get("patterns", []))
    context = prediction_result.get("context", {})
    
    cursor.execute("""
        INSERT INTO predictions (symbol, signal, probability, patterns, rsi, adx, atr)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        symbol,
        prediction_result["signal"],
        prediction_result["probability"],
        patterns_str,
        context.get("rsi", 0.0),
        context.get("adx", 0.0),
        context.get("atr", 0.0)
    ))
    conn.commit()
    conn.close()

# Auto-initialize database on import
init_db()
