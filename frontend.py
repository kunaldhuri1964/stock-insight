import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

def create_target_labels(df: pd.DataFrame, horizon: int = 5, multiplier: float = 1.5) -> pd.Series:
    """Target = 1 if stock rises by (multiplier * ATR) within 5 periods."""
    future_max_high = df['high'].shift(-horizon).rolling(window=horizon).max()
    target_threshold = df['close'] + (multiplier * df['atr'])
    return (future_max_high >= target_threshold).astype(int)

def train_and_save_options_model(csv_path: str = "historical_ohlc.csv", output_model_path: str = "lgbm_candlestick.txt"):
    print(f"1. Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    print("2. Computing Technical & Options Chain Features...")
    # Pass ticker symbol to extract options chain data
    symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else "AAPL"
    df = FeatureEngineer.extract_features(df, symbol=symbol)

    print("3. Generating Labels...")
    df['target'] = create_target_labels(df, horizon=5, multiplier=1.5)
    df = df.dropna().reset_index(drop=True)

    # Combined Feature Columns (Price Action + TA-Lib + Options Metrics)
    feature_cols = [
        'norm_body', 'norm_upper_shadow', 'norm_lower_shadow', 'candle_direction',
        'dist_ema_20', 'dist_ema_50', 'rsi', 'adx',
        'pattern_engulfing', 'pattern_morningstar', 'pattern_eveningstar', 
        'pattern_hammer', 'pattern_shootingstar',
        'pcr', 'max_pain_dist', 'straddle_cost_pct', 'avg_iv'
    ]

    X = df[feature_cols]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 6,
        'verbose': -1
    }

    print("4. Training LightGBM Model with Options Features...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30)]
    )

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    print(f"\nModel ROC-AUC Score: {roc_auc_score(y_test, y_pred):.4f}")

    print(f"5. Saving updated model to '{output_model_path}'...")
    model.save_model(output_model_path)
    print("Training finished!")

if __name__ == "__main__":
    train_and_save_options_model()
