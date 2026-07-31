import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Imports the shared feature engineer directly from prediction_service
from prediction_service import FeatureEngineer

def create_target_labels(df: pd.DataFrame, horizon: int = 5, multiplier: float = 1.5) -> pd.Series:
    """Target = 1 if high price increases by (1.5 * ATR) within next 5 bars."""
    future_max_high = df['high'].shift(-horizon).rolling(window=horizon).max()
    target_threshold = df['close'] + (multiplier * df['atr'])
    return (future_max_high >= target_threshold).astype(int)

def train_and_save_model(csv_path: str = "historical_ohlc.csv", output_model_path: str = "lgbm_candlestick.txt"):
    print(f"1. Loading CSV dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    print("2. Extracting features using shared FeatureEngineer pipeline...")
    df = FeatureEngineer.extract_features(df)

    print("3. Generating target labels...")
    df['target'] = create_target_labels(df, horizon=5, multiplier=1.5)
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        'norm_body', 'norm_upper_shadow', 'norm_lower_shadow', 'candle_direction',
        'dist_ema_20', 'dist_ema_50', 'rsi', 'adx',
        'pattern_engulfing', 'pattern_morningstar', 'pattern_eveningstar', 
        'pattern_hammer', 'pattern_shootingstar', 'pattern_doji', 'pattern_kicker'
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

    print("4. Training LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30)]
    )

    # Save trained weights file
    model.save_model(output_model_path)
    print(f"5. Saved trained model weights to '{output_model_path}'!")

if __name__ == "__main__":
    train_and_save_model()
