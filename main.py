import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from prediction_service import (
    ModelInferenceService, 
    FeatureEngineer, 
    PredictionRequest, 
    PredictionResponse
)
from database import log_prediction

app = FastAPI(title="Quant Market Prediction API")

MODEL_PATH = "lgbm_candlestick.txt"

# Load trained model if available
if os.path.exists(MODEL_PATH):
    print(f"Loading LightGBM model weights from {MODEL_PATH}")
    inference_engine = ModelInferenceService(model_path=MODEL_PATH)
else:
    print("WARNING: lgbm_candlestick.txt not found. Using heuristic fallback engine.")
    inference_engine = ModelInferenceService(model_path=None)

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_market_direction(payload: PredictionRequest):
    try:
        # Convert Pydantic request to DataFrame
        data = [bar.dict() for bar in payload.bars]
        df = pd.DataFrame(data)

        # 1. Feature Engineering
        engineered_df = FeatureEngineer.extract_features(df)

        # 2. Run Prediction
        result = inference_engine.predict(engineered_df)

        # 3. Log into local SQLite Database
        log_prediction(payload.symbol, result)

        return PredictionResponse(
            symbol=payload.symbol,
            signal=result["signal"],
            probability=result["probability"],
            detected_patterns=result["patterns"],
            market_context=result["context"]
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
