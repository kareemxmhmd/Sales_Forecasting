import sys
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    BEST_MODEL_PATH,
    QUANTILE_MODELS_PATH,
    PREPROCESSOR_PATH,
    FEATURE_MANIFEST_PATH,
    METADATA_PATH,
)

model = None
quantile_models = None
preprocessor = None
features_manifest = None
metadata = {}
historical_sales = {}

def load_artifacts():
    global model, quantile_models, preprocessor, features_manifest, metadata, historical_sales
    try:
        if BEST_MODEL_PATH.exists():
            model = joblib.load(BEST_MODEL_PATH)
            print(f"Loaded best model from {BEST_MODEL_PATH}")

        if QUANTILE_MODELS_PATH.exists():
            quantile_models = joblib.load(QUANTILE_MODELS_PATH)
            print(f"Loaded quantile models from {QUANTILE_MODELS_PATH}")

        if PREPROCESSOR_PATH.exists():
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            print(f"Loaded preprocessor from {PREPROCESSOR_PATH}")

        if FEATURE_MANIFEST_PATH.exists():
            with open(FEATURE_MANIFEST_PATH, "r") as f:
                features_manifest = json.load(f)
            print(f"Loaded feature manifest from {FEATURE_MANIFEST_PATH}")

        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r") as f:
                metadata = json.load(f)
            print(f"Loaded metadata from {METADATA_PATH}")

        hist_path = BASE_DIR / "artifacts" / "historical_sales.json"
        if hist_path.exists():
            with open(hist_path, "r") as f:
                # Convert string keys back to int
                raw_hist = json.load(f)
                historical_sales.update({int(k): v for k, v in raw_hist.items()})
            print(f"Loaded historical sales cache from {hist_path}")

    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        raise RuntimeError(f"Artifact loading failed: {e}") from e

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield

app = FastAPI(
    title="Sales Forecasting API",
    description="Production API for Rossmann Store Sales daily forecasting with uncertainty intervals",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StoreRecord(BaseModel):
    model_config = ConfigDict(extra="allow")
    Store: int = Field(..., description="Unique Store ID")
    DayOfWeek: Optional[int] = Field(None, description="Day of week (1=Monday ... 7=Sunday)")
    Date: Optional[str] = Field(None, description="Date in YYYY-MM-DD format")
    Open: int = Field(1, description="Store open status (1=Open, 0=Closed)")
    Promo: int = Field(0, description="Store is running a promotion today (1/0)")
    StateHoliday: Optional[str] = Field("0", description="State holiday (0, a, b, c)")
    SchoolHoliday: Optional[int] = Field(0, description="School holiday (1/0)")
    StoreType: Optional[str] = Field("a", description="Store model type (a, b, c, d)")
    Assortment: Optional[str] = Field("a", description="Assortment level (a, b, c)")
    CompetitionDistance: Optional[float] = Field(1270.0, description="Distance to nearest competitor in meters")
    CompetitionOpenSinceMonth: Optional[float] = Field(0.0, description="Month competitor opened")
    CompetitionOpenSinceYear: Optional[float] = Field(0.0, description="Year competitor opened")
    Promo2: Optional[int] = Field(0, description="Continuous promo active (1/0)")
    Promo2SinceWeek: Optional[float] = Field(0.0, description="Week continuous promo started")
    Promo2SinceYear: Optional[float] = Field(0.0, description="Year continuous promo started")
    PromoInterval: Optional[str] = Field("Unknown", description="Months continuous promo runs")
    Year: Optional[int] = Field(None, description="Calendar year")
    Month: Optional[int] = Field(None, description="Calendar month")
    Day: Optional[int] = Field(None, description="Calendar day")
    WeekOfYear: Optional[int] = Field(None, description="ISO calendar week")

class PredictionRequest(BaseModel):
    data: List[StoreRecord] = Field(..., description="List of store records to forecast")

class SinglePrediction(BaseModel):
    store: int
    date: Optional[str] = None
    prediction: float
    prediction_interval_p10: Optional[float] = None
    prediction_interval_p90: Optional[float] = None
    open: int

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    predictions: List[float]
    results: Optional[List[SinglePrediction]] = None
    model_name: Optional[str] = None

@app.get("/health")
async def health_check():
    if model is None or preprocessor is None or features_manifest is None:
        # Attempt fallback load if called outside lifespan context
        try:
            load_artifacts()
        except Exception:
            raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    return {
        "status": "healthy",
        "artifacts_loaded": True,
        "model_name": metadata.get("best_model", "XGBRegressor"),
        "model_type": metadata.get("model_type", "sklearn"),
        "target_transform": metadata.get("target_transform", "log1p"),
    }

@app.get("/metadata")
async def get_metadata():
    if not metadata:
        try:
            load_artifacts()
        except Exception:
            pass
    return metadata

def _prepare_dataframe(raw_data: List[Dict[str, Any]], required_features: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(raw_data)

    if "Date" in df.columns:
        dt = pd.to_datetime(df["Date"], errors="coerce")
        if "Year" not in df.columns or df["Year"].isna().any():
            df["Year"] = dt.dt.year.fillna(2015).astype(int)
        if "Month" not in df.columns or df["Month"].isna().any():
            df["Month"] = dt.dt.month.fillna(1).astype(int)
        if "Day" not in df.columns or df["Day"].isna().any():
            df["Day"] = dt.dt.day.fillna(1).astype(int)
        if "DayOfWeek" not in df.columns or df["DayOfWeek"].isna().any():
            df["DayOfWeek"] = dt.dt.dayofweek.fillna(0).astype(int) + 1
        if "WeekOfYear" not in df.columns or df["WeekOfYear"].isna().any():
            df["WeekOfYear"] = dt.dt.isocalendar().week.fillna(1).astype(int)

    defaults = {
        "Open": 1,
        "Promo": 0,
        "DayOfWeek": 1,
        "StateHoliday": "0",
        "SchoolHoliday": 0,
        "StoreType": "a",
        "Assortment": "a",
        "PromoInterval": "Unknown",
        "CompetitionDistance": 1270.0,
        "CompetitionOpenSinceMonth": 0.0,
        "CompetitionOpenSinceYear": 0.0,
        "Promo2": 0,
        "Promo2SinceWeek": 0.0,
        "Promo2SinceYear": 0.0,
        "Year": 2015,
        "Month": 1,
        "Day": 1,
        "WeekOfYear": 1,
        "Sales_rolling_mean_7": 0.0,
        "Sales_rolling_std_7": 0.0,
    }

    for lag_idx in range(1, 15):
        defaults[f"Sales_lag_{lag_idx}"] = 0.0

    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = df[col].fillna(default_val)

    if "StateHoliday" in df.columns:
        df["StateHoliday"] = df["StateHoliday"].astype(str)

    missing = [col for col in required_features if col not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")

    return df

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global model, preprocessor, features_manifest, quantile_models

    if model is None or preprocessor is None or features_manifest is None:
        try:
            load_artifacts()
        except Exception:
            raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    if not request.data:
        raise HTTPException(status_code=400, detail="Data payload cannot be empty")

    try:
        required_features = features_manifest["features"]
        raw_records = [item.model_dump() for item in request.data]
        df = _prepare_dataframe(raw_records, required_features)
        
        # Make sure RecursiveForecaster can be imported
        import sys
        if str(BASE_DIR / "src") not in sys.path:
            sys.path.insert(0, str(BASE_DIR / "src"))
        from recursive_forecast import RecursiveForecaster

        forecaster = RecursiveForecaster(model, preprocessor, features_manifest, quantile_models=quantile_models)
        preds_df = forecaster.forecast(df, historical_sales)
        
        results: List[SinglePrediction] = []
        for i, row in preds_df.iterrows():
            store_id = int(row.get("Store", 0))
            date_str = str(row.get("Date")) if "Date" in row and pd.notna(row["Date"]) else None
            open_val = int(row.get("Open", 1))
            
            p10 = row.get("Sales_P10", None)
            p90 = row.get("Sales_P90", None)
            
            res = SinglePrediction(
                store=store_id,
                date=date_str,
                prediction=round(float(row["Sales_Pred"]), 2),
                prediction_interval_p10=round(float(p10), 2) if pd.notna(p10) and p10 is not None else None,
                prediction_interval_p90=round(float(p90), 2) if pd.notna(p90) and p90 is not None else None,
                open=open_val,
            )
            results.append(res)

        return PredictionResponse(
            predictions=[res.prediction for res in results],
            results=results,
            model_name=metadata.get("best_model", "XGBRegressor"),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port)
