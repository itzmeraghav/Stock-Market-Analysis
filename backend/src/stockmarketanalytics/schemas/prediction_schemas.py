from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class PredictedDirection(str, Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    FLAT = "Flat"


class PredictionTrainResult(BaseModel):
    symbol: str
    model_name: str
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    trained_at: datetime


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    stock_id: int
    prediction_date: date
    target_date: date
    actual_close: float | None = None
    predicted_close: float
    predicted_direction: PredictedDirection
    confidence: float = Field(ge=0.0, le=1.0)
    model_name: str
    created_at: datetime


class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    expected_change_percent: float
    direction: PredictedDirection
    model: str


class BacktestResult(BaseModel):
    symbol: str
    model_name: str
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    actual_series: list[float]
    predicted_series: list[float]
    dates: list[date]


class ModelComparisonEntry(BaseModel):
    model_name: str
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    training_time_seconds: float


class ModelComparisonResponse(BaseModel):
    symbol: str
    results: list[ModelComparisonEntry]


class MultiModelPredictionResponse(BaseModel):
    symbol: str
    predictions: list[PredictionResponse]
    best_model: str
    best_model_rmse: float | None = None
