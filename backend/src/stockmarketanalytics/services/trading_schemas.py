from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator
from stockmarketanalytics.schemas.prediction_schemas import PredictedDirection


class TradingAnalysisRequest(BaseModel):
    investment_amount: float
    risk_percentage: float
    target_percentage: float

    @field_validator("investment_amount")
    @classmethod
    def investment_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("investment_amount must be greater than zero")
        return v

    @field_validator("risk_percentage")
    @classmethod
    def risk_percentage_must_be_valid(cls, v: float) -> float:
        if v <= 0 or v >= 100:
            raise ValueError("risk_percentage must be greater than 0 and less than 100")
        return v

    @field_validator("target_percentage")
    @classmethod
    def target_percentage_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("target_percentage must be greater than zero")
        return v


class TradingAnalysisResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    expected_change_percent: float
    direction: PredictedDirection
    model: str

    investment_amount: float
    shares: int
    invested_amount: float
    remaining_amount: float

    risk_percentage: float
    stop_loss_price: float

    target_percentage: float
    target_price: float

    maximum_loss: float
    potential_profit: float
    risk_reward_ratio: float


class TradingAnalysisOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    stock_id: int
    calculation_date: datetime
    current_price: float
    predicted_price: float
    expected_change_percent: float
    direction: str
    model_name: str

    investment_amount: float
    shares: int
    invested_amount: float
    remaining_amount: float

    risk_percentage: float
    stop_loss_price: float

    target_percentage: float
    target_price: float

    maximum_loss: float
    potential_profit: float
    risk_reward_ratio: float
