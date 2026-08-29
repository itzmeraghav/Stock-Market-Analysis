from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator

from stockmarketanalytics.schemas.prediction_schemas import PredictedDirection


class OptionCalculationRequest(BaseModel):
    symbol: str
    spot_price: float
    strike_price: float
    days_to_expiry: int
    risk_free_rate: float
    volatility: float

    @field_validator("spot_price", "strike_price")
    @classmethod
    def price_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("price fields must be greater than zero")
        return v

    @field_validator("days_to_expiry")
    @classmethod
    def expiry_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("days_to_expiry must be greater than zero")
        return v

    @field_validator("volatility")
    @classmethod
    def volatility_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("volatility must be greater than zero")
        return v


class OptionCalculationResponse(BaseModel):
    call_price: float
    put_price: float
    call_delta: float
    put_delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class OptionCalculationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    stock_id: int
    calculation_date: datetime
    spot_price: float
    strike_price: float
    time_to_expiry: float
    risk_free_rate: float
    volatility: float
    call_price: float
    put_price: float
    call_delta: float
    put_delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class OptionForecastRequest(BaseModel):
    strike_price: float
    days_to_expiry: int
    risk_free_rate: float
    volatility: float | None = None
    model_name: str = "LinearRegression"

    @field_validator("strike_price")
    @classmethod
    def strike_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("strike_price must be greater than zero")
        return v

    @field_validator("days_to_expiry")
    @classmethod
    def expiry_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("days_to_expiry must be greater than zero")
        return v

    @field_validator("volatility")
    @classmethod
    def volatility_must_be_positive_if_given(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("volatility must be greater than zero")
        return v


class OptionForecastResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    expected_change_percent: float
    direction: PredictedDirection
    model: str
    historical_volatility: float
    historical_volatility_percent: float
    call_price: float
    put_price: float
    call_delta: float
    put_delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
