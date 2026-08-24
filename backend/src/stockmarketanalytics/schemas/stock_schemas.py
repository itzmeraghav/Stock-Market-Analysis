from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class StockOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    symbol: str
    company_name: str
    exchange: str
    created_at: datetime


class StockPriceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    stock_id: int
    trading_date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


class TechnicalIndicatorOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    stock_price_id: int
    sma20: float
    sma50: float
    ema20: float
    rsi14: float
    macd: float
    bollinger_upper: float
    bollinger_lower: float
    volatility: float


class StockUpdateResult(BaseModel):
    symbol: str
    inserted: int
    indicators_inserted: int = 0
    predictions_reconciled: int = 0
