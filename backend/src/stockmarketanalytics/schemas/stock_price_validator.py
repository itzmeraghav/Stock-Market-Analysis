from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ValidationError, field_validator, model_validator


class StockPriceIn(BaseModel):
    symbol: str
    trading_date: date
    open: float
    high: float
    low: float
    close: float
    volume: int

    @field_validator("symbol")
    @classmethod
    def symbol_not_empty(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("symbol must not be empty")
        return v

    @field_validator("open", "high", "low", "close")
    @classmethod
    def price_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("price fields must be greater than zero")
        return v

    @field_validator("volume")
    @classmethod
    def volume_must_be_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("volume must not be negative")
        return v

    @model_validator(mode="after")
    def check_ohlc_consistency(self) -> StockPriceIn:
        if self.high < self.low:
            raise ValueError("high must not be less than low")
        if self.high < self.open or self.high < self.close:
            raise ValueError("high must be the highest value of the day")
        if self.low > self.open or self.low > self.close:
            raise ValueError("low must be the lowest value of the day")
        return self


class StockPriceValidationError(Exception):
    def __init__(self, symbol: str, trading_date: date, reason: str):
        self.symbol = symbol
        self.trading_date = trading_date
        self.reason = reason
        super().__init__(f"{symbol} on {trading_date}: {reason}")


def validate_price_frame(
    symbol: str, records: list[dict]
) -> tuple[list[StockPriceIn], list[StockPriceValidationError]]:
    valid: list[StockPriceIn] = []
    errors: list[StockPriceValidationError] = []

    for record in records:
        try:
            item = StockPriceIn(symbol=symbol, **record)
            valid.append(item)
        except ValidationError as exc:
            errors.append(
                StockPriceValidationError(symbol, record.get("trading_date"), str(exc))
            )

    return valid, errors


def dedupe_by_stock_id_and_date(valid_items: list[StockPriceIn]) -> list[StockPriceIn]:
    seen: set[tuple[str, date]] = set()
    deduped: list[StockPriceIn] = []

    for item in valid_items:
        key = (item.symbol, item.trading_date)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped
