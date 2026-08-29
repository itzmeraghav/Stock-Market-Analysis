from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from stockmarketanalytics.schemas.stock_schemas import (
    StockOut,
    StockPriceOut,
    StockUpdateResult,
    TechnicalIndicatorOut,
)


class TestStockOut:
    def test_builds_from_orm_style_object(self):
        class FakeStock:
            id = 1
            symbol = "INFY"
            company_name = "Infosys Ltd"
            exchange = "NSE"
            created_at = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)

        out = StockOut.model_validate(FakeStock())

        assert out.id == 1
        assert out.symbol == "INFY"
        assert out.exchange == "NSE"

    def test_builds_from_dict(self):
        payload = {
            "id": 2,
            "symbol": "TCS",
            "company_name": "Tata Consultancy Services",
            "exchange": "NSE",
            "created_at": datetime(2024, 1, 1, tzinfo=UTC),
        }

        out = StockOut(**payload)

        assert out.symbol == "TCS"

    def test_missing_required_field_raises(self):
        payload = {
            "id": 2,
            "company_name": "Tata Consultancy Services",
            "exchange": "NSE",
            "created_at": datetime(2024, 1, 1, tzinfo=UTC),
        }

        with pytest.raises(ValidationError):
            StockOut(**payload)


class TestStockPriceOut:
    def test_valid_payload(self):
        out = StockPriceOut(
            id=1,
            stock_id=1,
            trading_date=date(2024, 1, 2),
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0,
            volume=10_000,
        )

        assert out.close == 105.0
        assert out.volume == 10_000

    def test_non_numeric_price_raises(self):
        with pytest.raises(ValidationError):
            StockPriceOut(
                id=1,
                stock_id=1,
                trading_date=date(2024, 1, 2),
                open="not-a-number",
                high=110.0,
                low=95.0,
                close=105.0,
                volume=10_000,
            )


class TestTechnicalIndicatorOut:
    def test_valid_payload(self):
        out = TechnicalIndicatorOut(
            id=1,
            stock_price_id=1,
            sma20=100.0,
            sma50=98.0,
            ema20=101.0,
            rsi14=55.0,
            macd=0.5,
            bollinger_upper=110.0,
            bollinger_lower=90.0,
            volatility=0.2,
        )

        assert out.rsi14 == 55.0

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            TechnicalIndicatorOut(
                id=1,
                stock_price_id=1,
                sma20=100.0,
                sma50=98.0,
                ema20=101.0,
                rsi14=55.0,
                macd=0.5,
                bollinger_upper=110.0,
                volatility=0.2,
            )


class TestStockUpdateResult:
    def test_defaults_are_zero(self):
        result = StockUpdateResult(symbol="INFY", inserted=5)

        assert result.indicators_inserted == 0
        assert result.predictions_reconciled == 0

    def test_explicit_values_are_kept(self):
        result = StockUpdateResult(
            symbol="INFY",
            inserted=5,
            indicators_inserted=5,
            predictions_reconciled=2,
        )

        assert result.indicators_inserted == 5
        assert result.predictions_reconciled == 2
