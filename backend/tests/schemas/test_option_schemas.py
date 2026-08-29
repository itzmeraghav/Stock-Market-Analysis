from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from stockmarketanalytics.schemas.option_schemas import (
    OptionCalculationOut,
    OptionCalculationRequest,
    OptionCalculationResponse,
    OptionForecastRequest,
    OptionForecastResponse,
)
from stockmarketanalytics.schemas.prediction_schemas import PredictedDirection


class TestOptionCalculationRequest:
    def test_accepts_valid_payload(self):
        payload = {
            "symbol": "aapl",
            "spot_price": 150.0,
            "strike_price": 155.0,
            "days_to_expiry": 30,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
        }

        request = OptionCalculationRequest(**payload)

        assert request.symbol == "aapl"
        assert request.spot_price == 150.0
        assert request.strike_price == 155.0
        assert request.days_to_expiry == 30
        assert request.volatility == 0.2

    @pytest.mark.parametrize("field", ["spot_price", "strike_price"])
    @pytest.mark.parametrize("bad_value", [0, -10.5])
    def test_rejects_non_positive_price_fields(self, field, bad_value):
        payload = {
            "symbol": "AAPL",
            "spot_price": 150.0,
            "strike_price": 155.0,
            "days_to_expiry": 30,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
            field: bad_value,
        }

        with pytest.raises(ValidationError, match="greater than zero"):
            OptionCalculationRequest(**payload)

    @pytest.mark.parametrize("bad_value", [0, -5])
    def test_rejects_non_positive_days_to_expiry(self, bad_value):
        payload = {
            "symbol": "AAPL",
            "spot_price": 150.0,
            "strike_price": 155.0,
            "days_to_expiry": bad_value,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
        }

        with pytest.raises(
            ValidationError, match="days_to_expiry must be greater than zero"
        ):
            OptionCalculationRequest(**payload)

    @pytest.mark.parametrize("bad_value", [0, -0.1])
    def test_rejects_non_positive_volatility(self, bad_value):
        payload = {
            "symbol": "AAPL",
            "spot_price": 150.0,
            "strike_price": 155.0,
            "days_to_expiry": 30,
            "risk_free_rate": 0.05,
            "volatility": bad_value,
        }

        with pytest.raises(
            ValidationError, match="volatility must be greater than zero"
        ):
            OptionCalculationRequest(**payload)

    def test_missing_required_field_raises(self):
        payload = {
            "symbol": "AAPL",
            "spot_price": 150.0,
            "strike_price": 155.0,
            "days_to_expiry": 30,
            "volatility": 0.2,
        }

        with pytest.raises(ValidationError):
            OptionCalculationRequest(**payload)


class TestOptionForecastRequest:
    def test_accepts_valid_payload_with_defaults(self):
        payload = {
            "strike_price": 155.0,
            "days_to_expiry": 30,
            "risk_free_rate": 0.05,
        }

        request = OptionForecastRequest(**payload)

        assert request.volatility is None
        assert request.model_name == "LinearRegression"

    def test_accepts_explicit_volatility_and_model_name(self):
        payload = {
            "strike_price": 155.0,
            "days_to_expiry": 30,
            "risk_free_rate": 0.05,
            "volatility": 0.25,
            "model_name": "RandomForest",
        }

        request = OptionForecastRequest(**payload)

        assert request.volatility == 0.25
        assert request.model_name == "RandomForest"

    @pytest.mark.parametrize("bad_value", [0, -12])
    def test_rejects_non_positive_strike_price(self, bad_value):
        payload = {
            "strike_price": bad_value,
            "days_to_expiry": 30,
            "risk_free_rate": 0.05,
        }

        with pytest.raises(
            ValidationError, match="strike_price must be greater than zero"
        ):
            OptionForecastRequest(**payload)

    @pytest.mark.parametrize("bad_value", [0, -1])
    def test_rejects_non_positive_days_to_expiry(self, bad_value):
        payload = {
            "strike_price": 155.0,
            "days_to_expiry": bad_value,
            "risk_free_rate": 0.05,
        }

        with pytest.raises(
            ValidationError, match="days_to_expiry must be greater than zero"
        ):
            OptionForecastRequest(**payload)

    @pytest.mark.parametrize("bad_value", [0, -0.5])
    def test_rejects_non_positive_volatility_when_given(self, bad_value):
        payload = {
            "strike_price": 155.0,
            "days_to_expiry": 30,
            "risk_free_rate": 0.05,
            "volatility": bad_value,
        }

        with pytest.raises(
            ValidationError, match="volatility must be greater than zero"
        ):
            OptionForecastRequest(**payload)

    def test_none_volatility_is_valid(self):

        payload = {
            "strike_price": 155.0,
            "days_to_expiry": 30,
            "risk_free_rate": 0.05,
            "volatility": None,
        }

        request = OptionForecastRequest(**payload)

        assert request.volatility is None


class TestOptionCalculationOut:
    def test_builds_from_orm_like_object(self):
        from types import SimpleNamespace

        orm_like = SimpleNamespace(
            id=1,
            stock_id=42,
            calculation_date=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            spot_price=150.0,
            strike_price=155.0,
            time_to_expiry=0.0822,
            risk_free_rate=0.05,
            volatility=0.2,
            call_price=5.12,
            put_price=8.34,
            call_delta=0.45,
            put_delta=-0.55,
            gamma=0.03,
            vega=0.12,
            theta=-0.02,
            rho=0.04,
        )

        out = OptionCalculationOut.model_validate(orm_like)

        assert out.id == 1
        assert out.stock_id == 42
        assert out.call_price == 5.12
        assert out.put_delta == -0.55

    def test_rejects_missing_fields(self):
        incomplete_payload = {"id": 1, "stock_id": 42}

        with pytest.raises(ValidationError):
            OptionCalculationOut(**incomplete_payload)


class TestOptionCalculationResponse:
    def test_constructs_with_all_greeks(self):
        payload = {
            "call_price": 5.1234,
            "put_price": 8.3456,
            "call_delta": 0.4512,
            "put_delta": -0.5488,
            "gamma": 0.031245,
            "vega": 0.1234,
            "theta": -0.0234,
            "rho": 0.0456,
        }

        response = OptionCalculationResponse(**payload)

        assert response.model_dump() == payload


class TestOptionForecastResponse:
    def test_constructs_with_prediction_and_option_fields(self):
        payload = {
            "symbol": "AAPL",
            "current_price": 150.0,
            "predicted_price": 155.0,
            "expected_change_percent": 3.33,
            "direction": PredictedDirection.BULLISH,
            "model": "LinearRegression",
            "historical_volatility": 0.2123,
            "historical_volatility_percent": 21.23,
            "call_price": 5.12,
            "put_price": 8.34,
            "call_delta": 0.45,
            "put_delta": -0.55,
            "gamma": 0.03,
            "vega": 0.12,
            "theta": -0.02,
            "rho": 0.04,
        }

        response = OptionForecastResponse(**payload)

        assert response.direction == PredictedDirection.BULLISH
        assert response.symbol == "AAPL"

    def test_rejects_invalid_direction_value(self):
        payload = {
            "symbol": "AAPL",
            "current_price": 150.0,
            "predicted_price": 155.0,
            "expected_change_percent": 3.33,
            "direction": "sideways",
            "model": "LinearRegression",
            "historical_volatility": 0.2123,
            "historical_volatility_percent": 21.23,
            "call_price": 5.12,
            "put_price": 8.34,
            "call_delta": 0.45,
            "put_delta": -0.55,
            "gamma": 0.03,
            "vega": 0.12,
            "theta": -0.02,
            "rho": 0.04,
        }

        with pytest.raises(ValidationError):
            OptionForecastResponse(**payload)
