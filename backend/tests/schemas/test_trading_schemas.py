from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from stockmarketanalytics.schemas.trading_schemas import (
    TradingAnalysisOut,
    TradingAnalysisRequest,
    TradingAnalysisResponse,
)

VALID_REQUEST_PAYLOAD = {
    "investment_amount": 1000.0,
    "risk_percentage": 10.0,
    "target_percentage": 20.0,
}


class TestTradingAnalysisRequest:
    def test_accepts_valid_payload(self):
        request = TradingAnalysisRequest(**VALID_REQUEST_PAYLOAD)

        assert request.investment_amount == 1000.0
        assert request.risk_percentage == 10.0
        assert request.target_percentage == 20.0

    @pytest.mark.parametrize("investment_amount", [0, -1, -100.5])
    def test_rejects_non_positive_investment_amount(self, investment_amount):
        with pytest.raises(ValidationError):
            TradingAnalysisRequest(
                **{**VALID_REQUEST_PAYLOAD, "investment_amount": investment_amount}
            )

    @pytest.mark.parametrize("risk_percentage", [0, -5, 100, 150])
    def test_rejects_out_of_range_risk_percentage(self, risk_percentage):
        with pytest.raises(ValidationError):
            TradingAnalysisRequest(
                **{**VALID_REQUEST_PAYLOAD, "risk_percentage": risk_percentage}
            )

    @pytest.mark.parametrize("risk_percentage", [0.01, 50.0, 99.99])
    def test_accepts_risk_percentage_within_open_interval(self, risk_percentage):
        request = TradingAnalysisRequest(
            **{**VALID_REQUEST_PAYLOAD, "risk_percentage": risk_percentage}
        )

        assert request.risk_percentage == risk_percentage

    @pytest.mark.parametrize("target_percentage", [0, -1, -50.0])
    def test_rejects_non_positive_target_percentage(self, target_percentage):
        with pytest.raises(ValidationError):
            TradingAnalysisRequest(
                **{**VALID_REQUEST_PAYLOAD, "target_percentage": target_percentage}
            )

    def test_rejects_missing_required_field(self):
        payload = {
            k: v for k, v in VALID_REQUEST_PAYLOAD.items() if k != "risk_percentage"
        }

        with pytest.raises(ValidationError):
            TradingAnalysisRequest(**payload)


class TestTradingAnalysisResponse:
    def _payload(self, **overrides):
        from stockmarketanalytics.schemas.prediction_schemas import PredictedDirection

        payload = {
            "symbol": "RELIANCE",
            "current_price": 100.0,
            "predicted_price": 110.0,
            "expected_change_percent": 10.0,
            "direction": next(iter(PredictedDirection)),
            "model": "xgboost",
            "investment_amount": 1000.0,
            "shares": 10,
            "invested_amount": 1000.0,
            "remaining_amount": 0.0,
            "risk_percentage": 10.0,
            "stop_loss_price": 90.0,
            "target_percentage": 20.0,
            "target_price": 120.0,
            "maximum_loss": 100.0,
            "potential_profit": 200.0,
            "risk_reward_ratio": 2.0,
        }
        payload.update(overrides)
        return payload

    def test_builds_from_valid_payload(self):
        response = TradingAnalysisResponse(**self._payload())

        assert response.symbol == "RELIANCE"
        assert response.shares == 10
        assert response.risk_reward_ratio == pytest.approx(2.0)

    def test_rejects_invalid_direction_value(self):
        with pytest.raises(ValidationError):
            TradingAnalysisResponse(
                **self._payload(direction="SIDEWAYS_NOT_A_REAL_DIRECTION")
            )

    def test_rejects_missing_required_field(self):
        payload = self._payload()
        del payload["symbol"]

        with pytest.raises(ValidationError):
            TradingAnalysisResponse(**payload)


class TestTradingAnalysisOut:
    def test_builds_from_orm_like_object(self):
        from stockmarketanalytics.schemas.prediction_schemas import PredictedDirection

        record = SimpleNamespace(
            id=1,
            stock_id=1,
            calculation_date=datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
            current_price=100.0,
            predicted_price=110.0,
            expected_change_percent=10.0,
            direction=next(iter(PredictedDirection)).value,
            model_name="xgboost",
            investment_amount=1000.0,
            shares=10,
            invested_amount=1000.0,
            remaining_amount=0.0,
            risk_percentage=10.0,
            stop_loss_price=90.0,
            target_percentage=20.0,
            target_price=120.0,
            maximum_loss=100.0,
            potential_profit=200.0,
            risk_reward_ratio=2.0,
        )

        out = TradingAnalysisOut.model_validate(record)

        assert out.id == 1
        assert out.stock_id == 1
        assert out.model_name == "xgboost"
        assert out.calculation_date == datetime(2024, 1, 1, 9, 30, tzinfo=UTC)

    def test_rejects_missing_required_field(self):
        record = SimpleNamespace(id=1, stock_id=1)

        with pytest.raises(ValidationError):
            TradingAnalysisOut.model_validate(record)
