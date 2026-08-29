from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from stockmarketanalytics.endpoints import option_endpoints
from stockmarketanalytics.schemas.option_schemas import OptionCalculationResponse
from stockmarketanalytics.schemas.prediction_schemas import PredictedDirection


def _make_option_response(**overrides) -> OptionCalculationResponse:
    defaults = {
        "call_price": 5.12,
        "put_price": 8.34,
        "call_delta": 0.45,
        "put_delta": -0.55,
        "gamma": 0.03,
        "vega": 0.12,
        "theta": -0.02,
        "rho": 0.04,
    }
    defaults.update(overrides)
    return OptionCalculationResponse(**defaults)


class _FakeBlackScholesService:
    """Stand-in for BlackScholesService, configurable per-test."""

    last_kwargs: dict | None = None

    def __init__(self, db):
        self.db = db

    def calculate(self, **kwargs):
        _FakeBlackScholesService.last_kwargs = kwargs
        return _make_option_response()

    def get_history(self, stock, limit: int = 100):
        return []


class _RaisingBlackScholesService:
    def __init__(self, db):
        self.db = db

    def calculate(self, **kwargs):
        raise ValueError("volatility must be greater than zero")


class _FakePredictionService:
    def __init__(self, db):
        self.db = db

    def predict(self, stock, model_name: str = "LinearRegression"):
        return SimpleNamespace(
            current_price=150.0,
            predicted_price=160.0,
            expected_change_percent=6.67,
            direction=PredictedDirection.BEARISH,
            model=model_name,
        )


class _RaisingPredictionService:
    def __init__(self, db):
        self.db = db

    def predict(self, stock, model_name: str = "LinearRegression"):
        raise ValueError(f"Unknown model: {model_name}")


class _FakeVolatilityService:
    def __init__(self, db):
        self.db = db

    def calculate_historical_volatility(self, stock, years: int = 5):
        return 0.25


class _RaisingVolatilityService:
    def __init__(self, db):
        self.db = db

    def calculate_historical_volatility(self, stock, years: int = 5):
        raise ValueError("Insufficient price history")


class TestGetStockOr404:
    def test_returns_stock_when_found(
        self, mock_db_session, make_query_mock, fake_stock
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=fake_stock)

        result = option_endpoints._get_stock_or_404("reliance", mock_db_session)

        assert result is fake_stock

    def test_raises_404_when_not_found(self, mock_db_session, make_query_mock):
        mock_db_session.query.return_value = make_query_mock(first_return=None)

        with pytest.raises(HTTPException) as exc_info:
            option_endpoints._get_stock_or_404("ghost", mock_db_session)

        assert exc_info.value.status_code == 404
        assert "GHOST" in exc_info.value.detail


class TestCalculateOptionEndpoint:
    def test_returns_200_with_calculated_greeks(
        self, client, mock_db_session, make_query_mock, fake_stock, monkeypatch
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=fake_stock)
        monkeypatch.setattr(
            option_endpoints, "BlackScholesService", _FakeBlackScholesService
        )
        payload = {
            "symbol": "RELIANCE",
            "spot_price": 150.0,
            "strike_price": 155.0,
            "days_to_expiry": 30,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
        }

        response = client.post("/options/calculate", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["call_price"] == 5.12
        assert body["put_delta"] == -0.55

    def test_forwards_request_fields_to_service(
        self, client, mock_db_session, make_query_mock, fake_stock, monkeypatch
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=fake_stock)
        monkeypatch.setattr(
            option_endpoints, "BlackScholesService", _FakeBlackScholesService
        )
        payload = {
            "symbol": "RELIANCE",
            "spot_price": 150.0,
            "strike_price": 155.0,
            "days_to_expiry": 30,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
        }

        client.post("/options/calculate", json=payload)

        forwarded = _FakeBlackScholesService.last_kwargs
        assert forwarded["spot_price"] == 150.0
        assert forwarded["strike_price"] == 155.0
        assert forwarded["days_to_expiry"] == 30
        assert forwarded["stock"] is fake_stock

    def test_returns_404_when_stock_not_found(
        self, client, mock_db_session, make_query_mock
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=None)
        payload = {
            "symbol": "GHOST",
            "spot_price": 150.0,
            "strike_price": 155.0,
            "days_to_expiry": 30,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
        }

        response = client.post("/options/calculate", json=payload)

        assert response.status_code == 404
        assert "GHOST" in response.json()["detail"]

    def test_returns_422_for_invalid_payload(
        self, client, mock_db_session, make_query_mock, fake_stock
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=fake_stock)
        payload = {
            "symbol": "RELIANCE",
            "spot_price": -1,
            "strike_price": 155.0,
            "days_to_expiry": 30,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
        }

        response = client.post("/options/calculate", json=payload)

        assert response.status_code == 422


class TestForecastOptionEndpoint:
    def _payload(self, **overrides):
        payload = {
            "strike_price": 155.0,
            "days_to_expiry": 30,
            "risk_free_rate": 0.05,
        }
        payload.update(overrides)
        return payload

    def test_returns_200_with_combined_forecast_and_greeks(
        self, client, mock_db_session, make_query_mock, fake_stock, monkeypatch
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=fake_stock)
        monkeypatch.setattr(
            option_endpoints, "PredictionService", _FakePredictionService
        )
        monkeypatch.setattr(
            option_endpoints, "VolatilityService", _FakeVolatilityService
        )
        monkeypatch.setattr(
            option_endpoints, "BlackScholesService", _FakeBlackScholesService
        )

        response = client.post("/options/forecast/RELIANCE", json=self._payload())

        assert response.status_code == 200
        body = response.json()
        assert body["symbol"] == fake_stock.symbol
        assert body["current_price"] == 150.0
        assert body["predicted_price"] == 160.0
        assert body["direction"] == "Bearish"
        assert body["historical_volatility"] == 0.25
        assert body["historical_volatility_percent"] == 25.0
        assert body["call_price"] == 5.12

    def test_uses_current_price_from_prediction_as_spot_price(
        self, client, mock_db_session, make_query_mock, fake_stock, monkeypatch
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=fake_stock)
        monkeypatch.setattr(
            option_endpoints, "PredictionService", _FakePredictionService
        )
        monkeypatch.setattr(
            option_endpoints, "VolatilityService", _FakeVolatilityService
        )
        monkeypatch.setattr(
            option_endpoints, "BlackScholesService", _FakeBlackScholesService
        )

        client.post("/options/forecast/RELIANCE", json=self._payload())

        forwarded = _FakeBlackScholesService.last_kwargs
        assert forwarded["spot_price"] == 150.0
        assert forwarded["volatility"] == 0.25

    def test_returns_404_when_stock_not_found(
        self, client, mock_db_session, make_query_mock
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=None)

        response = client.post("/options/forecast/GHOST", json=self._payload())

        assert response.status_code == 404

    def test_returns_400_when_prediction_fails(
        self, client, mock_db_session, make_query_mock, fake_stock, monkeypatch
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=fake_stock)
        monkeypatch.setattr(
            option_endpoints, "PredictionService", _RaisingPredictionService
        )

        response = client.post(
            "/options/forecast/RELIANCE", json=self._payload(model_name="BogusModel")
        )

        assert response.status_code == 400
        assert "Unknown model" in response.json()["detail"]

    def test_returns_400_when_volatility_calculation_fails(
        self, client, mock_db_session, make_query_mock, fake_stock, monkeypatch
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=fake_stock)
        monkeypatch.setattr(
            option_endpoints, "PredictionService", _FakePredictionService
        )
        monkeypatch.setattr(
            option_endpoints, "VolatilityService", _RaisingVolatilityService
        )

        response = client.post("/options/forecast/RELIANCE", json=self._payload())

        assert response.status_code == 400
        assert "Insufficient price history" in response.json()["detail"]

    def test_returns_400_when_black_scholes_calculation_fails(
        self, client, mock_db_session, make_query_mock, fake_stock, monkeypatch
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=fake_stock)
        monkeypatch.setattr(
            option_endpoints, "PredictionService", _FakePredictionService
        )
        monkeypatch.setattr(
            option_endpoints, "VolatilityService", _FakeVolatilityService
        )
        monkeypatch.setattr(
            option_endpoints, "BlackScholesService", _RaisingBlackScholesService
        )

        response = client.post("/options/forecast/RELIANCE", json=self._payload())

        assert response.status_code == 400
        assert "volatility must be greater than zero" in response.json()["detail"]

    def test_request_volatility_field_is_not_forwarded_to_black_scholes(
        self, client, mock_db_session, make_query_mock, fake_stock, monkeypatch
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=fake_stock)
        monkeypatch.setattr(
            option_endpoints, "PredictionService", _FakePredictionService
        )
        monkeypatch.setattr(
            option_endpoints, "VolatilityService", _FakeVolatilityService
        )
        monkeypatch.setattr(
            option_endpoints, "BlackScholesService", _FakeBlackScholesService
        )

        client.post("/options/forecast/RELIANCE", json=self._payload(volatility=0.9))

        forwarded = _FakeBlackScholesService.last_kwargs
        assert forwarded["volatility"] == 0.25


class TestGetOptionHistoryEndpoint:
    def test_returns_200_with_history_list(
        self, client, mock_db_session, make_query_mock, fake_stock, monkeypatch
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=fake_stock)
        history_row = SimpleNamespace(
            id=1,
            stock_id=fake_stock.id,
            calculation_date="2024-01-01T12:00:00",
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

        class _FakeServiceWithHistory(_FakeBlackScholesService):
            def get_history(self, stock, limit: int = 100):
                return [history_row]

        monkeypatch.setattr(
            option_endpoints, "BlackScholesService", _FakeServiceWithHistory
        )

        response = client.get("/options/RELIANCE")

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["call_price"] == 5.12

    def test_returns_empty_list_when_no_history(
        self, client, mock_db_session, make_query_mock, fake_stock, monkeypatch
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=fake_stock)
        monkeypatch.setattr(
            option_endpoints, "BlackScholesService", _FakeBlackScholesService
        )

        response = client.get("/options/RELIANCE")

        assert response.status_code == 200
        assert response.json() == []

    def test_returns_404_when_stock_not_found(
        self, client, mock_db_session, make_query_mock
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=None)

        response = client.get("/options/GHOST")

        assert response.status_code == 404
