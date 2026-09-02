from __future__ import annotations

from unittest.mock import patch

import pytest

VALID_PAYLOAD = {
    "investment_amount": 1000.0,
    "risk_percentage": 10.0,
    "target_percentage": 20.0,
}


class TestAnalyzeTradeEndpoint:
    def test_returns_404_when_stock_not_found(
        self, trading_client, mock_db_session, make_query_mock
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=None)

        response = trading_client.post("/trading/analyze/UNKNOWN", json=VALID_PAYLOAD)

        assert response.status_code == 404
        assert "UNKNOWN" in response.json()["detail"]

    def test_returns_422_for_non_positive_investment_amount(
        self, trading_client, mock_db_session, fake_stock, make_query_mock
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=fake_stock)
        payload = {**VALID_PAYLOAD, "investment_amount": 0}

        response = trading_client.post("/trading/analyze/RELIANCE", json=payload)

        assert response.status_code == 422

    def test_returns_422_for_out_of_range_risk_percentage(
        self, trading_client, mock_db_session, fake_stock, make_query_mock
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=fake_stock)
        payload = {**VALID_PAYLOAD, "risk_percentage": 100.0}

        response = trading_client.post("/trading/analyze/RELIANCE", json=payload)

        assert response.status_code == 422

    def test_returns_422_for_non_positive_target_percentage(
        self, trading_client, mock_db_session, fake_stock, make_query_mock
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=fake_stock)
        payload = {**VALID_PAYLOAD, "target_percentage": 0}

        response = trading_client.post("/trading/analyze/RELIANCE", json=payload)

        assert response.status_code == 422

    @patch("stockmarketanalytics.endpoints.trading_endpoints.RiskManagementService")
    def test_returns_400_when_service_raises_value_error(
        self,
        mock_service_cls,
        trading_client,
        mock_db_session,
        fake_stock,
        make_query_mock,
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=fake_stock)
        mock_service_cls.return_value.analyze_trade.side_effect = ValueError(
            "Investment amount too small"
        )

        response = trading_client.post("/trading/analyze/RELIANCE", json=VALID_PAYLOAD)

        assert response.status_code == 400
        assert response.json()["detail"] == "Investment amount too small"

    @patch("stockmarketanalytics.endpoints.trading_endpoints.RiskManagementService")
    def test_returns_200_with_analysis_on_success(
        self,
        mock_service_cls,
        trading_client,
        mock_db_session,
        fake_stock,
        make_query_mock,
        fake_prediction,
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=fake_stock)
        expected = {
            "symbol": fake_stock.symbol,
            "current_price": 100.0,
            "predicted_price": 110.0,
            "expected_change_percent": 10.0,
            "direction": fake_prediction.direction,
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
        mock_service_cls.return_value.analyze_trade.return_value = expected

        response = trading_client.post("/trading/analyze/reliance", json=VALID_PAYLOAD)

        assert response.status_code == 200
        body = response.json()
        assert body["symbol"] == fake_stock.symbol
        assert body["shares"] == 10
        assert body["risk_reward_ratio"] == pytest.approx(2.0)

    @patch("stockmarketanalytics.endpoints.trading_endpoints.RiskManagementService")
    def test_passes_request_fields_through_to_service(
        self,
        mock_service_cls,
        trading_client,
        mock_db_session,
        fake_stock,
        make_query_mock,
        fake_prediction,
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=fake_stock)
        mock_service_cls.return_value.analyze_trade.return_value = {
            "symbol": fake_stock.symbol,
            "current_price": 100.0,
            "predicted_price": 110.0,
            "expected_change_percent": 10.0,
            "direction": fake_prediction.direction,
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

        trading_client.post("/trading/analyze/RELIANCE", json=VALID_PAYLOAD)

        _, kwargs = mock_service_cls.return_value.analyze_trade.call_args
        assert kwargs["investment_amount"] == VALID_PAYLOAD["investment_amount"]
        assert kwargs["risk_percentage"] == VALID_PAYLOAD["risk_percentage"]
        assert kwargs["target_percentage"] == VALID_PAYLOAD["target_percentage"]
        assert kwargs["stock"] is fake_stock
