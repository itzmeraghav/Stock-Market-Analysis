from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from stockmarketanalytics.services.risk_management_service import RiskManagementService


class TestCalculateShares:
    def test_returns_floor_of_investment_divided_by_price(self, mock_db_session):
        service = RiskManagementService(mock_db_session)
        assert service.calculate_shares(1000.0, 300.0) == 3

    def test_returns_zero_when_investment_smaller_than_price(self, mock_db_session):
        service = RiskManagementService(mock_db_session)
        assert service.calculate_shares(50.0, 300.0) == 0

    def test_raises_when_price_is_zero(self, mock_db_session):
        service = RiskManagementService(mock_db_session)
        with pytest.raises(ValueError):
            service.calculate_shares(1000.0, 0.0)

    def test_raises_when_price_is_negative(self, mock_db_session):
        service = RiskManagementService(mock_db_session)
        with pytest.raises(ValueError):
            service.calculate_shares(1000.0, -50.0)


class TestCalculateInvestedAmount:
    def test_multiplies_shares_by_price(self, mock_db_session):
        service = RiskManagementService(mock_db_session)
        assert service.calculate_invested_amount(3, 300.0) == 900.0

    def test_returns_zero_for_zero_shares(self, mock_db_session):
        service = RiskManagementService(mock_db_session)
        assert service.calculate_invested_amount(0, 300.0) == 0.0


class TestCalculateRemainingAmount:
    def test_subtracts_invested_from_investment(self, mock_db_session):
        service = RiskManagementService(mock_db_session)
        assert service.calculate_remaining_amount(1000.0, 900.0) == 100.0

    def test_returns_zero_when_fully_invested(self, mock_db_session):
        service = RiskManagementService(mock_db_session)
        assert service.calculate_remaining_amount(900.0, 900.0) == 0.0


class TestCalculateStopLoss:
    def test_applies_risk_percentage_below_current_price(self, mock_db_session):
        service = RiskManagementService(mock_db_session)
        assert service.calculate_stop_loss(100.0, 10.0) == pytest.approx(90.0)

    def test_returns_zero_when_risk_is_hundred_percent(self, mock_db_session):
        service = RiskManagementService(mock_db_session)
        assert service.calculate_stop_loss(100.0, 100.0) == pytest.approx(0.0)


class TestCalculateTargetPrice:
    def test_applies_target_percentage_above_current_price(self, mock_db_session):
        service = RiskManagementService(mock_db_session)
        assert service.calculate_target_price(100.0, 10.0) == pytest.approx(110.0)

    def test_returns_current_price_when_target_is_zero(self, mock_db_session):
        service = RiskManagementService(mock_db_session)
        assert service.calculate_target_price(100.0, 0.0) == pytest.approx(100.0)


class TestCalculateMaximumLoss:
    def test_multiplies_loss_per_share_by_shares(self, mock_db_session):
        service = RiskManagementService(mock_db_session)
        assert service.calculate_maximum_loss(100.0, 90.0, 10) == pytest.approx(100.0)

    def test_returns_zero_when_stop_loss_equals_current_price(self, mock_db_session):
        service = RiskManagementService(mock_db_session)
        assert service.calculate_maximum_loss(100.0, 100.0, 10) == pytest.approx(0.0)


class TestCalculatePotentialProfit:
    def test_multiplies_profit_per_share_by_shares(self, mock_db_session):
        service = RiskManagementService(mock_db_session)
        assert service.calculate_potential_profit(100.0, 110.0, 10) == pytest.approx(
            100.0
        )

    def test_returns_zero_when_target_equals_current_price(self, mock_db_session):
        service = RiskManagementService(mock_db_session)
        assert service.calculate_potential_profit(100.0, 100.0, 10) == pytest.approx(
            0.0
        )


class TestCalculateRiskReward:
    def test_divides_profit_by_loss(self, mock_db_session):
        service = RiskManagementService(mock_db_session)
        assert service.calculate_risk_reward(200.0, 100.0) == pytest.approx(2.0)

    def test_raises_when_maximum_loss_is_zero(self, mock_db_session):
        service = RiskManagementService(mock_db_session)
        with pytest.raises(ValueError):
            service.calculate_risk_reward(200.0, 0.0)

    def test_raises_when_maximum_loss_is_negative(self, mock_db_session):
        service = RiskManagementService(mock_db_session)
        with pytest.raises(ValueError):
            service.calculate_risk_reward(200.0, -50.0)


class TestAnalyzeTrade:
    @patch("stockmarketanalytics.services.risk_management_service.PredictionService")
    def test_returns_populated_response_for_valid_trade(
        self,
        mock_prediction_service_cls,
        mock_db_session,
        fake_stock,
        fake_prediction_result,
    ):
        mock_prediction_service_cls.return_value.predict_all.return_value = (
            fake_prediction_result
        )
        service = RiskManagementService(mock_db_session)

        response = service.analyze_trade(
            stock=fake_stock,
            investment_amount=1000.0,
            risk_percentage=10.0,
            target_percentage=20.0,
            persist=False,
        )

        assert response.symbol == fake_stock.symbol
        assert response.shares == 10
        assert response.invested_amount == pytest.approx(1000.0)
        assert response.remaining_amount == pytest.approx(0.0)
        assert response.stop_loss_price == pytest.approx(90.0)
        assert response.target_price == pytest.approx(120.0)
        assert response.maximum_loss == pytest.approx(100.0)
        assert response.potential_profit == pytest.approx(200.0)
        assert response.risk_reward_ratio == pytest.approx(2.0)

    @patch("stockmarketanalytics.services.risk_management_service.PredictionService")
    def test_selects_best_model_prediction(
        self, mock_prediction_service_cls, mock_db_session, fake_stock, fake_prediction
    ):
        other_prediction = SimpleNamespace(
            model="linear_regression",
            current_price=999.0,
            predicted_price=999.0,
            expected_change_percent=0.0,
            direction=fake_prediction.direction,
        )
        prediction_result = SimpleNamespace(
            best_model=fake_prediction.model,
            predictions=[other_prediction, fake_prediction],
        )
        mock_prediction_service_cls.return_value.predict_all.return_value = (
            prediction_result
        )
        service = RiskManagementService(mock_db_session)

        response = service.analyze_trade(
            stock=fake_stock,
            investment_amount=1000.0,
            risk_percentage=10.0,
            target_percentage=20.0,
            persist=False,
        )

        assert response.model == fake_prediction.model
        assert response.current_price == pytest.approx(fake_prediction.current_price)

    @patch("stockmarketanalytics.services.risk_management_service.PredictionService")
    def test_raises_when_investment_too_small_for_one_share(
        self,
        mock_prediction_service_cls,
        mock_db_session,
        fake_stock,
        fake_prediction_result,
    ):
        mock_prediction_service_cls.return_value.predict_all.return_value = (
            fake_prediction_result
        )
        service = RiskManagementService(mock_db_session)

        with pytest.raises(ValueError):
            service.analyze_trade(
                stock=fake_stock,
                investment_amount=1.0,
                risk_percentage=10.0,
                target_percentage=20.0,
                persist=False,
            )

    @patch("stockmarketanalytics.services.risk_management_service.PredictionService")
    def test_does_not_touch_db_when_persist_is_false(
        self,
        mock_prediction_service_cls,
        mock_db_session,
        fake_stock,
        fake_prediction_result,
    ):
        mock_prediction_service_cls.return_value.predict_all.return_value = (
            fake_prediction_result
        )
        service = RiskManagementService(mock_db_session)

        service.analyze_trade(
            stock=fake_stock,
            investment_amount=1000.0,
            risk_percentage=10.0,
            target_percentage=20.0,
            persist=False,
        )

        mock_db_session.add.assert_not_called()
        mock_db_session.commit.assert_not_called()

    @patch("stockmarketanalytics.services.risk_management_service.PredictionService")
    def test_persists_trading_analysis_when_persist_is_true(
        self,
        mock_prediction_service_cls,
        mock_db_session,
        fake_stock,
        fake_prediction_result,
    ):
        mock_prediction_service_cls.return_value.predict_all.return_value = (
            fake_prediction_result
        )
        service = RiskManagementService(mock_db_session)

        service.analyze_trade(
            stock=fake_stock,
            investment_amount=1000.0,
            risk_percentage=10.0,
            target_percentage=20.0,
            persist=True,
        )

        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        persisted = mock_db_session.add.call_args[0][0]
        assert persisted.stock_id == fake_stock.id
        assert persisted.shares == 10
        assert persisted.risk_reward_ratio == pytest.approx(2.0)
