from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from stockmarketanalytics.endpoints import prediction_endpoints
from stockmarketanalytics.schemas.prediction_schemas import ModelComparisonEntry


@pytest.fixture
def mock_db_session(monkeypatch):
    db = MagicMock()
    monkeypatch.setattr(
        prediction_endpoints,
        "get_db_session",
        MagicMock(return_value=db),
    )
    return db


@pytest.fixture
def stub_stock_lookup(monkeypatch, fake_stock):
    stub = MagicMock(return_value=fake_stock)
    monkeypatch.setattr(
        prediction_endpoints,
        "get_stock_or_404",
        stub,
    )
    return stub


@pytest.fixture
def mock_prediction_service(monkeypatch):
    instance = MagicMock()

    monkeypatch.setattr(
        prediction_endpoints,
        "PredictionService",
        MagicMock(return_value=instance),
    )

    return instance


class TestTrainPredictionModels:
    def test_returns_service_result_on_success(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_prediction_service,
    ):
        expected = [MagicMock(), MagicMock()]
        mock_prediction_service.train_all.return_value = expected

        result = prediction_endpoints.train_prediction_models("RELIANCE")

        assert result == expected

    def test_trains_the_stock_returned_by_lookup(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_prediction_service,
        fake_stock,
    ):
        prediction_endpoints.train_prediction_models("RELIANCE")

        mock_prediction_service.train_all.assert_called_once_with(fake_stock)

    def test_returns_400_when_training_fails(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_prediction_service,
    ):
        mock_prediction_service.train_all.side_effect = ValueError(
            "Not enough historical data"
        )

        with pytest.raises(HTTPException) as exc_info:
            prediction_endpoints.train_prediction_models("RELIANCE")

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Not enough historical data"


class TestGetPrediction:
    def test_returns_prediction_without_persisting(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_prediction_service,
    ):
        expected = MagicMock()
        mock_prediction_service.predict_all.return_value = expected

        result = prediction_endpoints.get_prediction("RELIANCE")

        assert result is expected

    def test_does_not_pass_persist_flag(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_prediction_service,
        fake_stock,
    ):
        prediction_endpoints.get_prediction("RELIANCE")

        mock_prediction_service.predict_all.assert_called_once_with(fake_stock)

    def test_returns_400_on_prediction_error(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_prediction_service,
    ):
        mock_prediction_service.predict_all.side_effect = ValueError("Model not found")

        with pytest.raises(HTTPException) as exc_info:
            prediction_endpoints.get_prediction("RELIANCE")

        assert exc_info.value.status_code == 400


class TestCreatePrediction:
    def test_persists_the_prediction(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_prediction_service,
        fake_stock,
    ):
        prediction_endpoints.create_prediction("RELIANCE")

        mock_prediction_service.predict_all.assert_called_once_with(
            fake_stock,
            persist=True,
        )

    def test_returns_400_on_prediction_error(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_prediction_service,
    ):
        mock_prediction_service.predict_all.side_effect = ValueError("boom")

        with pytest.raises(HTTPException) as exc_info:
            prediction_endpoints.create_prediction("RELIANCE")

        assert exc_info.value.status_code == 400


class TestReconcilePredictions:
    def test_returns_symbol_and_reconciled_count(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_prediction_service,
        fake_stock,
    ):
        mock_prediction_service.reconcile_actuals.return_value = 7

        result = prediction_endpoints.reconcile_predictions("RELIANCE")

        assert result == {
            "symbol": fake_stock.symbol,
            "reconciled": 7,
        }


class TestGetBacktest:
    def test_defaults_to_linear_regression_when_model_name_omitted(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_prediction_service,
        fake_stock,
    ):
        prediction_endpoints.get_backtest("RELIANCE")

        mock_prediction_service.backtest.assert_called_once_with(
            fake_stock,
            model_name="LinearRegression",
        )

    def test_uses_explicitly_requested_model(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_prediction_service,
        fake_stock,
    ):
        prediction_endpoints.get_backtest(
            "RELIANCE",
            model_name="LightGBM",
        )

        mock_prediction_service.backtest.assert_called_once_with(
            fake_stock,
            model_name="LightGBM",
        )

    def test_returns_400_on_invalid_model_name(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_prediction_service,
    ):
        mock_prediction_service.backtest.side_effect = ValueError(
            "Unsupported model: Foo"
        )

        with pytest.raises(HTTPException) as exc_info:
            prediction_endpoints.get_backtest(
                "RELIANCE",
                model_name="Foo",
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Unsupported model: Foo"


class TestCompareModels:
    def test_no_models_param_passes_none_to_service(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_prediction_service,
    ):
        mock_prediction_service.compare_models.return_value = []

        prediction_endpoints.compare_models(
            "RELIANCE",
            models=None,
        )

        _, kwargs = mock_prediction_service.compare_models.call_args

        assert kwargs["model_names"] is None

    def test_comma_separated_models_are_split_into_a_list(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_prediction_service,
    ):
        mock_prediction_service.compare_models.return_value = []

        prediction_endpoints.compare_models(
            "RELIANCE",
            models="LinearRegression,LightGBM",
        )

        _, kwargs = mock_prediction_service.compare_models.call_args

        assert kwargs["model_names"] == [
            "LinearRegression",
            "LightGBM",
        ]

    def test_whitespace_around_model_names_is_stripped(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_prediction_service,
    ):
        mock_prediction_service.compare_models.return_value = []

        prediction_endpoints.compare_models(
            "RELIANCE",
            models=" LinearRegression , LightGBM ",
        )

        _, kwargs = mock_prediction_service.compare_models.call_args

        assert kwargs["model_names"] == [
            "LinearRegression",
            "LightGBM",
        ]

    def test_response_includes_symbol_and_results(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_prediction_service,
        fake_stock,
    ):
        expected_results = [
            ModelComparisonEntry(
                model_name="LinearRegression",
                mae=1.0,
                rmse=1.0,
                mape=1.0,
                directional_accuracy=1.0,
                training_time_seconds=0.1,
            )
        ]

        mock_prediction_service.compare_models.return_value = expected_results

        response = prediction_endpoints.compare_models(
            "RELIANCE",
            models=None,
        )

        assert response.symbol == fake_stock.symbol
        assert response.results == expected_results
