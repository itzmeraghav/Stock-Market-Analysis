from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression

from stockmarketanalytics.models.prediction import Prediction
from stockmarketanalytics.schemas.prediction_schemas import (
    BacktestResult,
    ModelComparisonEntry,
    MultiModelPredictionResponse,
    PredictedDirection,
    PredictionResponse,
    PredictionTrainResult,
)
from stockmarketanalytics.services.prediction_service import (
    FEATURE_COLUMNS,
    PredictionService,
    _build_sklearn_model,
)


@pytest.fixture()
def service(session, tmp_path):
    return PredictionService(session, model_dir=tmp_path / "models")


@pytest.fixture()
def trained_stock(session, service, make_stock, make_full_history):
    """A stock with 150 days of price+indicator history, enough to train."""
    stock = make_stock(symbol="INFY")
    make_full_history(stock, n=150)
    return stock


class TestBuildSklearnModel:
    @pytest.mark.parametrize(
        "model_name,expected_type",
        [
            ("LinearRegression", LinearRegression),
            ("RandomForest", RandomForestRegressor),
            ("HistGradientBoosting", HistGradientBoostingRegressor),
            ("ExtraTrees", ExtraTreesRegressor),
        ],
    )
    def test_returns_expected_estimator_type(self, model_name, expected_type):
        model = _build_sklearn_model(model_name)

        assert isinstance(model, expected_type)

    def test_lightgbm_is_supported(self):
        from lightgbm import LGBMRegressor

        model = _build_sklearn_model("LightGBM")

        assert isinstance(model, LGBMRegressor)

    def test_unsupported_model_name_raises(self):
        with pytest.raises(ValueError, match="Unsupported model_name"):
            _build_sklearn_model("ProphetXYZ")


class TestLoadFeatureFrame:
    def test_raises_when_no_indicator_data(self, service, make_stock):
        stock = make_stock(symbol="INFY")

        with pytest.raises(ValueError, match="No indicator data"):
            service.load_feature_frame(stock.id)

    def test_returns_expected_derived_columns(self, service, trained_stock):
        df = service.load_feature_frame(trained_stock.id)

        for col in FEATURE_COLUMNS:
            assert col in df.columns
        assert not df["daily_return"].isna().any()

    def test_drops_leading_row_with_nan_daily_return(
        self, service, trained_stock, session
    ):
        from stockmarketanalytics.models.stock_price import StockPrice
        from stockmarketanalytics.models.technical_indicator import TechnicalIndicator

        joined_count = (
            session.query(StockPrice)
            .join(
                TechnicalIndicator, TechnicalIndicator.stock_price_id == StockPrice.id
            )
            .filter(StockPrice.stock_id == trained_stock.id)
            .count()
        )

        df = service.load_feature_frame(trained_stock.id)

        assert len(df) == joined_count - 1


class TestBuildTrainingSet:
    def test_shifts_target_return_and_drops_last_row(self, service, trained_stock):
        df = service.load_feature_frame(trained_stock.id)

        X, y, out_df = service.build_training_set(df)

        assert list(X.columns) == FEATURE_COLUMNS
        assert len(X) == len(y) == len(out_df)
        assert len(out_df) <= len(df) - 1


class TestDirectionalAccuracy:
    def test_all_matching_signs_gives_100_percent(self, service):
        y_test = np.array([0.01, -0.02, 0.03])
        preds = np.array([0.02, -0.01, 0.001])

        acc = service._directional_accuracy(y_test, preds)

        assert acc == pytest.approx(100.0)

    def test_all_mismatched_signs_gives_zero_percent(self, service):
        y_test = np.array([0.01, -0.02, 0.03])
        preds = np.array([-0.02, 0.01, -0.001])

        acc = service._directional_accuracy(y_test, preds)

        assert acc == pytest.approx(0.0)


class TestTrain:
    def test_raises_when_insufficient_rows(
        self, service, session, make_stock, make_full_history
    ):
        stock = make_stock(symbol="TCS")
        make_full_history(stock, n=65)

        with pytest.raises(ValueError, match="Not enough historical data"):
            service.train(stock, model_name="LinearRegression")

    def test_trains_and_returns_result(self, service, trained_stock):
        result = service.train(trained_stock, model_name="LinearRegression")

        assert isinstance(result, PredictionTrainResult)
        assert result.symbol == "INFY"
        assert result.model_name == "LinearRegression"
        assert result.mae >= 0
        assert result.rmse >= 0

    def test_persists_model_file(self, service, trained_stock):
        service.train(trained_stock, model_name="LinearRegression")

        model_path = service._model_path(trained_stock.symbol, "LinearRegression")
        assert model_path.exists()

    def test_unsupported_model_name_propagates_value_error(
        self, service, trained_stock
    ):
        with pytest.raises(ValueError, match="Unsupported model_name"):
            service.train(trained_stock, model_name="NotAModel")


class TestTrainAll:
    def test_skips_failing_models_and_returns_successes(
        self, service, trained_stock, monkeypatch
    ):
        import stockmarketanalytics.services.prediction_service as ps_module

        monkeypatch.setattr(
            ps_module, "models_train", ["LinearRegression", "BrokenModel"]
        )

        results = service.train_all(trained_stock)

        assert len(results) == 1
        assert results[0].model_name == "LinearRegression"

    def test_raises_when_every_model_fails(self, service, make_stock, monkeypatch):
        import stockmarketanalytics.services.prediction_service as ps_module

        stock = make_stock(symbol="ZZZ")
        monkeypatch.setattr(ps_module, "models_train", ["BrokenModel"])

        with pytest.raises(ValueError, match="Could not train any model"):
            service.train_all(stock)


class TestPredict:
    def test_raises_when_model_not_trained(self, service, trained_stock):
        with pytest.raises(ValueError, match="No trained LinearRegression model"):
            service.predict(trained_stock, model_name="LinearRegression")

    def test_returns_prediction_response_after_training(self, service, trained_stock):
        service.train(trained_stock, model_name="LinearRegression")

        response = service.predict(trained_stock, model_name="LinearRegression")

        assert isinstance(response, PredictionResponse)
        assert response.symbol == "INFY"
        assert response.model == "LinearRegression"

    def test_persist_true_creates_prediction_row(self, service, trained_stock, session):
        service.train(trained_stock, model_name="LinearRegression")

        service.predict(trained_stock, model_name="LinearRegression", persist=True)

        assert (
            session.query(Prediction).filter_by(stock_id=trained_stock.id).count() == 1
        )

    def test_persist_false_creates_no_row(self, service, trained_stock, session):
        service.train(trained_stock, model_name="LinearRegression")

        service.predict(trained_stock, model_name="LinearRegression", persist=False)

        assert (
            session.query(Prediction).filter_by(stock_id=trained_stock.id).count() == 0
        )


class TestFinalizePrediction:
    @pytest.mark.parametrize(
        "current_price,predicted_close,expected_direction",
        [
            (100.0, 100.20, PredictedDirection.BULLISH),
            (100.0, 99.80, PredictedDirection.BEARISH),
            (100.0, 100.05, PredictedDirection.FLAT),
        ],
    )
    def test_direction_thresholds(
        self, service, trained_stock, current_price, predicted_close, expected_direction
    ):
        response = service._finalize_prediction(
            trained_stock,
            current_price=current_price,
            predicted_close=predicted_close,
            model_name="LinearRegression",
            prediction_date=date(2024, 1, 1),
            persist=False,
        )

        assert response.direction is expected_direction

    def test_confidence_is_clamped_to_half_on_huge_moves(
        self, service, trained_stock, session
    ):
        service._finalize_prediction(
            trained_stock,
            current_price=100.0,
            predicted_close=200.0,
            model_name="LinearRegression",
            prediction_date=date(2024, 1, 1),
            persist=True,
        )

        record = session.query(Prediction).filter_by(stock_id=trained_stock.id).one()
        assert record.confidence == pytest.approx(0.5)

    def test_confidence_is_clamped_to_0_99_on_tiny_moves(
        self, service, trained_stock, session
    ):
        service._finalize_prediction(
            trained_stock,
            current_price=100.0,
            predicted_close=100.0001,
            model_name="LinearRegression",
            prediction_date=date(2024, 1, 1),
            persist=True,
        )

        record = session.query(Prediction).filter_by(stock_id=trained_stock.id).one()
        assert record.confidence == pytest.approx(0.99)

    def test_persisted_record_uses_next_day_as_target_date(
        self, service, trained_stock, session
    ):
        prediction_date = date(2024, 3, 1)

        service._finalize_prediction(
            trained_stock,
            current_price=100.0,
            predicted_close=101.0,
            model_name="LinearRegression",
            prediction_date=prediction_date,
            persist=True,
        )

        record = session.query(Prediction).filter_by(stock_id=trained_stock.id).one()
        assert record.target_date == prediction_date + timedelta(days=1)
        assert record.actual_close is None


class TestPredictAll:
    def test_raises_when_no_models_trained(self, service, trained_stock, monkeypatch):
        import stockmarketanalytics.services.prediction_service as ps_module

        monkeypatch.setattr(ps_module, "models_train", ["LinearRegression"])

        with pytest.raises(ValueError, match="No trained models available"):
            service.predict_all(trained_stock)

    def test_aggregates_predictions_and_picks_best_by_rmse(
        self, service, trained_stock, monkeypatch
    ):
        import stockmarketanalytics.services.prediction_service as ps_module

        monkeypatch.setattr(ps_module, "models_train", ["LinearRegression"])
        service.train(trained_stock, model_name="LinearRegression")

        response = service.predict_all(trained_stock)

        assert isinstance(response, MultiModelPredictionResponse)
        assert response.best_model == "LinearRegression"
        assert len(response.predictions) == 1


class TestBacktest:
    def test_raises_when_insufficient_rows(
        self, service, session, make_stock, make_full_history
    ):
        stock = make_stock(symbol="TCS")
        make_full_history(stock, n=65)

        with pytest.raises(ValueError, match="Not enough historical data"):
            service.backtest(stock, model_name="LinearRegression")

    def test_returns_backtest_result_with_matching_series_lengths(
        self, service, trained_stock
    ):
        result = service.backtest(
            trained_stock, model_name="LinearRegression", test_ratio=0.2
        )

        assert isinstance(result, BacktestResult)
        assert (
            len(result.actual_series)
            == len(result.predicted_series)
            == len(result.dates)
        )
        assert result.symbol == "INFY"

    def test_smaller_test_ratio_yields_smaller_test_set(self, service, trained_stock):
        small = service.backtest(
            trained_stock, model_name="LinearRegression", test_ratio=0.1
        )
        large = service.backtest(
            trained_stock, model_name="LinearRegression", test_ratio=0.4
        )

        assert len(small.dates) < len(large.dates)


class TestCompareModels:
    def test_sorts_results_by_ascending_rmse(self, service, trained_stock):
        results = service.compare_models(
            trained_stock, model_names=["LinearRegression", "RandomForest"]
        )

        assert [r.model_name for r in results] == sorted(
            [r.model_name for r in results],
            key=lambda name: next(r.rmse for r in results if r.model_name == name),
        )
        rmses = [r.rmse for r in results]
        assert rmses == sorted(rmses)

    def test_skips_names_outside_supported_models(self, service, trained_stock, caplog):
        results = service.compare_models(
            trained_stock, model_names=["LinearRegression", "NotAModel"]
        )

        assert all(isinstance(r, ModelComparisonEntry) for r in results)
        assert all(r.model_name != "NotAModel" for r in results)

    def test_defaults_to_all_supported_models_when_none_given(
        self, service, trained_stock, monkeypatch
    ):
        import stockmarketanalytics.services.prediction_service as ps_module

        monkeypatch.setattr(ps_module, "models_train", ["LinearRegression"])

        results = service.compare_models(trained_stock)

        assert {r.model_name for r in results} == {"LinearRegression"}


class TestReconcileActuals:
    def test_returns_zero_when_nothing_pending(self, service, trained_stock):
        assert service.reconcile_actuals(trained_stock) == 0

    def test_fills_actual_close_when_matching_price_exists(
        self, service, trained_stock, session, make_stock_price
    ):
        target_date = date(2030, 1, 2)
        make_stock_price(trained_stock, trading_date=target_date, close=123.45)

        prediction = Prediction(
            stock_id=trained_stock.id,
            prediction_date=target_date - timedelta(days=1),
            target_date=target_date,
            predicted_close=120.0,
            predicted_direction="Bullish",
            model_name="LinearRegression",
        )
        session.add(prediction)
        session.commit()

        updated = service.reconcile_actuals(trained_stock)

        session.refresh(prediction)
        assert updated == 1
        assert prediction.actual_close == pytest.approx(123.45)

    def test_leaves_prediction_pending_when_no_matching_price_row(
        self, service, trained_stock, session
    ):
        prediction = Prediction(
            stock_id=trained_stock.id,
            prediction_date=date(2030, 1, 1),
            target_date=date(2030, 1, 2),
            predicted_close=120.0,
            predicted_direction="Bullish",
            model_name="LinearRegression",
        )
        session.add(prediction)
        session.commit()

        updated = service.reconcile_actuals(trained_stock)

        session.refresh(prediction)
        assert updated == 0
        assert prediction.actual_close is None
