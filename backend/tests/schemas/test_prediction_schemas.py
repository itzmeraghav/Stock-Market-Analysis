from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from stockmarketanalytics.schemas.prediction_schemas import (
    BacktestResult,
    ModelComparisonEntry,
    ModelComparisonResponse,
    MultiModelPredictionResponse,
    PredictedDirection,
    PredictionOut,
    PredictionResponse,
    PredictionTrainResult,
)


class TestPredictedDirection:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Bullish", PredictedDirection.BULLISH),
            ("Bearish", PredictedDirection.BEARISH),
            ("Flat", PredictedDirection.FLAT),
        ],
    )
    def test_accepts_known_values(self, raw, expected):
        assert PredictedDirection(raw) is expected

    def test_rejects_unknown_value(self):
        with pytest.raises(ValueError):
            PredictedDirection("Sideways")


class TestPredictionOut:
    def _payload(self, **overrides):
        payload = {
            "id": 1,
            "stock_id": 1,
            "prediction_date": date(2024, 1, 1),
            "target_date": date(2024, 1, 2),
            "actual_close": None,
            "predicted_close": 150.5,
            "predicted_direction": "Bullish",
            "confidence": 0.8,
            "model_name": "lstm-v1",
            "created_at": datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
        }
        payload.update(overrides)
        return payload

    def test_valid_payload_builds(self):
        out = PredictionOut(**self._payload())

        assert out.predicted_direction is PredictedDirection.BULLISH
        assert out.confidence == 0.8

    @pytest.mark.parametrize("confidence", [-0.01, 1.01])
    def test_confidence_out_of_bounds_raises(self, confidence):
        with pytest.raises(ValidationError):
            PredictionOut(**self._payload(confidence=confidence))

    @pytest.mark.parametrize("confidence", [0.0, 1.0])
    def test_confidence_boundary_values_are_valid(self, confidence):
        out = PredictionOut(**self._payload(confidence=confidence))

        assert out.confidence == confidence

    def test_invalid_direction_raises(self):
        with pytest.raises(ValidationError):
            PredictionOut(**self._payload(predicted_direction="Sideways"))

    def test_actual_close_defaults_to_none(self):
        payload = self._payload()
        del payload["actual_close"]

        out = PredictionOut(**payload)

        assert out.actual_close is None

    def test_orm_mode_reads_attributes(self):
        class FakePrediction:
            id = 1
            stock_id = 1
            prediction_date = date(2024, 1, 1)
            target_date = date(2024, 1, 2)
            actual_close = 151.0
            predicted_close = 150.5
            predicted_direction = "Bullish"
            confidence = 0.8
            model_name = "lstm-v1"
            created_at = datetime(2024, 1, 1, 9, 0, tzinfo=UTC)

        out = PredictionOut.model_validate(FakePrediction())

        assert out.actual_close == 151.0


class TestPredictionTrainResult:
    def test_valid_payload(self):
        result = PredictionTrainResult(
            symbol="INFY",
            model_name="lstm-v1",
            mae=1.2,
            rmse=1.5,
            mape=2.1,
            directional_accuracy=0.62,
            trained_at=datetime(2024, 1, 1, tzinfo=UTC),
        )

        assert result.model_name == "lstm-v1"

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            PredictionTrainResult(
                symbol="INFY",
                model_name="lstm-v1",
                mae=1.2,
                rmse=1.5,
                mape=2.1,
                trained_at=datetime(2024, 1, 1, tzinfo=UTC),
            )


class TestPredictionResponse:
    def test_valid_payload(self):
        response = PredictionResponse(
            symbol="INFY",
            current_price=150.0,
            predicted_price=155.0,
            expected_change_percent=3.33,
            direction="Bullish",
            model="lstm-v1",
        )

        assert response.direction is PredictedDirection.BULLISH


class TestBacktestResult:
    def test_series_lengths_can_differ_but_values_are_typed(self):
        result = BacktestResult(
            symbol="INFY",
            model_name="lstm-v1",
            mae=1.0,
            rmse=1.5,
            mape=2.0,
            directional_accuracy=0.6,
            actual_series=[100.0, 101.0, 102.0],
            predicted_series=[99.5, 101.2, 101.8],
            dates=[date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
        )

        assert len(result.actual_series) == len(result.dates)

    def test_non_numeric_series_item_raises(self):
        with pytest.raises(ValidationError):
            BacktestResult(
                symbol="INFY",
                model_name="lstm-v1",
                mae=1.0,
                rmse=1.5,
                mape=2.0,
                directional_accuracy=0.6,
                actual_series=["bad"],
                predicted_series=[99.5],
                dates=[date(2024, 1, 1)],
            )


class TestModelComparison:
    def test_response_holds_multiple_entries(self):
        entries = [
            ModelComparisonEntry(
                model_name="lstm-v1",
                mae=1.0,
                rmse=1.5,
                mape=2.0,
                directional_accuracy=0.6,
                training_time_seconds=12.3,
            ),
            ModelComparisonEntry(
                model_name="xgboost-v1",
                mae=0.9,
                rmse=1.3,
                mape=1.8,
                directional_accuracy=0.65,
                training_time_seconds=4.1,
            ),
        ]

        response = ModelComparisonResponse(symbol="INFY", results=entries)

        assert len(response.results) == 2
        assert response.results[1].model_name == "xgboost-v1"

    def test_empty_results_list_is_valid(self):
        response = ModelComparisonResponse(symbol="INFY", results=[])

        assert response.results == []


class TestMultiModelPredictionResponse:
    def test_best_model_rmse_defaults_to_none(self):
        response = MultiModelPredictionResponse(
            symbol="INFY",
            predictions=[],
            best_model="lstm-v1",
        )

        assert response.best_model_rmse is None

    def test_predictions_list_is_validated(self):
        with pytest.raises(ValidationError):
            MultiModelPredictionResponse(
                symbol="INFY",
                predictions=[{"symbol": "INFY"}],
                best_model="lstm-v1",
            )
