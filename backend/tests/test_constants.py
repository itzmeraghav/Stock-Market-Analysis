from __future__ import annotations

from stockmarketanalytics.constants import SUPPORTED_MODELS


class TestSupportedModels:
    """Tests for the SUPPORTED_MODELS constant."""

    def test_supported_models_is_a_list(self):
        assert isinstance(SUPPORTED_MODELS, list)

    def test_supported_models_is_not_empty(self):
        assert len(SUPPORTED_MODELS) > 0

    def test_supported_models_contains_only_strings(self):
        assert all(isinstance(model, str) for model in SUPPORTED_MODELS)

    def test_supported_models_has_no_duplicates(self):
        unique_models = set(SUPPORTED_MODELS)

        assert len(unique_models) == len(SUPPORTED_MODELS)

    def test_supported_models_contains_expected_models(self):
        expected_models = {
            "LinearRegression",
            "RandomForest",
            "HistGradientBoosting",
            "LightGBM",
            "ExtraTrees",
        }

        assert expected_models.issubset(set(SUPPORTED_MODELS))

    def test_supported_models_entries_are_non_empty_strings(self):
        assert all(model.strip() != "" for model in SUPPORTED_MODELS)
