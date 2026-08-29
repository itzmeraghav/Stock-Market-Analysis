from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from stockmarketanalytics.endpoints import stock_endpoints


@pytest.fixture
def mock_db_session(monkeypatch):
    db = MagicMock()

    db.__enter__.return_value = db
    db.__exit__.return_value = None

    monkeypatch.setattr(
        stock_endpoints,
        "get_db_session",
        MagicMock(return_value=db),
    )

    return db


@pytest.fixture
def stub_stock_lookup(monkeypatch, fake_stock):
    """
    Mock the private stock lookup helper used by the endpoints.
    """
    stub = MagicMock(return_value=fake_stock)

    monkeypatch.setattr(
        stock_endpoints,
        "_get_stock_or_404",
        stub,
    )

    return stub


@pytest.fixture
def mock_market_data_service(monkeypatch):
    instance = MagicMock()

    monkeypatch.setattr(
        stock_endpoints,
        "MarketDataService",
        MagicMock(return_value=instance),
    )

    return instance


@pytest.fixture
def mock_indicator_service(monkeypatch):
    instance = MagicMock()

    monkeypatch.setattr(
        stock_endpoints,
        "IndicatorService",
        MagicMock(return_value=instance),
    )

    return instance


@pytest.fixture
def mock_prediction_service(monkeypatch):
    instance = MagicMock()

    monkeypatch.setattr(
        stock_endpoints,
        "PredictionService",
        MagicMock(return_value=instance),
    )

    return instance


class TestRangeToDays:
    @pytest.mark.parametrize(
        "range_value,expected_days",
        [
            ("30d", 30),
            ("90d", 90),
            ("1y", 365),
            ("5y", 1825),
        ],
    )
    def test_known_ranges_map_to_expected_day_counts(
        self,
        range_value,
        expected_days,
    ):
        assert stock_endpoints._range_to_days(range_value) == expected_days

    def test_unknown_range_returns_none(self):
        assert stock_endpoints._range_to_days("all") is None

    def test_empty_string_returns_none(self):
        assert stock_endpoints._range_to_days("") is None


class TestListStocks:
    def test_returns_stocks_ordered_by_symbol(self, mock_db_session):
        expected = [MagicMock(), MagicMock()]

        query_mock = MagicMock()
        query_mock.order_by.return_value.all.return_value = expected

        mock_db_session.query.return_value = query_mock

        result = stock_endpoints.list_stocks()

        assert result == expected

        mock_db_session.query.assert_called_once()


class TestGetStock:
    def test_returns_stock_from_lookup(
        self,
        mock_db_session,
        stub_stock_lookup,
        fake_stock,
    ):
        result = stock_endpoints.get_stock("RELIANCE")

        assert result is fake_stock

        stub_stock_lookup.assert_called_once_with(
            "RELIANCE",
            mock_db_session,
        )


class TestGetStockPrices:
    @staticmethod
    def _build_query_mock(rows):
        query_mock = MagicMock()

        (
            query_mock.filter.return_value.order_by.return_value.limit.return_value.all.return_value
        ) = rows

        return query_mock

    def test_default_range_uses_90_day_limit_not_the_limit_param(
        self,
        mock_db_session,
        stub_stock_lookup,
    ):
        query_mock = self._build_query_mock([])

        mock_db_session.query.return_value = query_mock

        stock_endpoints.get_stock_prices(
            "RELIANCE",
            range="90d",
            limit=500,
        )

        query_mock.filter.return_value.order_by.return_value.limit.assert_called_once_with(
            90
        )

    def test_unrecognized_range_falls_back_to_limit_param(
        self,
        mock_db_session,
        stub_stock_lookup,
    ):
        query_mock = self._build_query_mock([])

        mock_db_session.query.return_value = query_mock

        stock_endpoints.get_stock_prices(
            "RELIANCE",
            range="all",
            limit=250,
        )

        (
            query_mock.filter.return_value.order_by.return_value.limit.assert_called_once_with(
                250
            )
        )

    def test_results_are_returned_in_chronological_order(
        self,
        mock_db_session,
        stub_stock_lookup,
    ):
        newest_first = [
            MagicMock(name="day3"),
            MagicMock(name="day2"),
            MagicMock(name="day1"),
        ]

        query_mock = self._build_query_mock(newest_first)

        mock_db_session.query.return_value = query_mock

        result = stock_endpoints.get_stock_prices("RELIANCE")

        assert result == list(reversed(newest_first))


class TestFetchComputeAndReconcile:
    def test_happy_path_returns_full_update_result(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_market_data_service,
        mock_indicator_service,
        mock_prediction_service,
        fake_stock,
    ):
        mock_market_data_service.update_from_yfinance.return_value = 120
        mock_indicator_service.persist.return_value = 45
        mock_prediction_service.reconcile_actuals.return_value = 3

        result = stock_endpoints._fetch_compute_and_reconcile(
            "reliance",
            "5y",
            mock_db_session,
        )

        assert result.symbol == "RELIANCE"
        assert result.inserted == 120
        assert result.indicators_inserted == 45
        assert result.predictions_reconciled == 3

    def test_symbol_is_uppercased_for_market_data_fetch(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_market_data_service,
        mock_indicator_service,
        mock_prediction_service,
    ):
        mock_market_data_service.update_from_yfinance.return_value = 1
        mock_indicator_service.persist.return_value = 1
        mock_prediction_service.reconcile_actuals.return_value = 0

        stock_endpoints._fetch_compute_and_reconcile(
            "reliance",
            "1y",
            mock_db_session,
        )

        mock_market_data_service.update_from_yfinance.assert_called_once_with(
            "RELIANCE",
            period="1y",
        )

    def test_raises_404_when_symbol_not_found_by_market_data_provider(
        self,
        mock_db_session,
        mock_market_data_service,
    ):
        mock_market_data_service.update_from_yfinance.side_effect = ValueError(
            "No data for symbol: FAKE"
        )

        with pytest.raises(HTTPException) as exc_info:
            stock_endpoints._fetch_compute_and_reconcile(
                "FAKE",
                "5y",
                mock_db_session,
            )

        assert exc_info.value.status_code == 404

    def test_raises_400_when_indicator_computation_fails(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_market_data_service,
        mock_indicator_service,
    ):
        mock_market_data_service.update_from_yfinance.return_value = 10

        mock_indicator_service.persist.side_effect = ValueError(
            "Not enough price history"
        )

        with pytest.raises(HTTPException) as exc_info:
            stock_endpoints._fetch_compute_and_reconcile(
                "RELIANCE",
                "5y",
                mock_db_session,
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Not enough price history"

    def test_does_not_reconcile_predictions_when_indicators_fail(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_market_data_service,
        mock_indicator_service,
        mock_prediction_service,
    ):
        mock_market_data_service.update_from_yfinance.return_value = 10

        mock_indicator_service.persist.side_effect = ValueError("boom")

        with pytest.raises(HTTPException):
            stock_endpoints._fetch_compute_and_reconcile(
                "RELIANCE",
                "5y",
                mock_db_session,
            )

        mock_prediction_service.reconcile_actuals.assert_not_called()


class TestUpdateStockData:
    def test_delegates_to_fetch_compute_and_reconcile_with_default_period(
        self,
        mock_db_session,
        monkeypatch,
    ):
        stub = MagicMock(return_value=MagicMock())

        monkeypatch.setattr(
            stock_endpoints,
            "_fetch_compute_and_reconcile",
            stub,
        )

        stock_endpoints.update_stock_data("RELIANCE")

        stub.assert_called_once_with(
            "RELIANCE",
            "5y",
            mock_db_session,
        )

    def test_respects_explicit_period(
        self,
        mock_db_session,
        monkeypatch,
    ):
        stub = MagicMock(return_value=MagicMock())

        monkeypatch.setattr(
            stock_endpoints,
            "_fetch_compute_and_reconcile",
            stub,
        )

        stock_endpoints.update_stock_data(
            "RELIANCE",
            period="1y",
        )

        stub.assert_called_once_with(
            "RELIANCE",
            "1y",
            mock_db_session,
        )


class TestImportStockCsv:
    def test_success_returns_full_update_result(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_market_data_service,
        mock_indicator_service,
        mock_prediction_service,
    ):
        mock_market_data_service.update_from_csv.return_value = 200
        mock_indicator_service.persist.return_value = 80
        mock_prediction_service.reconcile_actuals.return_value = 5

        result = stock_endpoints.import_stock_csv(
            "RELIANCE",
            "/data/reliance.csv",
        )

        assert result.inserted == 200
        assert result.indicators_inserted == 80
        assert result.predictions_reconciled == 5

    @pytest.mark.parametrize(
        "error_cls",
        [
            ValueError,
            FileNotFoundError,
        ],
    )
    def test_returns_400_for_bad_csv_or_missing_file(
        self,
        mock_db_session,
        mock_market_data_service,
        error_cls,
    ):
        mock_market_data_service.update_from_csv.side_effect = error_cls("Bad CSV")

        with pytest.raises(HTTPException) as exc_info:
            stock_endpoints.import_stock_csv(
                "RELIANCE",
                "/bad/path.csv",
            )

        assert exc_info.value.status_code == 400

    def test_symbol_is_uppercased_before_import(
        self,
        mock_db_session,
        stub_stock_lookup,
        mock_market_data_service,
        mock_indicator_service,
        mock_prediction_service,
    ):
        mock_market_data_service.update_from_csv.return_value = 1
        mock_indicator_service.persist.return_value = 0
        mock_prediction_service.reconcile_actuals.return_value = 0

        stock_endpoints.import_stock_csv(
            "reliance",
            "/data/f.csv",
        )

        mock_market_data_service.update_from_csv.assert_called_once_with(
            "RELIANCE",
            "/data/f.csv",
        )


class TestGetIndicators:
    def test_default_limit_is_90(
        self,
        mock_db_session,
        stub_stock_lookup,
    ):
        query_chain = mock_db_session.query.return_value

        (
            query_chain.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value
        ) = []

        stock_endpoints.get_indicators("RELIANCE")

        limit_mock = query_chain.join.return_value.filter.return_value.order_by.return_value.limit

        args, kwargs = limit_mock.call_args

        actual_limit = args[0]

        if hasattr(actual_limit, "default"):
            actual_limit = actual_limit.default

        assert actual_limit == 90
        assert kwargs == {}

    def test_results_are_reversed_to_chronological_order(
        self,
        mock_db_session,
        stub_stock_lookup,
    ):
        newest_first = [
            MagicMock(),
            MagicMock(),
        ]

        query_chain = mock_db_session.query.return_value

        (
            query_chain.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value
        ) = newest_first

        result = stock_endpoints.get_indicators("RELIANCE")

        assert result == list(reversed(newest_first))


class TestComputeIndicators:
    def test_delegates_to_fetch_compute_and_reconcile_with_default_period(
        self,
        mock_db_session,
        monkeypatch,
    ):
        stub = MagicMock(return_value=MagicMock())

        monkeypatch.setattr(
            stock_endpoints,
            "_fetch_compute_and_reconcile",
            stub,
        )

        stock_endpoints.compute_indicators("RELIANCE")

        stub.assert_called_once_with(
            "RELIANCE",
            "5y",
            mock_db_session,
        )
