from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from stockmarketanalytics.models.stock import Stock
from stockmarketanalytics.models.stock_price import StockPrice
from stockmarketanalytics.services.market_data_service import MarketDataService


@pytest.fixture()
def service(session):
    return MarketDataService(session)


def _yf_history_df(n: int = 5, start: str = "2024-01-02") -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=n, freq="B")
    dates.name = "Date"
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [110.0 + i for i in range(n)],
            "Low": [95.0 + i for i in range(n)],
            "Close": [105.0 + i for i in range(n)],
            "Volume": [1_000_000 + i for i in range(n)],
        },
        index=dates,
    )


class TestGetOrCreateStock:
    def test_creates_new_stock_when_absent(self, service, session):
        stock = service.get_or_create_stock("INFY", company_name="Infosys Ltd")

        assert stock.id is not None
        assert session.query(Stock).filter_by(symbol="INFY").count() == 1

    def test_returns_existing_stock_without_duplicating(
        self, service, session, make_stock
    ):
        make_stock(symbol="INFY", company_name="Infosys Ltd")

        stock = service.get_or_create_stock("INFY")

        assert session.query(Stock).filter_by(symbol="INFY").count() == 1
        assert stock.symbol == "INFY"

    def test_company_name_defaults_to_symbol_when_blank(self, service):
        stock = service.get_or_create_stock("TCS")

        assert stock.company_name == "TCS"


class TestFetchFromYfinance:
    def test_appends_ns_suffix_for_bare_nse_symbols(self, service):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _yf_history_df()

        with patch(
            "stockmarketanalytics.services.market_data_service.yf.Ticker",
            return_value=mock_ticker,
        ) as ticker_cls:
            service.fetch_from_yfinance("INFY")

        ticker_cls.assert_called_once_with("INFY.NS")

    def test_does_not_suffix_symbols_that_already_have_a_dot(self, service):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _yf_history_df()

        with patch(
            "stockmarketanalytics.services.market_data_service.yf.Ticker",
            return_value=mock_ticker,
        ) as ticker_cls:
            service.fetch_from_yfinance("BRK.B")

        ticker_cls.assert_called_once_with("BRK.B")

    def test_does_not_suffix_index_symbols_starting_with_caret(self, service):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _yf_history_df()

        with patch(
            "stockmarketanalytics.services.market_data_service.yf.Ticker",
            return_value=mock_ticker,
        ) as ticker_cls:
            service.fetch_from_yfinance("^NSEI")

        ticker_cls.assert_called_once_with("^NSEI")

    def test_raises_when_no_data_returned(self, service):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with (
            patch(
                "stockmarketanalytics.services.market_data_service.yf.Ticker",
                return_value=mock_ticker,
            ),
            pytest.raises(ValueError, match="No data returned"),
        ):
            service.fetch_from_yfinance("INFY")

    def test_renames_and_selects_expected_columns(self, service):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _yf_history_df(n=3)

        with patch(
            "stockmarketanalytics.services.market_data_service.yf.Ticker",
            return_value=mock_ticker,
        ):
            df = service.fetch_from_yfinance("INFY")

        assert list(df.columns) == [
            "trading_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert len(df) == 3


class TestImportFromCsv:
    def test_reads_and_renames_valid_csv(self, service, tmp_path):
        csv_path = tmp_path / "prices.csv"
        pd.DataFrame(
            {
                "Date": ["2024-01-02", "2024-01-03"],
                "Open": [100.0, 101.0],
                "High": [110.0, 111.0],
                "Low": [95.0, 96.0],
                "Close": [105.0, 106.0],
                "Volume": [1000, 1100],
            }
        ).to_csv(csv_path, index=False)

        df = service.import_from_csv(csv_path)

        assert list(df.columns) == [
            "trading_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert df.iloc[0]["trading_date"] == date(2024, 1, 2)

    def test_raises_when_required_columns_missing(self, service, tmp_path):
        csv_path = tmp_path / "bad.csv"
        pd.DataFrame({"Date": ["2024-01-02"], "Open": [100.0]}).to_csv(
            csv_path, index=False
        )

        with pytest.raises(ValueError, match="missing required columns"):
            service.import_from_csv(csv_path)


class TestSavePrices:
    def test_inserts_valid_rows_and_skips_invalid_ones(
        self, service, session, make_stock
    ):
        stock = make_stock(symbol="INFY")
        df = pd.DataFrame(
            [
                {
                    "trading_date": date(2024, 1, 2),
                    "open": 100.0,
                    "high": 110.0,
                    "low": 95.0,
                    "close": 105.0,
                    "volume": 1000,
                },
                {
                    "trading_date": date(2024, 1, 3),
                    "open": 100.0,
                    "high": 90.0,
                    "low": 95.0,
                    "close": 92.0,
                    "volume": 1000,
                },
            ]
        )

        inserted = service.save_prices(stock, df)

        assert inserted == 1
        assert session.query(StockPrice).filter_by(stock_id=stock.id).count() == 1

    def test_skips_rows_that_already_exist_for_the_same_date(
        self, service, session, make_stock, make_stock_price
    ):
        stock = make_stock(symbol="INFY")
        make_stock_price(stock, trading_date=date(2024, 1, 2))

        df = pd.DataFrame(
            [
                {
                    "trading_date": date(2024, 1, 2),
                    "open": 100.0,
                    "high": 110.0,
                    "low": 95.0,
                    "close": 105.0,
                    "volume": 1000,
                }
            ]
        )

        inserted = service.save_prices(stock, df)

        assert inserted == 0
        assert session.query(StockPrice).filter_by(stock_id=stock.id).count() == 1

    def test_dedupes_repeated_rows_for_same_date_before_inserting(
        self, service, session, make_stock
    ):
        stock = make_stock(symbol="INFY")
        df = pd.DataFrame(
            [
                {
                    "trading_date": date(2024, 1, 2),
                    "open": 100.0,
                    "high": 110.0,
                    "low": 95.0,
                    "close": 105.0,
                    "volume": 1000,
                },
                {
                    "trading_date": date(2024, 1, 2),
                    "open": 101.0,
                    "high": 111.0,
                    "low": 96.0,
                    "close": 106.0,
                    "volume": 1100,
                },
            ]
        )

        inserted = service.save_prices(stock, df)

        assert inserted == 1
        assert session.query(StockPrice).filter_by(stock_id=stock.id).count() == 1


class TestUpdateFromYfinance:
    def test_creates_stock_and_saves_fetched_prices(self, service, session):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _yf_history_df(n=3)

        with patch(
            "stockmarketanalytics.services.market_data_service.yf.Ticker",
            return_value=mock_ticker,
        ):
            inserted = service.update_from_yfinance("INFY")

        assert inserted == 3
        stock = session.query(Stock).filter_by(symbol="INFY").one()
        assert session.query(StockPrice).filter_by(stock_id=stock.id).count() == 3


class TestUpdateFromCsv:
    def test_creates_stock_and_saves_prices_from_csv(self, service, session, tmp_path):
        csv_path = tmp_path / "prices.csv"
        pd.DataFrame(
            {
                "Date": ["2024-01-02", "2024-01-03"],
                "Open": [100.0, 101.0],
                "High": [110.0, 111.0],
                "Low": [95.0, 96.0],
                "Close": [105.0, 106.0],
                "Volume": [1000, 1100],
            }
        ).to_csv(csv_path, index=False)

        inserted = service.update_from_csv("TCS", csv_path)

        assert inserted == 2
        stock = session.query(Stock).filter_by(symbol="TCS").one()
        assert session.query(StockPrice).filter_by(stock_id=stock.id).count() == 2
