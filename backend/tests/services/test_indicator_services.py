from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from stockmarketanalytics.models.technical_indicator import TechnicalIndicator
from stockmarketanalytics.services.indicator_service import IndicatorService


@pytest.fixture()
def service(session):
    return IndicatorService(session)


class TestLoadPriceFrame:
    def test_raises_when_no_price_history(self, service):
        with pytest.raises(ValueError, match="No price history"):
            service.load_price_frame(stock_id=999)

    def test_returns_frame_ordered_by_trading_date(
        self, service, make_stock, make_price_series
    ):
        stock = make_stock()
        make_price_series(stock, n=5)

        df = service.load_price_frame(stock.id)

        assert list(df.columns) == ["id", "trading_date", "close"]
        assert df["trading_date"].is_monotonic_increasing
        assert len(df) == 5


class TestCalculateSma:
    def test_sma_matches_pandas_rolling_mean(self, service):
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        sma = service.calculate_sma(close, window=3)

        assert np.isnan(sma.iloc[0])
        assert np.isnan(sma.iloc[1])
        assert sma.iloc[2] == pytest.approx(2.0)
        assert sma.iloc[4] == pytest.approx(4.0)

    def test_window_larger_than_series_is_all_nan(self, service):
        close = pd.Series([1.0, 2.0, 3.0])

        sma = service.calculate_sma(close, window=10)

        assert sma.isna().all()


class TestCalculateEma:
    def test_ema_respects_min_periods(self, service):
        close = pd.Series([10.0, 11.0, 12.0, 13.0])

        ema = service.calculate_ema(close, span=2)

        assert np.isnan(ema.iloc[0])
        assert not np.isnan(ema.iloc[1])

    def test_ema_reacts_to_recent_prices_faster_than_sma(self, service):
        close = pd.Series([10.0] * 10 + [20.0])

        ema = service.calculate_ema(close, span=3)
        sma = service.calculate_sma(close, window=10)

        assert ema.iloc[-1] > sma.iloc[-1]


class TestCalculateRsi:
    def test_all_gains_falls_back_to_neutral_50(self, service):
        close = pd.Series([float(i) for i in range(1, 20)])

        rsi = service.calculate_rsi(close, window=14)

        assert rsi.iloc[-1] == pytest.approx(50.0)

    def test_all_losses_gives_rsi_near_zero(self, service):
        close = pd.Series([float(i) for i in range(20, 1, -1)])

        rsi = service.calculate_rsi(close, window=14)

        assert rsi.iloc[-1] == pytest.approx(0.0, abs=0.5)

    def test_flat_series_defaults_to_50(self, service):
        close = pd.Series([10.0] * 20)

        rsi = service.calculate_rsi(close, window=14)

        assert rsi.iloc[-1] == pytest.approx(50.0)

    def test_never_returns_nan(self, service):
        close = pd.Series([10.0, 10.0, 11.0, 9.0, 10.0])

        rsi = service.calculate_rsi(close, window=3)

        assert not rsi.isna().any()


class TestCalculateMacd:
    def test_macd_is_difference_of_ema12_and_ema26(self, service):
        close = pd.Series(np.linspace(100, 150, 40))

        macd = service.calculate_macd(close)
        expected = service.calculate_ema(close, 12) - service.calculate_ema(close, 26)

        pd.testing.assert_series_equal(macd, expected)


class TestCalculateBollingerBands:
    def test_upper_and_lower_bracket_the_sma(self, service):
        close = pd.Series(np.linspace(90, 110, 30))

        upper, lower = service.calculate_bollinger_bands(close, window=20, num_std=2.0)
        sma = service.calculate_sma(close, 20)

        valid = sma.notna()
        assert (upper[valid] >= sma[valid]).all()
        assert (lower[valid] <= sma[valid]).all()

    def test_zero_std_collapses_bands_to_sma(self, service):
        close = pd.Series([50.0] * 25)

        upper, lower = service.calculate_bollinger_bands(close, window=20, num_std=2.0)
        sma = service.calculate_sma(close, 20)

        valid = sma.notna()
        assert np.allclose(upper[valid], sma[valid])
        assert np.allclose(lower[valid], sma[valid])


class TestCalculateDailyReturnAndVolatility:
    def test_daily_return_first_value_is_nan(self, service):
        close = pd.Series([100.0, 101.0, 99.0])

        returns = service.calculate_daily_return(close)

        assert np.isnan(returns.iloc[0])
        assert returns.iloc[1] == pytest.approx(0.01)

    def test_volatility_is_annualized(self, service):
        close = pd.Series(np.linspace(100, 130, 40))
        returns = service.calculate_daily_return(close)

        vol = service.calculate_volatility(returns, window=20)
        manual = returns.rolling(20, min_periods=20).std() * np.sqrt(252)

        pd.testing.assert_series_equal(vol, manual)


class TestComputeAll:
    def test_adds_all_indicator_columns(self, service, make_stock, make_price_series):
        stock = make_stock()
        make_price_series(stock, n=60)

        df = service.compute_all(stock.id)

        expected_columns = {
            "sma20",
            "sma50",
            "ema20",
            "rsi14",
            "macd",
            "bollinger_upper",
            "bollinger_lower",
            "volatility",
        }
        assert expected_columns.issubset(df.columns)
        assert len(df) == 60


class TestPersist:
    def test_inserts_rows_only_once_sma50_window_is_satisfied(
        self, service, make_stock, make_price_series, session
    ):
        stock = make_stock()
        make_price_series(stock, n=60)

        inserted = service.persist(stock.id)

        assert inserted == 60 - 49
        assert session.query(TechnicalIndicator).count() == inserted

    def test_is_idempotent_on_second_call(
        self, service, make_stock, make_price_series, session
    ):
        stock = make_stock()
        make_price_series(stock, n=60)

        first = service.persist(stock.id)
        second = service.persist(stock.id)

        assert first > 0
        assert second == 0
        assert session.query(TechnicalIndicator).count() == first

    def test_new_rows_are_picked_up_on_subsequent_persist(
        self, service, make_stock, make_price_series, make_stock_price, session
    ):
        stock = make_stock()
        make_price_series(stock, n=60)
        service.persist(stock.id)

        make_stock_price(stock, trading_date=date(2023, 6, 1), close=123.4)
        second = service.persist(stock.id)

        assert second == 1

    def test_volatility_defaults_to_zero_when_nan(
        self, service, make_stock, make_price_series, session
    ):
        stock = make_stock()
        make_price_series(stock, n=50)

        service.persist(stock.id)

        rows = session.query(TechnicalIndicator).all()
        assert all(row.volatility is not None for row in rows)
