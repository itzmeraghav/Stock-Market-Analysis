from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest
from stockmarketanalytics.models.stock_price import StockPrice
from stockmarketanalytics.services.volatility_service import (
    MIN_TRADING_DAYS_REQUIRED,
    TRADING_DAYS_PER_YEAR,
    VolatilityService,
)


@pytest.fixture()
def service(session):
    return VolatilityService(session)


def _insert_prices(session, stock, closes: list[float], start: date):
    current = start
    for close in closes:
        session.add(
            StockPrice(
                stock_id=stock.id,
                trading_date=current,
                open=close,
                high=close * 1.01,
                low=close * 0.99,
                close=close,
                volume=1_000_000,
            )
        )
        current += timedelta(days=1)
    session.commit()


class TestCalculateHistoricalVolatility:
    def test_raises_when_no_price_data_at_all(self, service, make_stock):
        stock = make_stock()

        with pytest.raises(ValueError, match="No historical price data"):
            service.calculate_historical_volatility(stock)

    def test_raises_when_fewer_rows_than_minimum_required(
        self, session, service, make_stock
    ):
        stock = make_stock()
        closes = [100.0 + i for i in range(MIN_TRADING_DAYS_REQUIRED - 1)]
        _insert_prices(
            session,
            stock,
            closes,
            start=datetime.now(UTC).date() - timedelta(days=len(closes)),
        )

        with pytest.raises(ValueError, match="Insufficient historical data"):
            service.calculate_historical_volatility(stock)

    def test_raises_when_a_closing_price_is_zero_or_negative(
        self, session, service, make_stock
    ):
        stock = make_stock()
        closes = [100.0 + i for i in range(MIN_TRADING_DAYS_REQUIRED + 5)]
        closes[3] = 0.0
        _insert_prices(
            session,
            stock,
            closes,
            start=datetime.now(UTC).date() - timedelta(days=len(closes)),
        )

        with pytest.raises(ValueError, match="Invalid closing prices"):
            service.calculate_historical_volatility(stock)

    def test_returns_positive_float_for_valid_history(
        self, session, service, make_stock
    ):
        stock = make_stock()
        rng = np.random.default_rng(7)
        closes = list(100.0 * np.cumprod(1 + rng.normal(0.0, 0.01, 60)))
        _insert_prices(
            session,
            stock,
            closes,
            start=datetime.now(UTC).date() - timedelta(days=len(closes)),
        )

        volatility = service.calculate_historical_volatility(stock)

        assert isinstance(volatility, float)
        assert volatility > 0

    def test_matches_manual_annualized_std_of_log_returns(
        self, session, service, make_stock
    ):
        stock = make_stock()
        rng = np.random.default_rng(11)
        closes = list(100.0 * np.cumprod(1 + rng.normal(0.0, 0.01, 60)))
        _insert_prices(
            session,
            stock,
            closes,
            start=datetime.now(UTC).date() - timedelta(days=len(closes)),
        )

        volatility = service.calculate_historical_volatility(stock)

        closes_array = np.array(closes)
        log_returns = np.log(closes_array[1:] / closes_array[:-1])
        expected = float(np.std(log_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

        assert volatility == pytest.approx(expected)

    def test_falls_back_to_full_history_when_nothing_within_cutoff(
        self, session, service, make_stock
    ):
        stock = make_stock()
        rng = np.random.default_rng(3)
        closes = list(100.0 * np.cumprod(1 + rng.normal(0.0, 0.01, 40)))
        old_start = datetime.now(UTC).date() - timedelta(days=20 * 365)
        _insert_prices(session, stock, closes, start=old_start)

        volatility = service.calculate_historical_volatility(stock, years=1)

        assert volatility > 0

    def test_uses_only_rows_within_cutoff_when_recent_data_exists(
        self, session, service, make_stock
    ):
        stock = make_stock()
        old_closes = [100.0, 200.0, 50.0, 300.0, 20.0] * 10
        _insert_prices(
            session,
            stock,
            old_closes,
            start=datetime.now(UTC).date() - timedelta(days=20 * 365),
        )

        rng = np.random.default_rng(5)
        recent_closes = list(100.0 * np.cumprod(1 + rng.normal(0.0, 0.001, 40)))
        _insert_prices(
            session,
            stock,
            recent_closes,
            start=datetime.now(UTC).date() - timedelta(days=39),
        )

        volatility = service.calculate_historical_volatility(stock, years=1)

        closes_array = np.array(recent_closes)
        log_returns = np.log(closes_array[1:] / closes_array[:-1])
        expected = float(np.std(log_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

        assert volatility == pytest.approx(expected)
