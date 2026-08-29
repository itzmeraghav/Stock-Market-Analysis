from __future__ import annotations

import os
from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from stockmarketanalytics.data.app_db_context import Base
from stockmarketanalytics.models.stock import Stock
from stockmarketanalytics.models.stock_price import StockPrice

# ---------------------------------------------------------------------------
# IMPORTANT: stockmarketanalytics.settings reads os.environ at *module import
# time* (module-level `os.getenv` calls), and app_db_context creates a real
# SQLAlchemy engine at import time using settings.database_url.
#
# These env vars MUST be set before the first `import stockmarketanalytics...`
# anywhere in the test session, so we set them here, at the top of conftest,
# which pytest always loads before collecting test modules.
# ---------------------------------------------------------------------------
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("APP_NAME", "stockmarketanalytics-test")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-not-for-prod")
os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("ACCESS_TOKEN_EXPIRE_MINUTES", "15")
os.environ.setdefault("REFRESH_TOKEN_EXPIRE_DAYS", "7")
os.environ.setdefault("LOGIN_RATE_LIMIT_PER_IP", "5/minute")
os.environ.setdefault("LOGIN_MAX_FAILED_ATTEMPTS", "5")
os.environ.setdefault("LOGIN_LOCKOUT_MINUTES", "15")
os.environ.setdefault("REFRESH_MIN_INTERVAL_SECONDS", "30")
os.environ.setdefault("TOKEN_GLOBAL_CAP_PER_HOUR", "1000/hour")
os.environ.setdefault("RATE_LIMIT_AUTHENTICATED", "100/minute")
os.environ.setdefault("RATE_LIMIT_UNAUTHENTICATED", "20/minute")


@pytest.fixture
def mock_db_session() -> MagicMock:
    """A MagicMock standing in for a SQLAlchemy Session.

    Spec'd against sqlalchemy.orm.Session so typos / wrong method names
    on the mock raise AttributeError instead of silently returning
    another MagicMock.
    """
    from sqlalchemy.orm import Session

    session = MagicMock(spec=Session)
    return session


@pytest.fixture
def make_query_mock():
    """Factory fixture: builds a chained `.query().filter().first()` mock.

    Usage:
        db.query.return_value = make_query_mock(first_return=None)
    """

    def _factory(first_return=None):
        query_mock = MagicMock()
        query_mock.filter.return_value.first.return_value = first_return
        return query_mock

    return _factory


@pytest.fixture
def anyio_backend() -> str:
    """Restrict async tests to the asyncio backend only."""
    return "asyncio"


@pytest.fixture
def fake_stock():
    """A lightweight stand-in for a Stock ORM instance.

    Endpoint helpers only ever *read* `.symbol`/`.id` off the object
    returned by the DB layer (the actual query is mocked out separately),
    so a SimpleNamespace is sufficient and avoids depending on the real
    SQLAlchemy model's constructor.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        id=1, symbol="RELIANCE", company_name="Reliance Industries", exchange="NSE"
    )


def make_exception(exc_cls, message: str = "error", **extra_attrs):
    """Build an exception instance without invoking its real __init__.

    Several custom exceptions in this project (AuthError, AccountLockedError,
    InvalidCredentialsError, ...) have constructor signatures we don't have
    visibility into from the endpoint modules alone. Using __new__ + manual
    attribute assignment lets tests construct a valid, isinstance-correct
    exception (so `except AccountLockedError` still matches) without
    guessing at __init__ arguments.
    """
    exc = exc_cls.__new__(exc_cls)
    exc.args = (message,)
    for key, value in extra_attrs.items():
        setattr(exc, key, value)
    return exc


@pytest.fixture
def exception_factory():
    return make_exception


@pytest.fixture
def app(mock_db_session):
    """A minimal FastAPI app wired with only the options router, with
    get_db overridden to return the shared `mock_db_session` mock.
    """
    from fastapi import FastAPI

    from stockmarketanalytics.data.app_db_context import get_db
    from stockmarketanalytics.endpoints import option_endpoints

    test_app = FastAPI()
    test_app.include_router(option_endpoints.router)
    test_app.dependency_overrides[get_db] = lambda: mock_db_session
    return test_app


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)


@pytest.fixture()
def engine():
    engine = create_engine("sqlite:///:memory:", future=True)

    @event.listens_for(engine, "connect")
    def _enable_fk(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture()
def session(engine) -> Session:
    factory = sessionmaker(bind=engine, future=True)
    with factory() as db:
        yield db


@pytest.fixture()
def make_stock(session):
    def _make(
        symbol: str = "INFY", company_name: str = "Infosys Ltd", exchange: str = "NSE"
    ) -> Stock:
        stock = Stock(symbol=symbol, company_name=company_name, exchange=exchange)
        session.add(stock)
        session.commit()
        session.refresh(stock)
        return stock

    return _make


@pytest.fixture()
def make_stock_price(session):
    def _make(
        stock: Stock,
        trading_date,
        open=100.0,
        high=110.0,
        low=95.0,
        close=105.0,
        volume=1000,
    ) -> StockPrice:
        price = StockPrice(
            stock_id=stock.id,
            trading_date=trading_date,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
        session.add(price)
        session.commit()
        session.refresh(price)
        return price

    return _make


@pytest.fixture()
def make_price_series(session):
    """Bulk-insert a deterministic, business-day-spaced price history.
    Returns the list of persisted StockPrice rows, ordered by trading_date.
    Closes follow a fixed pseudo-random walk (seeded) so tests are
    reproducible; open/high/low are derived to always satisfy the OHLC
    consistency rules.
    """

    def _make(stock, n: int = 120, start: date = date(2023, 1, 2), seed: int = 42):
        rng = np.random.default_rng(seed)
        steps = rng.normal(loc=0.0005, scale=0.01, size=n)
        closes = 100.0 * np.cumprod(1 + steps)
        rows = []
        current = start
        for i, close in enumerate(closes):
            # Skip weekends to mimic trading-day spacing.
            while current.weekday() >= 5:
                current += timedelta(days=1)
            open_ = closes[i - 1] if i > 0 else close
            high = max(open_, close) * 1.01
            low = min(open_, close) * 0.99
            price = StockPrice(
                stock_id=stock.id,
                trading_date=current,
                open=float(open_),
                high=float(high),
                low=float(low),
                close=float(close),
                volume=int(1_000_000 + rng.integers(0, 500_000)),
            )
            session.add(price)
            rows.append(price)
            current += timedelta(days=1)
        session.commit()
        for row in rows:
            session.refresh(row)
        return rows

    return _make


@pytest.fixture()
def make_full_history(session, make_price_series):
    """Insert a price history plus persisted technical indicators for it.
    Uses the real IndicatorService so the derived features are internally
    consistent with the code under test elsewhere in the suite.
    """

    def _make(stock, n: int = 120, start: date = date(2023, 1, 2), seed: int = 42):
        from stockmarketanalytics.services.indicator_service import IndicatorService

        prices = make_price_series(stock, n=n, start=start, seed=seed)
        IndicatorService(session).persist(stock.id)
        return prices

    return _make
