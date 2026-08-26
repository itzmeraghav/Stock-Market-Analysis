from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.orm import Session
from stockmarketanalytics.data.app_db_context import get_db
from stockmarketanalytics.models.stock import Stock
from stockmarketanalytics.models.stock_price import StockPrice
from stockmarketanalytics.models.technical_indicator import TechnicalIndicator
from stockmarketanalytics.schemas.stock_schemas import (
    StockOut,
    StockPriceOut,
    StockUpdateResult,
    TechnicalIndicatorOut,
)
from stockmarketanalytics.services.indicator_service import IndicatorService
from stockmarketanalytics.services.market_data_service import MarketDataService
from stockmarketanalytics.services.prediction_service import PredictionService

router = APIRouter(prefix="/stocks", tags=["stocks"])
indicator_router = APIRouter(prefix="/indicators", tags=["indicators"])


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    db_generator = get_db()
    db = next(db_generator)

    try:
        yield db
    finally:
        db_generator.close()


def _get_stock_or_404(symbol: str, db: Session) -> Stock:
    stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()

    if stock is None:
        raise HTTPException(
            status_code=404,
            detail=f"Stock not found: {symbol.upper()}",
        )

    return stock


def _range_to_days(range_: str) -> int | None:
    mapping = {
        "30d": 30,
        "90d": 90,
        "1y": 365,
        "5y": 365 * 5,
    }

    return mapping.get(range_)


@router.get("", response_model=list[StockOut])
def list_stocks():
    with get_db_session() as db:
        return db.query(Stock).order_by(Stock.symbol.asc()).all()


@router.get("/{symbol}", response_model=StockOut)
def get_stock(symbol: str):
    with get_db_session() as db:
        return _get_stock_or_404(symbol, db)


@router.get("/{symbol}/prices", response_model=list[StockPriceOut])
def get_stock_prices(
    symbol: str,
    range: str = Query(
        default="90d",
        description="30d, 90d, 1y, 5y, or 'all'",
    ),
    limit: int = Query(
        default=500,
        le=5000,
    ),
):
    with get_db_session() as db:
        stock = _get_stock_or_404(symbol, db)

        query = (
            db.query(StockPrice)
            .filter(StockPrice.stock_id == stock.id)
            .order_by(StockPrice.trading_date.desc())
        )

        days = _range_to_days(range)

        if days is not None:
            query = query.limit(days)
        else:
            query = query.limit(limit)

        rows = query.all()

        return list(reversed(rows))


def _fetch_compute_and_reconcile(
    symbol: str,
    period: str,
    db: Session,
) -> StockUpdateResult:
    market_data_service = MarketDataService(db)

    try:
        inserted = market_data_service.update_from_yfinance(
            symbol.upper(),
            period=period,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    stock = _get_stock_or_404(symbol, db)

    indicators_inserted = 0

    try:
        indicators_inserted = IndicatorService(db).persist(stock.id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    predictions_reconciled = PredictionService(db).reconcile_actuals(stock)

    return StockUpdateResult(
        symbol=symbol.upper(),
        inserted=inserted,
        indicators_inserted=indicators_inserted,
        predictions_reconciled=predictions_reconciled,
    )


@router.post("/{symbol}/update", response_model=StockUpdateResult)
def update_stock_data(
    symbol: str,
    period: str = "5y",
):
    with get_db_session() as db:
        return _fetch_compute_and_reconcile(
            symbol,
            period,
            db,
        )


@router.post("/{symbol}/import-csv", response_model=StockUpdateResult)
def import_stock_csv(
    symbol: str,
    csv_path: str,
):
    with get_db_session() as db:
        market_data_service = MarketDataService(db)

        try:
            inserted = market_data_service.update_from_csv(
                symbol.upper(),
                csv_path,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            )

        stock = _get_stock_or_404(symbol, db)

        indicators_inserted = 0

        try:
            indicators_inserted = IndicatorService(db).persist(stock.id)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            )

        predictions_reconciled = PredictionService(db).reconcile_actuals(stock)

        return StockUpdateResult(
            symbol=symbol.upper(),
            inserted=inserted,
            indicators_inserted=indicators_inserted,
            predictions_reconciled=predictions_reconciled,
        )


@indicator_router.get(
    "/{symbol}",
    response_model=list[TechnicalIndicatorOut],
)
def get_indicators(
    symbol: str,
    limit: int = Query(
        default=90,
        le=2000,
    ),
):
    with get_db_session() as db:
        stock = _get_stock_or_404(symbol, db)

        rows = (
            db.query(TechnicalIndicator)
            .join(
                StockPrice,
                TechnicalIndicator.stock_price_id == StockPrice.id,
            )
            .filter(StockPrice.stock_id == stock.id)
            .order_by(StockPrice.trading_date.desc())
            .limit(limit)
            .all()
        )

        return list(reversed(rows))


@indicator_router.post(
    "/{symbol}/compute",
    response_model=StockUpdateResult,
)
def compute_indicators(
    symbol: str,
    period: str = "5y",
):
    with get_db_session() as db:
        return _fetch_compute_and_reconcile(
            symbol,
            period,
            db,
        )
