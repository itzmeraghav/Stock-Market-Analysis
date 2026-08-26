from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from stockmarketanalytics.data.app_db_context import SessionLocal
from stockmarketanalytics.models.stock import Stock
from stockmarketanalytics.schemas.prediction_schemas import (
    BacktestResult,
    ModelComparisonResponse,
    MultiModelPredictionResponse,
    PredictionTrainResult,
)
from stockmarketanalytics.services.prediction_service import PredictionService

router = APIRouter(prefix="/predictions", tags=["predictions"])
backtest_router = APIRouter(prefix="/backtest", tags=["backtest"])


def get_db_session() -> Session:
    return SessionLocal()


def get_stock_or_404(db: Session, symbol: str) -> Stock:
    stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()

    if stock is None:
        raise HTTPException(
            status_code=404,
            detail=f"Stock '{symbol}' not found",
        )

    return stock


@router.post("/train/{symbol}", response_model=list[PredictionTrainResult])
def train_prediction_models(symbol: str):
    db = get_db_session()

    try:
        stock = get_stock_or_404(db, symbol)
        service = PredictionService(db)
        return service.train_all(stock)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    finally:
        db.close()


@router.get("/{symbol}", response_model=MultiModelPredictionResponse)
def get_prediction(symbol: str):
    db = get_db_session()

    try:
        stock = get_stock_or_404(db, symbol)
        service = PredictionService(db)
        return service.predict_all(stock)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    finally:
        db.close()


@router.post("/{symbol}", response_model=MultiModelPredictionResponse)
def create_prediction(symbol: str):
    db = get_db_session()

    try:
        stock = get_stock_or_404(db, symbol)
        service = PredictionService(db)
        return service.predict_all(stock, persist=True)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    finally:
        db.close()


@router.post("/reconcile/{symbol}")
def reconcile_predictions(symbol: str):
    db = get_db_session()

    try:
        stock = get_stock_or_404(db, symbol)
        service = PredictionService(db)

        updated = service.reconcile_actuals(stock)

        return {
            "symbol": stock.symbol,
            "reconciled": updated,
        }

    finally:
        db.close()


@backtest_router.get("/{symbol}", response_model=BacktestResult)
def get_backtest(
    symbol: str,
    model_name: str = "LinearRegression",
):
    db = get_db_session()

    try:
        stock = get_stock_or_404(db, symbol)
        service = PredictionService(db)

        return service.backtest(
            stock,
            model_name=model_name,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    finally:
        db.close()


@router.get("/compare/{symbol}", response_model=ModelComparisonResponse)
def compare_models(
    symbol: str,
    models: str | None = None,
):
    db = get_db_session()

    try:
        stock = get_stock_or_404(db, symbol)
        service = PredictionService(db)

        model_names = [model.strip() for model in models.split(",")] if models else None

        results = service.compare_models(
            stock,
            model_names=model_names,
        )

        return ModelComparisonResponse(
            symbol=stock.symbol,
            results=results,
        )

    finally:
        db.close()
