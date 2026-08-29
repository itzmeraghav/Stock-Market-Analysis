from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from stockmarketanalytics.data.app_db_context import get_db
from stockmarketanalytics.models.stock import Stock
from stockmarketanalytics.schemas.option_schemas import (
    OptionCalculationOut,
    OptionCalculationRequest,
    OptionCalculationResponse,
    OptionForecastRequest,
    OptionForecastResponse,
)
from stockmarketanalytics.services.black_scholes_service import BlackScholesService
from stockmarketanalytics.services.prediction_service import PredictionService
from stockmarketanalytics.services.volatility_service import VolatilityService

router = APIRouter(prefix="/options", tags=["options"])

DB_DEPENDENCY = Depends(get_db)


def _get_stock_or_404(symbol: str, db: Session) -> Stock:
    stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()
    if stock is None:
        raise HTTPException(
            status_code=404, detail=f"Stock not found: {symbol.upper()}"
        )
    return stock


@router.post("/calculate", response_model=OptionCalculationResponse)
def calculate_option(request: OptionCalculationRequest, db: Session = DB_DEPENDENCY):
    stock = _get_stock_or_404(request.symbol, db)

    service = BlackScholesService(db)
    return service.calculate(
        stock=stock,
        spot_price=request.spot_price,
        strike_price=request.strike_price,
        days_to_expiry=request.days_to_expiry,
        risk_free_rate=request.risk_free_rate,
        volatility=request.volatility,
    )


@router.post("/forecast/{symbol}", response_model=OptionForecastResponse)
def forecast_option(
    symbol: str, request: OptionForecastRequest, db: Session = DB_DEPENDENCY
):
    stock = _get_stock_or_404(symbol, db)

    prediction_service = PredictionService(db)
    try:
        prediction = prediction_service.predict(stock, model_name=request.model_name)
    except ValueError as exc:
        # raise HTTPException(status_code=400, detail=str(exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    volatility_service = VolatilityService(db)
    try:
        volatility = volatility_service.calculate_historical_volatility(stock, years=5)
    except ValueError as exc:
        # raise HTTPException(status_code=400, detail=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    black_scholes_service = BlackScholesService(db)
    try:
        option_result = black_scholes_service.calculate(
            stock=stock,
            spot_price=prediction.current_price,
            strike_price=request.strike_price,
            days_to_expiry=request.days_to_expiry,
            risk_free_rate=request.risk_free_rate,
            volatility=volatility,
        )
    except ValueError as exc:
        # raise HTTPException(status_code=400, detail=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return OptionForecastResponse(
        symbol=stock.symbol,
        current_price=prediction.current_price,
        predicted_price=prediction.predicted_price,
        expected_change_percent=prediction.expected_change_percent,
        direction=prediction.direction,
        model=prediction.model,
        historical_volatility=round(volatility, 4),
        historical_volatility_percent=round(volatility * 100, 2),
        call_price=option_result.call_price,
        put_price=option_result.put_price,
        call_delta=option_result.call_delta,
        put_delta=option_result.put_delta,
        gamma=option_result.gamma,
        vega=option_result.vega,
        theta=option_result.theta,
        rho=option_result.rho,
    )


@router.get("/{symbol}", response_model=list[OptionCalculationOut])
def get_option_history(symbol: str, db: Session = DB_DEPENDENCY):
    stock = _get_stock_or_404(symbol, db)

    service = BlackScholesService(db)
    return service.get_history(stock)
