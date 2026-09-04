from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from stockmarketanalytics.data.app_db_context import get_db
from stockmarketanalytics.models.stock import Stock
from stockmarketanalytics.schemas.trading_schemas import (
    TradingAnalysisRequest,
    TradingAnalysisResponse,
)
from stockmarketanalytics.services.risk_management_service import RiskManagementService

router = APIRouter(prefix="/trading", tags=["trading"])

DB_DEPENDENCY = Depends(get_db)


def _get_stock_or_404(symbol: str, db: Session) -> Stock:
    stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()
    if stock is None:
        raise HTTPException(
            status_code=404, detail=f"Stock not found: {symbol.upper()}"
        )
    return stock


@router.post("/analyze/{symbol}", response_model=TradingAnalysisResponse)
def analyze_trade(
    symbol: str, request: TradingAnalysisRequest, db: Session = DB_DEPENDENCY
):
    stock = _get_stock_or_404(symbol, db)

    service = RiskManagementService(db)
    try:
        return service.analyze_trade(
            stock=stock,
            investment_amount=request.investment_amount,
            risk_percentage=request.risk_percentage,
            target_percentage=request.target_percentage,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
