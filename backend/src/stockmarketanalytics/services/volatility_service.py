from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy.orm import Session
from stockmarketanalytics.models.stock import Stock
from stockmarketanalytics.models.stock_price import StockPrice

logger = logging.getLogger("volatility_service")

TRADING_DAYS_PER_YEAR = 252
MIN_TRADING_DAYS_REQUIRED = 30


class VolatilityService:
    def __init__(self, db: Session):
        self.db = db

    def calculate_historical_volatility(self, stock: Stock, years: int = 5) -> float:
        cutoff_date = datetime.now(UTC).date() - timedelta(days=years * 365)

        rows = (
            self.db.query(StockPrice)
            .filter(
                StockPrice.stock_id == stock.id, StockPrice.trading_date >= cutoff_date
            )
            .order_by(StockPrice.trading_date.asc())
            .all()
        )

        if not rows:
            rows = (
                self.db.query(StockPrice)
                .filter(StockPrice.stock_id == stock.id)
                .order_by(StockPrice.trading_date.asc())
                .all()
            )

        if not rows:
            raise ValueError(
                f"No historical price data found for {stock.symbol}. Fetch price data first."
            )

        if len(rows) < MIN_TRADING_DAYS_REQUIRED:
            raise ValueError(
                f"Insufficient historical data for {stock.symbol} "
                f"(need at least {MIN_TRADING_DAYS_REQUIRED} trading days, found {len(rows)})"
            )

        closes = [row.close for row in rows]

        if any(c <= 0 for c in closes):
            raise ValueError(
                f"Invalid closing prices found for {stock.symbol} (zero or negative values present)"
            )

        closes_array = np.array(closes)
        log_returns = np.log(closes_array[1:] / closes_array[:-1])

        if len(log_returns) < 2:
            raise ValueError(
                f"Not enough return observations to calculate volatility for {stock.symbol}"
            )

        daily_std = np.std(log_returns, ddof=1)
        annualized_volatility = float(daily_std * np.sqrt(TRADING_DAYS_PER_YEAR))

        if annualized_volatility <= 0 or np.isnan(annualized_volatility):
            raise ValueError(f"Calculated volatility is invalid for {stock.symbol}")

        logger.info(
            "Calculated historical volatility for %s: %.4f (%d trading days used)",
            stock.symbol,
            annualized_volatility,
            len(rows),
        )
        return annualized_volatility
