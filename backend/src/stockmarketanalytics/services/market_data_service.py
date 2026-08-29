from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session
from stockmarketanalytics.models.stock import Stock
from stockmarketanalytics.models.stock_price import StockPrice
from stockmarketanalytics.schemas.stock_price_validator import (
    dedupe_by_stock_id_and_date,
    validate_price_frame,
)

logger = logging.getLogger("market_data_service")


class MarketDataService:
    def __init__(self, db: Session):
        self.db = db

    def get_or_create_stock(
        self, symbol: str, company_name: str = "", exchange: str = "NSE"
    ) -> Stock:
        stock = self.db.query(Stock).filter(Stock.symbol == symbol).first()
        if stock is None:
            stock = Stock(
                symbol=symbol,
                company_name=company_name or symbol,
                exchange=exchange,
                created_at=datetime.now(UTC),
            )
            self.db.add(stock)
            self.db.commit()
            self.db.refresh(stock)
        return stock

    def fetch_from_yfinance(
        self, symbol: str, period: str = "5y", interval: str = "1d"
    ) -> pd.DataFrame:
        needs_suffix = "." not in symbol and not symbol.startswith("^")
        yf_symbol = f"{symbol}.NS" if needs_suffix else symbol
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data returned for symbol: {yf_symbol}")
        df = df.reset_index()
        df = df.rename(
            columns={
                "Date": "trading_date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
        return df[["trading_date", "open", "high", "low", "close", "volume"]]

    def import_from_csv(self, csv_path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"CSV missing required columns: {required - set(df.columns)}"
            )
        df = df.rename(
            columns={
                "Date": "trading_date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
        return df[["trading_date", "open", "high", "low", "close", "volume"]]

    def save_prices(self, stock: Stock, df: pd.DataFrame) -> int:
        records = df.to_dict(orient="records")
        valid_items, errors = validate_price_frame(stock.symbol, records)

        for error in errors:
            logger.warning("Rejected row for %s: %s", stock.symbol, error)

        valid_items = dedupe_by_stock_id_and_date(valid_items)

        inserted = 0
        for item in valid_items:
            exists = (
                self.db.query(StockPrice)
                .filter(
                    StockPrice.stock_id == stock.id,
                    StockPrice.trading_date == item.trading_date,
                )
                .first()
            )
            if exists:
                continue

            price = StockPrice(
                stock_id=stock.id,
                trading_date=item.trading_date,
                open=item.open,
                high=item.high,
                low=item.low,
                close=item.close,
                volume=item.volume,
            )
            self.db.add(price)
            inserted += 1

        self.db.commit()
        logger.info(
            "Inserted %d new price records for %s (%d rejected)",
            inserted,
            stock.symbol,
            len(errors),
        )
        return inserted

    def update_from_yfinance(self, symbol: str, period: str = "5y") -> int:
        stock = self.get_or_create_stock(symbol)
        df = self.fetch_from_yfinance(symbol, period=period)
        return self.save_prices(stock, df)

    def update_from_csv(self, symbol: str, csv_path: str | Path) -> int:
        stock = self.get_or_create_stock(symbol)
        df = self.import_from_csv(csv_path)
        return self.save_prices(stock, df)
