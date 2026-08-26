from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from stockmarketanalytics.models.stock_price import StockPrice
from stockmarketanalytics.models.technical_indicator import TechnicalIndicator

logger = logging.getLogger("indicator_service")


class IndicatorService:
    def __init__(self, db: Session):
        self.db = db

    def load_price_frame(self, stock_id: int) -> pd.DataFrame:
        rows = (
            self.db.query(StockPrice)
            .filter(StockPrice.stock_id == stock_id)
            .order_by(StockPrice.trading_date.asc())
            .all()
        )
        if not rows:
            raise ValueError(f"No price history found for stock_id={stock_id}")

        df = pd.DataFrame(
            [
                {
                    "id": r.id,
                    "trading_date": r.trading_date,
                    "close": r.close,
                }
                for r in rows
            ]
        )
        return df

    def calculate_sma(self, close: pd.Series, window: int) -> pd.Series:
        return close.rolling(window=window, min_periods=window).mean()

    def calculate_ema(self, close: pd.Series, span: int) -> pd.Series:
        return close.ewm(span=span, adjust=False, min_periods=span).mean()

    def calculate_rsi(self, close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def calculate_macd(self, close: pd.Series) -> pd.Series:
        ema12 = self.calculate_ema(close, 12)
        ema26 = self.calculate_ema(close, 26)
        return ema12 - ema26

    def calculate_bollinger_bands(
        self, close: pd.Series, window: int = 20, num_std: float = 2.0
    ):
        sma = self.calculate_sma(close, window)
        std = close.rolling(window=window, min_periods=window).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        return upper, lower

    def calculate_daily_return(self, close: pd.Series) -> pd.Series:
        return close.pct_change()

    def calculate_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        return returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)

    def compute_all(self, stock_id: int) -> pd.DataFrame:
        df = self.load_price_frame(stock_id)
        close = df["close"]

        returns = self.calculate_daily_return(close)

        df["sma20"] = self.calculate_sma(close, 20)
        df["sma50"] = self.calculate_sma(close, 50)
        df["ema20"] = self.calculate_ema(close, 20)
        df["rsi14"] = self.calculate_rsi(close, 14)
        df["macd"] = self.calculate_macd(close)
        df["bollinger_upper"], df["bollinger_lower"] = self.calculate_bollinger_bands(
            close, 20
        )
        df["volatility"] = self.calculate_volatility(returns, 20)

        return df

    def persist(self, stock_id: int) -> int:
        df = self.compute_all(stock_id)

        existing_ids = {
            row.stock_price_id
            for row in self.db.query(TechnicalIndicator.stock_price_id)
            .join(StockPrice, TechnicalIndicator.stock_price_id == StockPrice.id)
            .filter(StockPrice.stock_id == stock_id)
            .all()
        }

        inserted = 0
        for _, row in df.iterrows():
            if row["id"] in existing_ids:
                continue
            if pd.isna(row["sma50"]):
                continue

            indicator = TechnicalIndicator(
                stock_price_id=int(row["id"]),
                sma20=float(row["sma20"]),
                sma50=float(row["sma50"]),
                ema20=float(row["ema20"]),
                rsi14=float(row["rsi14"]),
                macd=float(row["macd"]),
                bollinger_upper=float(row["bollinger_upper"]),
                bollinger_lower=float(row["bollinger_lower"]),
                volatility=(
                    float(row["volatility"]) if not pd.isna(row["volatility"]) else 0.0
                ),
            )
            self.db.add(indicator)
            inserted += 1

        self.db.commit()
        logger.info("Inserted %d indicator rows for stock_id=%s", inserted, stock_id)
        return inserted
