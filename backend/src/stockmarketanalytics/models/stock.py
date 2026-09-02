from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from stockmarketanalytics.data.app_db_context import Base


class Stock(Base):
    __tablename__ = "stocks"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    symbol: Mapped[str] = mapped_column(
        String(20),
        unique=True,
        nullable=False,
        index=True,
    )

    company_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    exchange: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="NSE",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    prices = relationship(
        "StockPrice", back_populates="stock", cascade="all, delete-orphan"
    )

    predictions = relationship(
        "Prediction", back_populates="stock", cascade="all, delete-orphan"
    )

    option_calculations = relationship(
        "OptionCalculation", back_populates="stock", cascade="all, delete-orphan"
    )

    trading_analyses = relationship(
        "TradingAnalysis", back_populates="stock", cascade="all, delete-orphan"
    )
