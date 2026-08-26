from __future__ import annotations

from datetime import date

from sqlalchemy import Date, Float, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from stockmarketanalytics.data.app_db_context import Base


class StockPrice(Base):
    __tablename__ = "stock_prices"

    __table_args__ = (
        UniqueConstraint(
            "stock_id",
            "trading_date",
            name="uq_stock_price_stock_date",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    stock_id: Mapped[int] = mapped_column(
        ForeignKey("stocks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    trading_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        index=True,
    )

    open: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    high: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    low: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    close: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    volume: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    stock = relationship("Stock", back_populates="prices")

    indicator = relationship(
        "TechnicalIndicator",
        back_populates="stock_price",
        uselist=False,
        cascade="all, delete-orphan",
    )
