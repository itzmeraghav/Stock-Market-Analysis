from __future__ import annotations

from sqlalchemy import Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from stockmarketanalytics.data.app_db_context import Base


class TechnicalIndicator(Base):
    __tablename__ = "technical_indicators"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    stock_price_id: Mapped[int] = mapped_column(
        ForeignKey(
            "stock_prices.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        unique=True,
        index=True,
    )

    sma20: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    sma50: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    ema20: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    rsi14: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    macd: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    bollinger_upper: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    bollinger_lower: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    volatility: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    stock_price = relationship("StockPrice", back_populates="indicator")
