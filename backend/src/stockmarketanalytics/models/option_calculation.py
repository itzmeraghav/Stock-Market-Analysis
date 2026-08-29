from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from stockmarketanalytics.data.app_db_context import Base


class OptionCalculation(Base):
    __tablename__ = "option_calculations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    stock_id: Mapped[int] = mapped_column(
        ForeignKey("stocks.id"), nullable=False, index=True
    )
    calculation_date: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    spot_price: Mapped[float] = mapped_column(Float, nullable=False)
    strike_price: Mapped[float] = mapped_column(Float, nullable=False)
    time_to_expiry: Mapped[float] = mapped_column(Float, nullable=False)
    risk_free_rate: Mapped[float] = mapped_column(Float, nullable=False)
    volatility: Mapped[float] = mapped_column(Float, nullable=False)
    call_price: Mapped[float] = mapped_column(Float, nullable=False)
    put_price: Mapped[float] = mapped_column(Float, nullable=False)
    call_delta: Mapped[float] = mapped_column(Float, nullable=False)
    put_delta: Mapped[float] = mapped_column(Float, nullable=False)
    gamma: Mapped[float] = mapped_column(Float, nullable=False)
    vega: Mapped[float] = mapped_column(Float, nullable=False)
    theta: Mapped[float] = mapped_column(Float, nullable=False)
    rho: Mapped[float] = mapped_column(Float, nullable=False)

    stock = relationship("Stock", back_populates="option_calculations")
