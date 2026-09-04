from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from stockmarketanalytics.data.app_db_context import Base


class TradingAnalysis(Base):
    __tablename__ = "trading_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    stock_id: Mapped[int] = mapped_column(
        ForeignKey("stocks.id"), nullable=False, index=True
    )
    calculation_date: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )

    current_price: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_price: Mapped[float] = mapped_column(Float, nullable=False)
    expected_change_percent: Mapped[float] = mapped_column(Float, nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)

    investment_amount: Mapped[float] = mapped_column(Float, nullable=False)
    shares: Mapped[int] = mapped_column(Integer, nullable=False)
    invested_amount: Mapped[float] = mapped_column(Float, nullable=False)
    remaining_amount: Mapped[float] = mapped_column(Float, nullable=False)

    risk_percentage: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss_price: Mapped[float] = mapped_column(Float, nullable=False)

    target_percentage: Mapped[float] = mapped_column(Float, nullable=False)
    target_price: Mapped[float] = mapped_column(Float, nullable=False)

    maximum_loss: Mapped[float] = mapped_column(Float, nullable=False)
    potential_profit: Mapped[float] = mapped_column(Float, nullable=False)
    risk_reward_ratio: Mapped[float] = mapped_column(Float, nullable=False)

    stock = relationship("Stock", back_populates="trading_analyses")
