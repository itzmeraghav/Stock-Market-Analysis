from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from stockmarketanalytics.data.app_db_context import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    stock_id: Mapped[int] = mapped_column(
        ForeignKey("stocks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    prediction_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    target_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    actual_close: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    predicted_close: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    predicted_direction: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    model_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    stock = relationship("Stock", back_populates="predictions")
