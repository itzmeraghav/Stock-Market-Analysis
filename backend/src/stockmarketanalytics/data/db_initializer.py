from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from stockmarketanalytics.data.app_db_context import Base, engine
from stockmarketanalytics.models.stock import Stock

logger = logging.getLogger("db_initializer")


def create_all_tables() -> None:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created (if not already present)")


def seed_default_stocks(db: Session) -> None:
    default_symbols = [
        ("RELIANCE", "Reliance Industries", "NSE"),
        ("TCS", "Tata Consultancy Services", "NSE"),
        ("INFY", "Infosys", "NSE"),
    ]

    for symbol, company_name, exchange in default_symbols:
        exists = db.query(Stock).filter(Stock.symbol == symbol).first()
        if exists:
            continue
        db.add(Stock(symbol=symbol, company_name=company_name, exchange=exchange))

    db.commit()
    logger.info("Default stock seed check complete")


def initialize_database(db: Session, seed: bool = True) -> None:
    create_all_tables()
    if seed:
        seed_default_stocks(db)
