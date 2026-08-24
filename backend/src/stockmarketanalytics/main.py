from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from stockmarketanalytics.data.app_db_context import SessionLocal
from stockmarketanalytics.data.db_initializer import initialize_database
from stockmarketanalytics.endpoints.prediction_endpoints import backtest_router
from stockmarketanalytics.endpoints.prediction_endpoints import (
    router as prediction_router,
)
from stockmarketanalytics.endpoints.stock_endpoints import indicator_router
from stockmarketanalytics.endpoints.stock_endpoints import router as stock_router

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("main")

app = FastAPI(title="Stock Market Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


app.include_router(stock_router, prefix="/api")
app.include_router(indicator_router, prefix="/api")
app.include_router(prediction_router, prefix="/api")
app.include_router(backtest_router, prefix="/api")


@app.on_event("startup")
def on_startup() -> None:
    db = SessionLocal()
    try:
        initialize_database(db, seed=True)
    finally:
        db.close()
    logger.info("Application startup complete")
