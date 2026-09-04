from __future__ import annotations

import logging

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from stockmarketanalytics.auth_dependencies import get_current_user
from stockmarketanalytics.data.app_db_context import SessionLocal
from stockmarketanalytics.data.db_initializer import initialize_database
from stockmarketanalytics.endpoints.auth_endpoints import router as auth_router
from stockmarketanalytics.endpoints.option_endpoints import router as option_router
from stockmarketanalytics.endpoints.prediction_endpoints import backtest_router
from stockmarketanalytics.endpoints.prediction_endpoints import (
    router as prediction_router,
)
from stockmarketanalytics.endpoints.stock_endpoints import indicator_router
from stockmarketanalytics.endpoints.stock_endpoints import router as stock_router
from stockmarketanalytics.endpoints.trading_endpoints import router as trading_router
from stockmarketanalytics.rate_limiter import GeneralRateLimitMiddleware, limiter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("main")

app = FastAPI(title="Stock Market Analytics API", version="1.0.0")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(GeneralRateLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_require_auth = [Depends(get_current_user)]


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


app.include_router(auth_router, prefix="/api")
app.include_router(stock_router, prefix="/api", dependencies=_require_auth)
app.include_router(indicator_router, prefix="/api", dependencies=_require_auth)
app.include_router(prediction_router, prefix="/api", dependencies=_require_auth)
app.include_router(backtest_router, prefix="/api", dependencies=_require_auth)
app.include_router(option_router, prefix="/api", dependencies=_require_auth)
app.include_router(trading_router, prefix="/api", dependencies=_require_auth)


@app.on_event("startup")
def on_startup() -> None:
    db = SessionLocal()
    try:
        initialize_database(db, seed=True)
    finally:
        db.close()
    logger.info("Application startup complete")
