from __future__ import annotations

import logging
import math

from scipy.stats import norm
from sqlalchemy.orm import Session
from stockmarketanalytics.models.option_calculation import OptionCalculation
from stockmarketanalytics.models.stock import Stock
from stockmarketanalytics.schemas.option_schemas import OptionCalculationResponse

logger = logging.getLogger("black_scholes_service")


class BlackScholesService:
    def __init__(self, db: Session):
        self.db = db

    def _d1(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
    ) -> float:
        numerator = (
            math.log(spot / strike)
            + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry
        )
        denominator = volatility * math.sqrt(time_to_expiry)
        return numerator / denominator

    def _d2(self, d1: float, volatility: float, time_to_expiry: float) -> float:
        return d1 - volatility * math.sqrt(time_to_expiry)

    def calculate(
        self,
        stock: Stock,
        spot_price: float,
        strike_price: float,
        days_to_expiry: int,
        risk_free_rate: float,
        volatility: float,
        persist: bool = True,
    ) -> OptionCalculationResponse:
        if spot_price <= 0 or strike_price <= 0:
            raise ValueError("spot_price and strike_price must be greater than zero")
        if days_to_expiry <= 0:
            raise ValueError("days_to_expiry must be greater than zero")
        if volatility <= 0:
            raise ValueError("volatility must be greater than zero")

        time_to_expiry = days_to_expiry / 365.0

        d1 = self._d1(
            spot_price, strike_price, time_to_expiry, risk_free_rate, volatility
        )
        d2 = self._d2(d1, volatility, time_to_expiry)

        discount_factor = math.exp(-risk_free_rate * time_to_expiry)

        call_price = spot_price * norm.cdf(
            d1
        ) - strike_price * discount_factor * norm.cdf(d2)
        put_price = strike_price * discount_factor * norm.cdf(
            -d2
        ) - spot_price * norm.cdf(-d1)

        call_delta = norm.cdf(d1)
        put_delta = norm.cdf(d1) - 1

        gamma = norm.pdf(d1) / (spot_price * volatility * math.sqrt(time_to_expiry))

        vega = spot_price * norm.pdf(d1) * math.sqrt(time_to_expiry) / 100

        theta = (
            -spot_price * norm.pdf(d1) * volatility / (2 * math.sqrt(time_to_expiry))
            - risk_free_rate * strike_price * discount_factor * norm.cdf(d2)
        ) / 365

        rho = strike_price * time_to_expiry * discount_factor * norm.cdf(d2) / 100

        response = OptionCalculationResponse(
            call_price=round(call_price, 4),
            put_price=round(put_price, 4),
            call_delta=round(call_delta, 4),
            put_delta=round(put_delta, 4),
            gamma=round(gamma, 6),
            vega=round(vega, 4),
            theta=round(theta, 4),
            rho=round(rho, 4),
        )

        if persist:
            record = OptionCalculation(
                stock_id=stock.id,
                spot_price=spot_price,
                strike_price=strike_price,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                call_price=response.call_price,
                put_price=response.put_price,
                call_delta=response.call_delta,
                put_delta=response.put_delta,
                gamma=response.gamma,
                vega=response.vega,
                theta=response.theta,
                rho=response.rho,
            )
            self.db.add(record)
            self.db.commit()
            logger.info(
                "Persisted option calculation for %s (K=%.2f, %dd)",
                stock.symbol,
                strike_price,
                days_to_expiry,
            )

        return response

    def get_history(self, stock: Stock, limit: int = 100) -> list[OptionCalculation]:
        return (
            self.db.query(OptionCalculation)
            .filter(OptionCalculation.stock_id == stock.id)
            .order_by(OptionCalculation.calculation_date.desc())
            .limit(limit)
            .all()
        )
