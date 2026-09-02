from __future__ import annotations

import logging
import math

from sqlalchemy.orm import Session
from stockmarketanalytics.models.stock import Stock
from stockmarketanalytics.models.trading_analysis import TradingAnalysis
from stockmarketanalytics.schemas.trading_schemas import TradingAnalysisResponse
from stockmarketanalytics.services.prediction_service import PredictionService

logger = logging.getLogger("risk_management_service")


class RiskManagementService:
    def __init__(self, db: Session):
        self.db = db

    def calculate_shares(self, investment_amount: float, current_price: float) -> int:
        if current_price <= 0:
            raise ValueError("current_price must be greater than zero")
        return math.floor(investment_amount / current_price)

    def calculate_invested_amount(self, shares: int, current_price: float) -> float:
        return shares * current_price

    def calculate_remaining_amount(
        self, investment_amount: float, invested_amount: float
    ) -> float:
        return investment_amount - invested_amount

    def calculate_stop_loss(
        self, current_price: float, risk_percentage: float
    ) -> float:
        return current_price * (1 - risk_percentage / 100)

    def calculate_target_price(
        self, current_price: float, target_percentage: float
    ) -> float:
        return current_price * (1 + target_percentage / 100)

    def calculate_maximum_loss(
        self, current_price: float, stop_loss_price: float, shares: int
    ) -> float:
        loss_per_share = current_price - stop_loss_price
        return loss_per_share * shares

    def calculate_potential_profit(
        self, current_price: float, target_price: float, shares: int
    ) -> float:
        profit_per_share = target_price - current_price
        return profit_per_share * shares

    def calculate_risk_reward(
        self, potential_profit: float, maximum_loss: float
    ) -> float:
        if maximum_loss <= 0:
            raise ValueError(
                "maximum_loss must be greater than zero to calculate a risk/reward ratio"
            )
        return potential_profit / maximum_loss

    def analyze_trade(
        self,
        stock: Stock,
        investment_amount: float,
        risk_percentage: float,
        target_percentage: float,
        persist: bool = True,
    ) -> TradingAnalysisResponse:
        prediction_service = PredictionService(self.db)

        prediction_result = prediction_service.predict_all(stock, persist=False)
        best_model = prediction_result.best_model
        prediction = next(
            p for p in prediction_result.predictions if p.model == best_model
        )

        current_price = prediction.current_price
        predicted_price = prediction.predicted_price

        shares = self.calculate_shares(investment_amount, current_price)
        if shares <= 0:
            raise ValueError(
                "Investment amount is too small to buy at least one share at the current price"
            )

        invested_amount = self.calculate_invested_amount(shares, current_price)
        remaining_amount = self.calculate_remaining_amount(
            investment_amount, invested_amount
        )

        stop_loss_price = self.calculate_stop_loss(current_price, risk_percentage)
        target_price = self.calculate_target_price(current_price, target_percentage)

        maximum_loss = self.calculate_maximum_loss(
            current_price, stop_loss_price, shares
        )
        potential_profit = self.calculate_potential_profit(
            current_price, target_price, shares
        )
        risk_reward_ratio = self.calculate_risk_reward(potential_profit, maximum_loss)

        responses = TradingAnalysisResponse(
            symbol=stock.symbol,
            current_price=round(current_price, 2),
            predicted_price=round(predicted_price, 2),
            expected_change_percent=round(prediction.expected_change_percent, 4),
            direction=prediction.direction,
            model=prediction.model,
            investment_amount=investment_amount,
            shares=shares,
            invested_amount=round(invested_amount, 2),
            remaining_amount=round(remaining_amount, 2),
            risk_percentage=risk_percentage,
            stop_loss_price=round(stop_loss_price, 2),
            target_percentage=target_percentage,
            target_price=round(target_price, 2),
            maximum_loss=round(maximum_loss, 2),
            potential_profit=round(potential_profit, 2),
            risk_reward_ratio=round(risk_reward_ratio, 2),
        )

        if persist:
            record = TradingAnalysis(
                stock_id=stock.id,
                current_price=responses.current_price,
                predicted_price=responses.predicted_price,
                expected_change_percent=responses.expected_change_percent,
                direction=responses.direction.value,
                model_name=responses.model,
                investment_amount=responses.investment_amount,
                shares=responses.shares,
                invested_amount=responses.invested_amount,
                remaining_amount=responses.remaining_amount,
                risk_percentage=responses.risk_percentage,
                stop_loss_price=responses.stop_loss_price,
                target_percentage=responses.target_percentage,
                target_price=responses.target_price,
                maximum_loss=responses.maximum_loss,
                potential_profit=responses.potential_profit,
                risk_reward_ratio=responses.risk_reward_ratio,
            )
            self.db.add(record)
            self.db.commit()

            logger.info(
                "Persisted trading analysis for %s: shares=%d, R:R=1:%.2f",
                stock.symbol,
                shares,
                risk_reward_ratio,
            )

        return responses
