from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy.exc import IntegrityError
from stockmarketanalytics.models.prediction import Prediction
from stockmarketanalytics.models.stock import Stock
from stockmarketanalytics.models.stock_price import StockPrice
from stockmarketanalytics.models.technical_indicator import TechnicalIndicator


class TestStock:
    def test_insert_and_read_back(self, session, make_stock):
        make_stock(symbol="INFY", company_name="Infosys Ltd")

        fetched = session.query(Stock).filter_by(symbol="INFY").one()

        assert fetched.company_name == "Infosys Ltd"
        assert fetched.exchange == "NSE"
        assert fetched.created_at is not None

    def test_exchange_defaults_to_nse_when_omitted(self, session):
        stock = Stock(symbol="TCS", company_name="Tata Consultancy Services")
        session.add(stock)
        session.commit()
        session.refresh(stock)

        assert stock.exchange == "NSE"

    def test_symbol_must_be_unique(self, session, make_stock):
        make_stock(symbol="INFY")
        session.add(Stock(symbol="INFY", company_name="Duplicate Infosys"))

        with pytest.raises(IntegrityError):
            session.commit()


class TestStockPrice:
    def test_insert_and_relationship_to_stock(
        self, session, make_stock, make_stock_price
    ):
        stock = make_stock(symbol="INFY")
        price = make_stock_price(stock, trading_date=date(2024, 1, 2))

        assert price.stock_id == stock.id
        assert price.stock.symbol == "INFY"
        assert price in stock.prices

    def test_unique_constraint_on_stock_and_date(
        self, session, make_stock, make_stock_price
    ):
        stock = make_stock(symbol="INFY")
        make_stock_price(stock, trading_date=date(2024, 1, 2))

        session.add(
            StockPrice(
                stock_id=stock.id,
                trading_date=date(2024, 1, 2),
                open=101.0,
                high=111.0,
                low=96.0,
                close=106.0,
                volume=2000,
            )
        )

        with pytest.raises(IntegrityError):
            session.commit()

    def test_same_date_allowed_for_different_stocks(
        self, session, make_stock, make_stock_price
    ):
        infy = make_stock(symbol="INFY")
        tcs = make_stock(symbol="TCS")

        make_stock_price(infy, trading_date=date(2024, 1, 2))
        make_stock_price(tcs, trading_date=date(2024, 1, 2))

        assert session.query(StockPrice).count() == 2

    def test_deleting_stock_cascades_to_prices(
        self, session, make_stock, make_stock_price
    ):
        stock = make_stock(symbol="INFY")
        make_stock_price(stock, trading_date=date(2024, 1, 2))

        session.delete(stock)
        session.commit()

        assert session.query(StockPrice).count() == 0


class TestTechnicalIndicator:
    def test_insert_and_relationship_to_stock_price(
        self, session, make_stock, make_stock_price
    ):
        stock = make_stock(symbol="INFY")
        price = make_stock_price(stock, trading_date=date(2024, 1, 2))

        indicator = TechnicalIndicator(
            stock_price_id=price.id,
            sma20=100.0,
            sma50=98.0,
            ema20=101.0,
            rsi14=55.0,
            macd=0.5,
            bollinger_upper=110.0,
            bollinger_lower=90.0,
            volatility=0.2,
        )
        session.add(indicator)
        session.commit()
        session.refresh(price)

        assert price.indicator.rsi14 == 55.0
        assert indicator.stock_price is price

    def test_nullable_metrics_default_to_none(
        self, session, make_stock, make_stock_price
    ):
        stock = make_stock(symbol="INFY")
        price = make_stock_price(stock, trading_date=date(2024, 1, 2))

        indicator = TechnicalIndicator(stock_price_id=price.id)
        session.add(indicator)
        session.commit()
        session.refresh(indicator)

        assert indicator.sma20 is None
        assert indicator.rsi14 is None

    def test_stock_price_id_must_be_unique(self, session, make_stock, make_stock_price):
        stock = make_stock(symbol="INFY")
        price = make_stock_price(stock, trading_date=date(2024, 1, 2))

        session.add(TechnicalIndicator(stock_price_id=price.id, sma20=100.0))
        session.commit()

        session.add(TechnicalIndicator(stock_price_id=price.id, sma20=101.0))

        with pytest.raises(IntegrityError):
            session.commit()

    def test_deleting_stock_price_cascades_to_indicator(
        self, session, make_stock, make_stock_price
    ):
        stock = make_stock(symbol="INFY")
        price = make_stock_price(stock, trading_date=date(2024, 1, 2))
        session.add(TechnicalIndicator(stock_price_id=price.id, sma20=100.0))
        session.commit()

        session.delete(price)
        session.commit()

        assert session.query(TechnicalIndicator).count() == 0


class TestPrediction:
    def test_insert_and_relationship_to_stock(self, session, make_stock):
        stock = make_stock(symbol="INFY")

        prediction = Prediction(
            stock_id=stock.id,
            prediction_date=date(2024, 1, 1),
            target_date=date(2024, 1, 2),
            predicted_close=150.5,
            predicted_direction="Bullish",
            confidence=0.8,
            model_name="lstm-v1",
        )
        session.add(prediction)
        session.commit()
        session.refresh(prediction)

        assert prediction.stock.symbol == "INFY"
        assert prediction in stock.predictions
        assert prediction.actual_close is None

    def test_created_at_is_populated_automatically(self, session, make_stock):
        stock = make_stock(symbol="INFY")
        prediction = Prediction(
            stock_id=stock.id,
            prediction_date=date(2024, 1, 1),
            target_date=date(2024, 1, 2),
            predicted_close=150.5,
            predicted_direction="Bullish",
            model_name="lstm-v1",
        )
        session.add(prediction)
        session.commit()
        session.refresh(prediction)

        assert prediction.created_at is not None

    def test_deleting_stock_cascades_to_predictions(self, session, make_stock):
        stock = make_stock(symbol="INFY")
        session.add(
            Prediction(
                stock_id=stock.id,
                prediction_date=date(2024, 1, 1),
                target_date=date(2024, 1, 2),
                predicted_close=150.5,
                predicted_direction="Bullish",
                model_name="lstm-v1",
            )
        )
        session.commit()

        session.delete(stock)
        session.commit()

        assert session.query(Prediction).count() == 0

    def test_missing_required_field_raises(self, session, make_stock):
        stock = make_stock(symbol="INFY")
        session.add(
            Prediction(
                stock_id=stock.id,
                prediction_date=date(2024, 1, 1),
                target_date=date(2024, 1, 2),
                predicted_close=150.5,
            )
        )

        with pytest.raises(IntegrityError):
            session.commit()
