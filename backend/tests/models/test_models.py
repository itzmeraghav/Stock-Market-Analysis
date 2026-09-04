from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from stockmarketanalytics.models.option_calculation import OptionCalculation
from stockmarketanalytics.models.prediction import Prediction
from stockmarketanalytics.models.refresh_tokens import RefreshToken
from stockmarketanalytics.models.stock import Stock
from stockmarketanalytics.models.stock_price import StockPrice
from stockmarketanalytics.models.technical_indicator import TechnicalIndicator
from stockmarketanalytics.models.trading_analysis import TradingAnalysis
from stockmarketanalytics.models.users import User


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


class TestOptionCalculationModel:
    def test_table_name_is_option_calculations(self):
        assert OptionCalculation.__tablename__ == "option_calculations"

    def test_has_expected_columns(self):
        expected_columns = {
            "id",
            "stock_id",
            "calculation_date",
            "spot_price",
            "strike_price",
            "time_to_expiry",
            "risk_free_rate",
            "volatility",
            "call_price",
            "put_price",
            "call_delta",
            "put_delta",
            "gamma",
            "vega",
            "theta",
            "rho",
        }

        mapper = inspect(OptionCalculation)
        actual_columns = {col.key for col in mapper.columns}

        assert actual_columns == expected_columns

    def test_id_is_primary_key(self):
        mapper = inspect(OptionCalculation)

        assert mapper.columns["id"].primary_key is True

    def test_stock_id_is_foreign_key_and_not_nullable(self):
        mapper = inspect(OptionCalculation)
        stock_id_column = mapper.columns["stock_id"]

        assert stock_id_column.nullable is False
        fk_targets = {fk.target_fullname for fk in stock_id_column.foreign_keys}
        assert fk_targets == {"stocks.id"}

    def test_numeric_fields_are_not_nullable(self):
        numeric_fields = [
            "spot_price",
            "strike_price",
            "time_to_expiry",
            "risk_free_rate",
            "volatility",
            "call_price",
            "put_price",
            "call_delta",
            "put_delta",
            "gamma",
            "vega",
            "theta",
            "rho",
        ]

        mapper = inspect(OptionCalculation)

        for field in numeric_fields:
            assert mapper.columns[field].nullable is False, (
                f"{field} should be non-nullable"
            )

    def test_calculation_date_has_server_default(self):
        mapper = inspect(OptionCalculation)
        calculation_date_column = mapper.columns["calculation_date"]

        assert calculation_date_column.server_default is not None

    def test_stock_relationship_is_configured(self):
        mapper = inspect(OptionCalculation)
        relationship_names = {rel.key for rel in mapper.relationships}

        assert "stock" in relationship_names
        assert mapper.relationships["stock"].back_populates == "option_calculations"

    def test_instantiation_with_python_side_defaults(self):
        record = OptionCalculation(
            stock_id=1,
            spot_price=150.0,
            strike_price=155.0,
            time_to_expiry=0.0822,
            risk_free_rate=0.05,
            volatility=0.2,
            call_price=5.12,
            put_price=8.34,
            call_delta=0.45,
            put_delta=-0.55,
            gamma=0.03,
            vega=0.12,
            theta=-0.02,
            rho=0.04,
        )

        assert record.stock_id == 1
        assert record.call_price == 5.12
        assert record.calculation_date is None


def _build_analysis(stock_id: int) -> TradingAnalysis:
    return TradingAnalysis(
        stock_id=stock_id,
        current_price=100.0,
        predicted_price=110.0,
        expected_change_percent=10.0,
        direction="UP",
        model_name="xgboost",
        investment_amount=1000.0,
        shares=10,
        invested_amount=1000.0,
        remaining_amount=0.0,
        risk_percentage=10.0,
        stop_loss_price=90.0,
        target_percentage=20.0,
        target_price=120.0,
        maximum_loss=100.0,
        potential_profit=200.0,
        risk_reward_ratio=2.0,
    )


class TestTradingAnalysisPersistence:
    def test_creates_and_persists_with_required_fields(self, session, make_stock):
        stock = make_stock()
        analysis = _build_analysis(stock.id)

        session.add(analysis)
        session.commit()
        session.refresh(analysis)

        assert analysis.id is not None
        assert analysis.stock_id == stock.id
        assert analysis.model_name == "xgboost"
        assert analysis.shares == 10
        assert analysis.risk_reward_ratio == pytest.approx(2.0)

    def test_calculation_date_defaults_to_now_on_insert(self, session, make_stock):
        stock = make_stock()
        analysis = _build_analysis(stock.id)

        session.add(analysis)
        session.commit()
        session.refresh(analysis)

        assert analysis.calculation_date is not None
        assert isinstance(analysis.calculation_date, datetime)

    def test_raises_when_stock_id_is_missing(self, session):
        analysis = _build_analysis(stock_id=None)

        session.add(analysis)
        with pytest.raises(IntegrityError):
            session.commit()

    def test_raises_when_stock_id_references_nonexistent_stock(self, session):
        analysis = _build_analysis(stock_id=999999)

        session.add(analysis)
        with pytest.raises(IntegrityError):
            session.commit()

    def test_multiple_analyses_can_reference_same_stock(self, session, make_stock):
        stock = make_stock()
        first = _build_analysis(stock.id)
        second = _build_analysis(stock.id)

        session.add_all([first, second])
        session.commit()

        assert first.id != second.id
        assert first.stock_id == second.stock_id == stock.id


class TestTradingAnalysisRelationship:
    def test_stock_relationship_loads_the_related_stock(self, session, make_stock):
        stock = make_stock()
        analysis = _build_analysis(stock.id)

        session.add(analysis)
        session.commit()
        session.refresh(analysis)

        assert analysis.stock.id == stock.id
        assert analysis.stock.symbol == stock.symbol

    def test_stock_back_populates_trading_analyses(self, session, make_stock):
        stock = make_stock()
        analysis = _build_analysis(stock.id)

        session.add(analysis)
        session.commit()
        session.refresh(stock)

        assert analysis in stock.trading_analyses


@pytest.fixture
def make_user(session):
    def _make(username: str = "someone", hashed_password: str = "hashedvalue") -> User:
        user = User(username=username, hashed_password=hashed_password)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

    return _make


class TestUserModel:
    def test_user_is_persisted_with_defaults(self, make_user):
        user = make_user()

        assert user.id is not None
        assert user.is_active is True
        assert user.failed_login_attempts == 0
        assert user.locked_until is None
        assert user.created_at is not None

    def test_duplicate_username_raises_integrity_error(self, session, make_user):
        make_user(username="dupeuser")

        with pytest.raises(IntegrityError):
            session.add(User(username="dupeuser", hashed_password="anotherhash"))
            session.commit()

    def test_user_refresh_tokens_relationship_starts_empty(self, make_user):
        user = make_user()

        assert user.refresh_tokens == []

    def test_deleting_user_cascades_to_refresh_tokens(self, session, make_user):
        user = make_user()
        token = RefreshToken(
            user_id=user.id,
            token_hash="somehash",
            expires_at=datetime.now(UTC) + timedelta(days=1),
        )
        session.add(token)
        session.commit()

        session.delete(user)
        session.commit()

        remaining = (
            session.query(RefreshToken).filter(RefreshToken.user_id == user.id).all()
        )
        assert remaining == []


class TestRefreshTokenModel:
    def test_refresh_token_is_persisted_with_defaults(self, session, make_user):
        user = make_user()
        token = RefreshToken(
            user_id=user.id,
            token_hash="somehash",
            expires_at=datetime.now(UTC) + timedelta(days=7),
        )

        session.add(token)
        session.commit()
        session.refresh(token)

        assert token.id is not None
        assert token.revoked is False
        assert token.issued_at is not None
        assert token.last_used_at is None

    def test_duplicate_token_hash_raises_integrity_error(self, session, make_user):
        user = make_user()
        session.add(
            RefreshToken(
                user_id=user.id,
                token_hash="duplicatehash",
                expires_at=datetime.now(UTC) + timedelta(days=1),
            )
        )
        session.commit()

        with pytest.raises(IntegrityError):
            session.add(
                RefreshToken(
                    user_id=user.id,
                    token_hash="duplicatehash",
                    expires_at=datetime.now(UTC) + timedelta(days=1),
                )
            )
            session.commit()

    def test_refresh_token_user_relationship_resolves(self, session, make_user):
        user = make_user()
        token = RefreshToken(
            user_id=user.id,
            token_hash="somehash",
            expires_at=datetime.now(UTC) + timedelta(days=1),
        )
        session.add(token)
        session.commit()
        session.refresh(token)

        assert token.user.id == user.id

    def test_refresh_token_requires_valid_user_id(self, session):
        session.add(
            RefreshToken(
                user_id=9999,
                token_hash="orphanhash",
                expires_at=datetime.now(UTC) + timedelta(days=1),
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
