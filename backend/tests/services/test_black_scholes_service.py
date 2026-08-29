from __future__ import annotations

import pytest
from stockmarketanalytics.services.black_scholes_service import BlackScholesService


class _FakeSession:
    """Minimal fake SQLAlchemy Session recording add/commit calls."""

    def __init__(self):
        self.added = []
        self.committed = False

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True


@pytest.fixture
def fake_session():
    return _FakeSession()


@pytest.fixture
def service(fake_session):
    return BlackScholesService(fake_session)


class TestCalculateExactValues:
    def test_at_the_money_one_year(self, service, fake_stock):
        result = service.calculate(
            stock=fake_stock,
            spot_price=100.0,
            strike_price=100.0,
            days_to_expiry=365,
            risk_free_rate=0.05,
            volatility=0.2,
            persist=False,
        )

        assert result.call_price == 10.4506
        assert result.put_price == 5.5735
        assert result.call_delta == 0.6368
        assert result.put_delta == -0.3632
        assert result.gamma == 0.018762
        assert result.vega == 0.3752
        assert result.theta == -0.0176
        assert result.rho == 0.5323

    def test_out_of_the_money_call_short_dated(self, service, fake_stock):
        result = service.calculate(
            stock=fake_stock,
            spot_price=150.0,
            strike_price=160.0,
            days_to_expiry=90,
            risk_free_rate=0.03,
            volatility=0.25,
            persist=False,
        )

        assert result.call_price == 4.0262
        assert result.put_price == 12.847
        assert result.call_delta == 0.3452
        assert result.put_delta == -0.6548
        assert result.gamma == 0.019791
        assert result.vega == 0.2745
        assert result.theta == -0.0421
        assert result.rho == 0.1178

    def test_in_the_money_put_short_dated(self, service, fake_stock):
        result = service.calculate(
            stock=fake_stock,
            spot_price=200.0,
            strike_price=210.0,
            days_to_expiry=30,
            risk_free_rate=0.04,
            volatility=0.35,
            persist=False,
        )

        assert result.call_price == 4.3607
        assert result.put_price == 13.6714
        assert result.call_delta == 0.3434
        assert result.put_delta == -0.6566
        assert result.gamma == 0.018326
        assert result.vega == 0.2109
        assert result.theta == -0.1301
        assert result.rho == 0.0529


class TestCalculateInvariants:
    @pytest.mark.parametrize(
        "spot,strike,days,rate,vol",
        [
            (100.0, 100.0, 365, 0.05, 0.2),
            (150.0, 160.0, 90, 0.03, 0.25),
            (200.0, 210.0, 30, 0.04, 0.35),
            (50.0, 45.0, 180, 0.02, 0.15),
        ],
    )
    def test_put_call_parity_holds(
        self, service, fake_stock, spot, strike, days, rate, vol
    ):
        import math

        time_to_expiry = days / 365.0

        result = service.calculate(
            stock=fake_stock,
            spot_price=spot,
            strike_price=strike,
            days_to_expiry=days,
            risk_free_rate=rate,
            volatility=vol,
            persist=False,
        )

        lhs = result.call_price - result.put_price
        rhs = spot - strike * math.exp(-rate * time_to_expiry)
        assert lhs == pytest.approx(rhs, abs=0.01)

    @pytest.mark.parametrize(
        "spot,strike,days,rate,vol",
        [
            (100.0, 100.0, 365, 0.05, 0.2),
            (150.0, 160.0, 90, 0.03, 0.25),
            (200.0, 210.0, 30, 0.04, 0.35),
        ],
    )
    def test_put_delta_equals_call_delta_minus_one(
        self, service, fake_stock, spot, strike, days, rate, vol
    ):
        result = service.calculate(
            stock=fake_stock,
            spot_price=spot,
            strike_price=strike,
            days_to_expiry=days,
            risk_free_rate=rate,
            volatility=vol,
            persist=False,
        )

        assert result.put_delta == pytest.approx(result.call_delta - 1, abs=1e-9)

    def test_call_delta_increases_as_option_moves_deeper_in_the_money(
        self, service, fake_stock
    ):
        otm = service.calculate(
            stock=fake_stock,
            spot_price=90.0,
            strike_price=100.0,
            days_to_expiry=180,
            risk_free_rate=0.05,
            volatility=0.2,
            persist=False,
        )
        atm = service.calculate(
            stock=fake_stock,
            spot_price=100.0,
            strike_price=100.0,
            days_to_expiry=180,
            risk_free_rate=0.05,
            volatility=0.2,
            persist=False,
        )
        itm = service.calculate(
            stock=fake_stock,
            spot_price=110.0,
            strike_price=100.0,
            days_to_expiry=180,
            risk_free_rate=0.05,
            volatility=0.2,
            persist=False,
        )

        assert otm.call_delta < atm.call_delta < itm.call_delta

    def test_call_and_put_prices_are_non_negative(self, service, fake_stock):
        result = service.calculate(
            stock=fake_stock,
            spot_price=75.0,
            strike_price=120.0,
            days_to_expiry=14,
            risk_free_rate=0.01,
            volatility=0.6,
            persist=False,
        )

        assert result.call_price >= 0
        assert result.put_price >= 0

    def test_gamma_is_identical_for_call_and_put(self, service, fake_stock):
        result = service.calculate(
            stock=fake_stock,
            spot_price=100.0,
            strike_price=105.0,
            days_to_expiry=60,
            risk_free_rate=0.05,
            volatility=0.3,
            persist=False,
        )

        assert result.gamma > 0

    def test_higher_volatility_increases_both_call_and_put_price(
        self, service, fake_stock
    ):
        low_vol = service.calculate(
            stock=fake_stock,
            spot_price=100.0,
            strike_price=100.0,
            days_to_expiry=90,
            risk_free_rate=0.05,
            volatility=0.15,
            persist=False,
        )
        high_vol = service.calculate(
            stock=fake_stock,
            spot_price=100.0,
            strike_price=100.0,
            days_to_expiry=90,
            risk_free_rate=0.05,
            volatility=0.45,
            persist=False,
        )

        assert high_vol.call_price > low_vol.call_price
        assert high_vol.put_price > low_vol.put_price


class TestCalculateValidation:
    @pytest.mark.parametrize("bad_spot", [0, -50.0])
    def test_raises_for_non_positive_spot_price(self, service, fake_stock, bad_spot):
        with pytest.raises(
            ValueError, match="spot_price and strike_price must be greater than zero"
        ):
            service.calculate(
                stock=fake_stock,
                spot_price=bad_spot,
                strike_price=100.0,
                days_to_expiry=30,
                risk_free_rate=0.05,
                volatility=0.2,
            )

    @pytest.mark.parametrize("bad_strike", [0, -100.0])
    def test_raises_for_non_positive_strike_price(
        self, service, fake_stock, bad_strike
    ):
        with pytest.raises(
            ValueError, match="spot_price and strike_price must be greater than zero"
        ):
            service.calculate(
                stock=fake_stock,
                spot_price=100.0,
                strike_price=bad_strike,
                days_to_expiry=30,
                risk_free_rate=0.05,
                volatility=0.2,
            )

    @pytest.mark.parametrize("bad_days", [0, -10])
    def test_raises_for_non_positive_days_to_expiry(
        self, service, fake_stock, bad_days
    ):
        with pytest.raises(
            ValueError, match="days_to_expiry must be greater than zero"
        ):
            service.calculate(
                stock=fake_stock,
                spot_price=100.0,
                strike_price=100.0,
                days_to_expiry=bad_days,
                risk_free_rate=0.05,
                volatility=0.2,
            )

    @pytest.mark.parametrize("bad_vol", [0, -0.2])
    def test_raises_for_non_positive_volatility(self, service, fake_stock, bad_vol):
        with pytest.raises(ValueError, match="volatility must be greater than zero"):
            service.calculate(
                stock=fake_stock,
                spot_price=100.0,
                strike_price=100.0,
                days_to_expiry=30,
                risk_free_rate=0.05,
                volatility=bad_vol,
            )

    def test_does_not_touch_db_when_validation_fails(
        self, service, fake_stock, fake_session
    ):
        with pytest.raises(ValueError):
            service.calculate(
                stock=fake_stock,
                spot_price=-1,
                strike_price=100.0,
                days_to_expiry=30,
                risk_free_rate=0.05,
                volatility=0.2,
            )

        assert fake_session.added == []
        assert fake_session.committed is False


class TestCalculatePersistence:
    def test_persist_true_adds_and_commits_record(
        self, service, fake_stock, fake_session
    ):
        result = service.calculate(
            stock=fake_stock,
            spot_price=100.0,
            strike_price=100.0,
            days_to_expiry=365,
            risk_free_rate=0.05,
            volatility=0.2,
            persist=True,
        )

        assert len(fake_session.added) == 1
        assert fake_session.committed is True
        record = fake_session.added[0]
        assert record.stock_id == fake_stock.id
        assert record.call_price == result.call_price
        assert record.put_price == result.put_price
        assert record.time_to_expiry == pytest.approx(365 / 365.0)

    def test_persist_false_does_not_touch_db(self, service, fake_stock, fake_session):
        service.calculate(
            stock=fake_stock,
            spot_price=100.0,
            strike_price=100.0,
            days_to_expiry=365,
            risk_free_rate=0.05,
            volatility=0.2,
            persist=False,
        )

        assert fake_session.added == []
        assert fake_session.committed is False

    def test_persist_defaults_to_true(self, service, fake_stock, fake_session):
        service.calculate(
            stock=fake_stock,
            spot_price=100.0,
            strike_price=100.0,
            days_to_expiry=365,
            risk_free_rate=0.05,
            volatility=0.2,
        )

        assert fake_session.committed is True

    def test_persisted_record_stores_rounded_greeks_not_raw_values(
        self, service, fake_stock, fake_session
    ):
        service.calculate(
            stock=fake_stock,
            spot_price=150.0,
            strike_price=160.0,
            days_to_expiry=90,
            risk_free_rate=0.03,
            volatility=0.25,
            persist=True,
        )

        record = fake_session.added[0]
        assert record.call_price == 4.0262
        assert record.gamma == 0.019791


class TestGetHistory:
    def test_queries_filters_orders_and_limits(self, fake_stock, monkeypatch):
        calls = {}

        class _FakeQuery:
            def filter(self, *args, **kwargs):
                calls["filter"] = True
                return self

            def order_by(self, *args, **kwargs):
                calls["order_by"] = True
                return self

            def limit(self, n):
                calls["limit"] = n
                return self

            def all(self):
                calls["all"] = True
                return ["record-1", "record-2"]

        class _FakeSessionWithQuery:
            def query(self, model):
                calls["query_model"] = model
                return _FakeQuery()

        service = BlackScholesService(_FakeSessionWithQuery())

        result = service.get_history(fake_stock, limit=25)

        assert result == ["record-1", "record-2"]
        assert calls["filter"] is True
        assert calls["order_by"] is True
        assert calls["limit"] == 25
        assert calls["all"] is True

    def test_default_limit_is_100(self, fake_stock):
        captured_limit = {}

        class _FakeQuery:
            def filter(self, *args, **kwargs):
                return self

            def order_by(self, *args, **kwargs):
                return self

            def limit(self, n):
                captured_limit["value"] = n
                return self

            def all(self):
                return []

        class _FakeSessionWithQuery:
            def query(self, model):
                return _FakeQuery()

        service = BlackScholesService(_FakeSessionWithQuery())

        service.get_history(fake_stock)

        assert captured_limit["value"] == 100
