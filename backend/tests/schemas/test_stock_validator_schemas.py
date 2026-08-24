from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from stockmarketanalytics.schemas.stock_price_validator import (
    StockPriceIn,
    StockPriceValidationError,
    dedupe_by_stock_id_and_date,
    validate_price_frame,
)


def _valid_record(**overrides):
    record = {
        "trading_date": date(2024, 1, 2),
        "open": 100.0,
        "high": 110.0,
        "low": 95.0,
        "close": 105.0,
        "volume": 1000,
    }
    record.update(overrides)
    return record


class TestStockPriceInFieldValidation:
    def test_valid_record_builds(self):
        item = StockPriceIn(symbol="infy", **_valid_record())

        assert item.symbol == "INFY"

    def test_symbol_is_stripped_and_uppercased(self):
        item = StockPriceIn(symbol="  tcs  ", **_valid_record())

        assert item.symbol == "TCS"

    def test_blank_symbol_raises(self):
        with pytest.raises(ValidationError):
            StockPriceIn(symbol="   ", **_valid_record())

    @pytest.mark.parametrize("field", ["open", "high", "low", "close"])
    def test_zero_price_raises(self, field):
        with pytest.raises(ValidationError):
            StockPriceIn(symbol="INFY", **_valid_record(**{field: 0}))

    @pytest.mark.parametrize("field", ["open", "high", "low", "close"])
    def test_negative_price_raises(self, field):
        with pytest.raises(ValidationError):
            StockPriceIn(symbol="INFY", **_valid_record(**{field: -1}))

    def test_negative_volume_raises(self):
        with pytest.raises(ValidationError):
            StockPriceIn(symbol="INFY", **_valid_record(volume=-1))

    def test_zero_volume_is_valid(self):
        item = StockPriceIn(symbol="INFY", **_valid_record(volume=0))

        assert item.volume == 0


class TestOhlcConsistency:
    def test_high_below_low_raises(self):
        with pytest.raises(ValidationError):
            StockPriceIn(symbol="INFY", **_valid_record(high=90.0, low=95.0))

    def test_high_below_open_raises(self):
        with pytest.raises(ValidationError):
            StockPriceIn(symbol="INFY", **_valid_record(high=100.0, open=105.0))

    def test_high_below_close_raises(self):
        with pytest.raises(ValidationError):
            StockPriceIn(symbol="INFY", **_valid_record(high=100.0, close=105.0))

    def test_low_above_open_raises(self):
        with pytest.raises(ValidationError):
            StockPriceIn(symbol="INFY", **_valid_record(low=101.0, open=100.0))

    def test_low_above_close_raises(self):
        with pytest.raises(ValidationError):
            StockPriceIn(symbol="INFY", **_valid_record(low=106.0, close=105.0))

    def test_high_equal_to_open_and_close_is_valid(self):
        item = StockPriceIn(
            symbol="INFY",
            **_valid_record(open=110.0, high=110.0, low=95.0, close=110.0),
        )

        assert item.high == 110.0

    def test_low_equal_to_open_and_close_is_valid(self):
        item = StockPriceIn(
            symbol="INFY", **_valid_record(open=95.0, high=110.0, low=95.0, close=95.0)
        )

        assert item.low == 95.0


class TestValidatePriceFrame:
    def test_all_valid_records_return_no_errors(self):
        records = [_valid_record(), _valid_record(trading_date=date(2024, 1, 3))]

        valid, errors = validate_price_frame("INFY", records)

        assert len(valid) == 2
        assert errors == []

    def test_invalid_record_is_collected_as_error(self):
        records = [_valid_record(), _valid_record(high=1.0, low=95.0)]

        valid, errors = validate_price_frame("INFY", records)

        assert len(valid) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], StockPriceValidationError)

    def test_error_retains_symbol_and_date(self):
        records = [_valid_record(high=1.0, low=95.0)]

        _, errors = validate_price_frame("INFY", records)

        assert errors[0].symbol == "INFY"
        assert errors[0].trading_date == date(2024, 1, 2)

    def test_empty_input_returns_empty_results(self):
        valid, errors = validate_price_frame("INFY", [])

        assert valid == []
        assert errors == []


class TestDedupeByStockIdAndDate:
    def test_removes_duplicate_symbol_date_pairs(self):
        items = [
            StockPriceIn(symbol="INFY", **_valid_record()),
            StockPriceIn(symbol="INFY", **_valid_record(close=106.0)),
        ]

        deduped = dedupe_by_stock_id_and_date(items)

        assert len(deduped) == 1
        assert deduped[0].close == 105.0

    def test_keeps_entries_for_different_dates(self):
        items = [
            StockPriceIn(symbol="INFY", **_valid_record(trading_date=date(2024, 1, 2))),
            StockPriceIn(symbol="INFY", **_valid_record(trading_date=date(2024, 1, 3))),
        ]

        deduped = dedupe_by_stock_id_and_date(items)

        assert len(deduped) == 2

    def test_keeps_entries_for_different_symbols_on_same_date(self):
        items = [
            StockPriceIn(symbol="INFY", **_valid_record()),
            StockPriceIn(symbol="TCS", **_valid_record()),
        ]

        deduped = dedupe_by_stock_id_and_date(items)

        assert len(deduped) == 2

    def test_empty_input_returns_empty_list(self):
        assert dedupe_by_stock_id_and_date([]) == []
