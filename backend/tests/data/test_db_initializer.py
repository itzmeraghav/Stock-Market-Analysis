from __future__ import annotations

from unittest.mock import MagicMock

from stockmarketanalytics.data import db_initializer
from stockmarketanalytics.models.stock import Stock


class TestCreateAllTables:
    """Tests for create_all_tables()."""

    def test_create_all_tables_calls_metadata_create_all_with_engine(self, monkeypatch):
        mock_create_all = MagicMock()
        monkeypatch.setattr(db_initializer.Base.metadata, "create_all", mock_create_all)

        db_initializer.create_all_tables()

        mock_create_all.assert_called_once_with(bind=db_initializer.engine)


class TestSeedDefaultStocks:
    """Tests for seed_default_stocks()."""

    def test_adds_all_default_stocks_when_none_exist(
        self, mock_db_session, make_query_mock
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=None)

        db_initializer.seed_default_stocks(mock_db_session)

        assert mock_db_session.add.call_count == 3
        added_symbols = {c.args[0].symbol for c in mock_db_session.add.call_args_list}
        assert added_symbols == {"RELIANCE", "TCS", "INFY"}

    def test_commits_once_regardless_of_number_of_stocks_added(
        self, mock_db_session, make_query_mock
    ):
        mock_db_session.query.return_value = make_query_mock(first_return=None)

        db_initializer.seed_default_stocks(mock_db_session)

        mock_db_session.commit.assert_called_once()

    def test_skips_stock_that_already_exists(self, mock_db_session):
        """seed_default_stocks iterates default_symbols in a fixed order
        (RELIANCE, TCS, INFY), calling db.query(...).filter(...).first()
        once per symbol. We simulate "RELIANCE already exists" by returning
        a truthy value on the first .first() call and None afterwards,
        without depending on SQLAlchemy's internal expression structure.
        """
        existing_stock = Stock(
            symbol="RELIANCE", company_name="Reliance Industries", exchange="NSE"
        )
        first_call_results = iter([existing_stock, None, None])

        query_mock = MagicMock()
        query_mock.filter.return_value.first.side_effect = lambda: next(
            first_call_results
        )
        mock_db_session.query.return_value = query_mock

        db_initializer.seed_default_stocks(mock_db_session)

        added_symbols = {c.args[0].symbol for c in mock_db_session.add.call_args_list}
        assert added_symbols == {"TCS", "INFY"}
        assert mock_db_session.add.call_count == 2

    def test_adds_nothing_when_all_default_stocks_already_exist(
        self, mock_db_session, make_query_mock
    ):
        already_exists = Stock(
            symbol="RELIANCE", company_name="Reliance Industries", exchange="NSE"
        )
        mock_db_session.query.return_value = make_query_mock(
            first_return=already_exists
        )

        db_initializer.seed_default_stocks(mock_db_session)

        mock_db_session.add.assert_not_called()
        mock_db_session.commit.assert_called_once()

    def test_added_stocks_have_correct_exchange(self, mock_db_session, make_query_mock):
        mock_db_session.query.return_value = make_query_mock(first_return=None)

        db_initializer.seed_default_stocks(mock_db_session)

        added_stocks = [c.args[0] for c in mock_db_session.add.call_args_list]
        assert all(stock.exchange == "NSE" for stock in added_stocks)


class TestInitializeDatabase:
    """Tests for initialize_database()."""

    def test_always_creates_tables(self, mock_db_session, monkeypatch):
        mock_create_tables = MagicMock()
        mock_seed = MagicMock()
        monkeypatch.setattr(db_initializer, "create_all_tables", mock_create_tables)
        monkeypatch.setattr(db_initializer, "seed_default_stocks", mock_seed)

        db_initializer.initialize_database(mock_db_session, seed=False)

        mock_create_tables.assert_called_once()

    def test_seeds_when_seed_flag_is_true(self, mock_db_session, monkeypatch):
        monkeypatch.setattr(db_initializer, "create_all_tables", MagicMock())
        mock_seed = MagicMock()
        monkeypatch.setattr(db_initializer, "seed_default_stocks", mock_seed)

        db_initializer.initialize_database(mock_db_session, seed=True)

        mock_seed.assert_called_once_with(mock_db_session)

    def test_does_not_seed_when_seed_flag_is_false(self, mock_db_session, monkeypatch):
        monkeypatch.setattr(db_initializer, "create_all_tables", MagicMock())
        mock_seed = MagicMock()
        monkeypatch.setattr(db_initializer, "seed_default_stocks", mock_seed)

        db_initializer.initialize_database(mock_db_session, seed=False)

        mock_seed.assert_not_called()

    def test_defaults_to_seeding_when_seed_arg_omitted(
        self, mock_db_session, monkeypatch
    ):
        monkeypatch.setattr(db_initializer, "create_all_tables", MagicMock())
        mock_seed = MagicMock()
        monkeypatch.setattr(db_initializer, "seed_default_stocks", mock_seed)

        db_initializer.initialize_database(mock_db_session)

        mock_seed.assert_called_once_with(mock_db_session)
