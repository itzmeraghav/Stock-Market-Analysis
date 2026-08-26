from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session
from stockmarketanalytics.data import app_db_context


class TestEngineConfiguration:
    """Tests for the module-level SQLAlchemy engine/session factory."""

    def test_session_local_is_bound_to_the_module_engine(self):
        assert app_db_context.SessionLocal.kw["bind"] is app_db_context.engine

    def test_session_local_does_not_autoflush(self):
        assert app_db_context.SessionLocal.kw["autoflush"] is False

    def test_session_local_does_not_autocommit(self):
        assert app_db_context.SessionLocal.kw["autocommit"] is False

    def test_session_local_expire_on_commit_is_disabled(self):
        """expire_on_commit=False means objects stay usable (e.g. across a
        FastAPI response serialization step) after commit without triggering
        a fresh DB round-trip / DetachedInstanceError."""
        assert app_db_context.SessionLocal.kw["expire_on_commit"] is False


class TestGetDb:
    """Tests for the get_db FastAPI dependency generator."""

    def test_get_db_yields_a_session_instance(self, monkeypatch):
        fake_session = MagicMock(spec=Session)
        fake_session_local = MagicMock(return_value=fake_session)
        monkeypatch.setattr(app_db_context, "SessionLocal", fake_session_local)

        generator = app_db_context.get_db()
        yielded_session = next(generator)

        assert yielded_session is fake_session
        fake_session.close.assert_not_called()

        with pytest.raises(StopIteration):
            next(generator)

    def test_get_db_closes_session_after_use(self, monkeypatch):
        fake_session = MagicMock(spec=Session)
        fake_session_local = MagicMock(return_value=fake_session)
        monkeypatch.setattr(app_db_context, "SessionLocal", fake_session_local)

        generator = app_db_context.get_db()
        next(generator)
        with pytest.raises(StopIteration):
            next(generator)

        fake_session.close.assert_called_once()

    def test_get_db_closes_session_even_if_consumer_raises(self, monkeypatch):
        """Mirrors how FastAPI drives the generator: it throws the request
        handler's exception into the generator via .throw(), and the
        dependency's `finally` block must still close the session."""
        fake_session = MagicMock(spec=Session)
        fake_session_local = MagicMock(return_value=fake_session)
        monkeypatch.setattr(app_db_context, "SessionLocal", fake_session_local)

        generator = app_db_context.get_db()
        next(generator)

        with pytest.raises(ValueError):
            generator.throw(ValueError("simulated handler failure"))

        fake_session.close.assert_called_once()

    def test_get_db_creates_a_new_session_per_call(self, monkeypatch):

        fake_session_local = MagicMock(side_effect=lambda: MagicMock(spec=Session))
        monkeypatch.setattr(app_db_context, "SessionLocal", fake_session_local)

        first_session = next(app_db_context.get_db())
        second_session = next(app_db_context.get_db())

        assert first_session is not second_session
        assert fake_session_local.call_count == 2
