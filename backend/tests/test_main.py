from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app(monkeypatch):
    """Import the FastAPI app with DB initialization mocked out, so the
    test suite never touches a real database on startup."""
    monkeypatch.setattr(
        "stockmarketanalytics.data.db_initializer.initialize_database",
        MagicMock(),
    )
    import stockmarketanalytics.main as main_module

    monkeypatch.setattr(main_module, "initialize_database", MagicMock())
    return main_module.app


@pytest.fixture
def client(app) -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


class TestHealthCheck:
    """Tests for the public /api/health endpoint."""

    def test_health_check_returns_200(self, client):
        response = client.get("/api/health")

        assert response.status_code == 200

    def test_health_check_returns_ok_status(self, client):
        response = client.get("/api/health")

        assert response.json() == {"status": "ok"}

    def test_health_check_does_not_require_authentication(self, client):
        response = client.get("/api/health")

        assert response.status_code == 200


class TestAppStartup:
    """Tests for the FastAPI app's startup lifecycle."""

    def test_startup_calls_initialize_database_with_seed_true(self, monkeypatch):
        mock_initialize = MagicMock()
        monkeypatch.setattr(
            "stockmarketanalytics.data.db_initializer.initialize_database",
            mock_initialize,
        )
        import stockmarketanalytics.main as main_module

        monkeypatch.setattr(main_module, "initialize_database", mock_initialize)
        monkeypatch.setattr(main_module, "SessionLocal", MagicMock())

        with TestClient(main_module.app):
            pass
        mock_initialize.assert_called_once()
        _, kwargs = mock_initialize.call_args
        assert kwargs.get("seed", True) is True

    def test_startup_closes_db_session_even_if_initialize_fails(self, monkeypatch):
        import stockmarketanalytics.main as main_module

        fake_session = MagicMock()
        fake_session_local = MagicMock(return_value=fake_session)
        monkeypatch.setattr(main_module, "SessionLocal", fake_session_local)
        monkeypatch.setattr(
            main_module,
            "initialize_database",
            MagicMock(side_effect=RuntimeError("boom")),
        )

        with pytest.raises(RuntimeError), TestClient(main_module.app):
            pass

        fake_session.close.assert_called_once()


class TestAppConfiguration:
    """Tests validating the app is wired up as expected (metadata,
    middleware, routers) without exercising real business logic."""

    def test_app_has_expected_title_and_version(self, app):
        assert app.title == "Stock Market Analytics API"
        assert app.version == "1.0.0"
