from __future__ import annotations

from types import SimpleNamespace

import pytest
from stockmarketanalytics.services.auth_service import (
    AccountLockedError,
    AuthError,
    AuthService,
    InvalidCredentialsError,
)


@pytest.fixture
def auth_app(mock_db_session):
    from fastapi import FastAPI
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from stockmarketanalytics.data.app_db_context import get_db
    from stockmarketanalytics.endpoints import auth_endpoints
    from stockmarketanalytics.rate_limiter import limiter

    test_app = FastAPI()
    test_app.state.limiter = limiter
    test_app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    test_app.include_router(auth_endpoints.router)
    test_app.dependency_overrides[get_db] = lambda: mock_db_session
    return test_app


@pytest.fixture
def auth_client(auth_app):
    from fastapi.testclient import TestClient

    return TestClient(auth_app)


@pytest.fixture
def fake_user():
    return SimpleNamespace(id=1, username="someone")


class TestRegisterEndpoint:
    def test_register_returns_201_payload_on_success(
        self, auth_client, monkeypatch, fake_user
    ):
        monkeypatch.setattr(
            AuthService, "register_user", lambda self, username, password: fake_user
        )

        response = auth_client.post(
            "/auth/register", json={"username": "someone", "password": "longenoughpw"}
        )

        assert response.status_code == 200
        assert response.json() == {"id": 1, "username": "someone"}

    def test_register_returns_400_when_username_taken(self, auth_client, monkeypatch):
        def _raise(self, username, password):
            raise AuthError(f"Username already exists: {username}")

        monkeypatch.setattr(AuthService, "register_user", _raise)

        response = auth_client.post(
            "/auth/register", json={"username": "someone", "password": "longenoughpw"}
        )

        assert response.status_code == 400

    def test_register_returns_422_for_short_password(self, auth_client):
        response = auth_client.post(
            "/auth/register", json={"username": "someone", "password": "short"}
        )

        assert response.status_code == 422

    def test_register_returns_422_for_missing_username(self, auth_client):
        response = auth_client.post("/auth/register", json={"password": "longenoughpw"})

        assert response.status_code == 422


class TestLoginEndpoint:
    def test_login_returns_token_pair_on_success(
        self, auth_client, monkeypatch, fake_user
    ):
        monkeypatch.setattr(
            AuthService, "authenticate", lambda self, username, password: fake_user
        )
        monkeypatch.setattr(
            AuthService,
            "issue_token_pair",
            lambda self, user: ("access-token", "refresh-token"),
        )

        response = auth_client.post(
            "/auth/login", json={"username": "someone", "password": "correctpassword"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["access_token"] == "access-token"
        assert body["refresh_token"] == "refresh-token"

    def test_login_returns_401_for_invalid_credentials(self, auth_client, monkeypatch):
        def _raise(self, username, password):
            raise InvalidCredentialsError("Invalid username or password")

        monkeypatch.setattr(AuthService, "authenticate", _raise)

        response = auth_client.post(
            "/auth/login", json={"username": "someone", "password": "wrongpassword"}
        )

        assert response.status_code == 401

    def test_login_returns_429_for_locked_account(self, auth_client, monkeypatch):
        def _raise(self, username, password):
            raise AccountLockedError(retry_after_seconds=60)

        monkeypatch.setattr(AuthService, "authenticate", _raise)

        response = auth_client.post(
            "/auth/login", json={"username": "someone", "password": "correctpassword"}
        )

        assert response.status_code == 429
        assert response.headers["Retry-After"] == "60"

    def test_login_returns_422_for_missing_password(self, auth_client):
        response = auth_client.post("/auth/login", json={"username": "someone"})

        assert response.status_code == 422


class TestRefreshEndpoint:
    def test_refresh_returns_new_token_pair_on_success(self, auth_client, monkeypatch):
        monkeypatch.setattr(
            AuthService,
            "rotate_refresh_token",
            lambda self, refresh_token: ("new-access-token", "new-refresh-token"),
        )

        response = auth_client.post(
            "/auth/refresh", json={"refresh_token": "old-refresh-token"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["access_token"] == "new-access-token"
        assert body["refresh_token"] == "new-refresh-token"

    def test_refresh_returns_401_for_invalid_token(self, auth_client, monkeypatch):
        def _raise(self, refresh_token):
            raise AuthError("Refresh token has been revoked or does not exist")

        monkeypatch.setattr(AuthService, "rotate_refresh_token", _raise)

        response = auth_client.post(
            "/auth/refresh", json={"refresh_token": "bad-token"}
        )

        assert response.status_code == 401

    def test_refresh_returns_422_for_missing_token(self, auth_client):
        response = auth_client.post("/auth/refresh", json={})

        assert response.status_code == 422


class TestLogoutEndpoint:
    def test_logout_returns_success_message(self, auth_client, monkeypatch):
        monkeypatch.setattr(
            AuthService, "revoke_refresh_token", lambda self, refresh_token: None
        )

        response = auth_client.post("/auth/logout", json={"refresh_token": "sometoken"})

        assert response.status_code == 200
        assert response.json() == {"detail": "Logged out successfully"}

    def test_logout_returns_422_for_missing_token(self, auth_client):
        response = auth_client.post("/auth/logout", json={})

        assert response.status_code == 422
