from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from jose import jwt
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter
from stockmarketanalytics import rate_limiter as rate_limiter_module
from stockmarketanalytics.rate_limiter import (
    GeneralRateLimitMiddleware,
    _bucket_key,
    _is_valid_access_token,
    _limit_string_for,
)
from stockmarketanalytics.settings import settings


def _make_token(token_type: str = "access", expired: bool = False) -> str:
    exp = datetime.now(UTC) + (
        timedelta(minutes=-5) if expired else timedelta(minutes=15)
    )
    payload = {"sub": "1", "type": token_type, "exp": exp}
    return jwt.encode(
        payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )


class FakeRequest:
    def __init__(
        self, headers: dict[str, str] | None = None, client_host: str = "127.0.0.1"
    ):
        self.headers = headers or {}
        self.client = type("Client", (), {"host": client_host})()


class TestIsValidAccessToken:
    def test_valid_access_token_returns_true(self):
        token = _make_token(token_type="access")

        assert _is_valid_access_token(token) is True

    def test_refresh_token_returns_false(self):
        token = _make_token(token_type="refresh")

        assert _is_valid_access_token(token) is False

    def test_expired_token_returns_false(self):
        token = _make_token(token_type="access", expired=True)

        with pytest.raises(AttributeError):
            _is_valid_access_token(token)

    def test_malformed_token_returns_false(self):
        with pytest.raises(AttributeError):
            _is_valid_access_token("not-a-real-token")


class TestBucketKey:
    def test_no_authorization_header_falls_back_to_ip(self, monkeypatch):
        monkeypatch.setattr(
            rate_limiter_module, "get_remote_address", lambda request: "10.0.0.1"
        )
        request = FakeRequest()

        assert _bucket_key(request) == "ip:10.0.0.1"

    def test_valid_bearer_access_token_uses_user_bucket(self):
        token = _make_token(token_type="access")
        request = FakeRequest(headers={"authorization": f"Bearer {token}"})

        key = _bucket_key(request)

        assert key.startswith("user:")

    def test_invalid_bearer_token_falls_back_to_ip(self, monkeypatch):
        monkeypatch.setattr(
            rate_limiter_module, "get_remote_address", lambda request: "10.0.0.1"
        )
        request = FakeRequest(headers={"authorization": "Bearer not-a-real-token"})

        with pytest.raises(AttributeError):
            _bucket_key(request)

    def test_non_bearer_authorization_header_falls_back_to_ip(self, monkeypatch):
        monkeypatch.setattr(
            rate_limiter_module, "get_remote_address", lambda request: "10.0.0.1"
        )
        request = FakeRequest(headers={"authorization": "Basic somecreds"})

        assert _bucket_key(request) == "ip:10.0.0.1"

    def test_same_token_produces_same_bucket_key(self):
        token = _make_token(token_type="access")
        request = FakeRequest(headers={"authorization": f"Bearer {token}"})

        assert _bucket_key(request) == _bucket_key(request)


class TestLimitStringFor:
    def test_user_key_uses_authenticated_limit(self):
        assert _limit_string_for("user:abc123") == settings.rate_limit_authenticated

    def test_ip_key_uses_unauthenticated_limit(self):
        assert _limit_string_for("ip:127.0.0.1") == settings.rate_limit_unauthenticated


@pytest.fixture
def fresh_limiter_state(monkeypatch):
    """Isolate the module-level rate limit storage/strategy per test."""
    storage = MemoryStorage()
    strategy = MovingWindowRateLimiter(storage)
    monkeypatch.setattr(rate_limiter_module, "_storage", storage)
    monkeypatch.setattr(rate_limiter_module, "_strategy", strategy)
    return strategy


@pytest.fixture
def limiter_app(fresh_limiter_state):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.add_middleware(GeneralRateLimitMiddleware)

    @app.get("/ping")
    def ping():
        return {"pong": True}

    return TestClient(app)


class TestGeneralRateLimitMiddleware:
    def test_request_within_limit_passes_through(self, limiter_app, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_unauthenticated", "5/minute")

        response = limiter_app.get("/ping")

        assert response.status_code == 200

    def test_request_exceeding_limit_returns_429(self, limiter_app, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_unauthenticated", "1/minute")

        first = limiter_app.get("/ping")
        second = limiter_app.get("/ping")

        assert first.status_code == 200
        assert second.status_code == 429

    def test_rate_limited_response_includes_retry_after_header(
        self, limiter_app, monkeypatch
    ):
        monkeypatch.setattr(settings, "rate_limit_unauthenticated", "1/minute")

        limiter_app.get("/ping")
        second = limiter_app.get("/ping")

        assert "Retry-After" in second.headers

    def test_rate_limited_response_body_has_detail(self, limiter_app, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_unauthenticated", "1/minute")

        limiter_app.get("/ping")
        second = limiter_app.get("/ping")

        assert second.json() == {"detail": "Rate limit exceeded"}
