from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from stockmarketanalytics.auth_dependencies import (
    get_current_user,
    get_current_user_optional,
)
from stockmarketanalytics.services.auth_service import AuthError, AuthService


@pytest.fixture
def credentials():
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials="sometoken")


class TestGetCurrentUser:
    def test_returns_user_for_valid_token(
        self, credentials, mock_db_session, monkeypatch, fake_stock
    ):
        monkeypatch.setattr(
            AuthService, "get_user_from_access_token", lambda self, token: fake_stock
        )

        result = get_current_user(credentials=credentials, db=mock_db_session)

        assert result is fake_stock

    def test_raises_401_for_invalid_token(
        self, credentials, mock_db_session, monkeypatch
    ):
        def _raise(self, token):
            raise AuthError("Invalid token")

        monkeypatch.setattr(AuthService, "get_user_from_access_token", _raise)

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(credentials=credentials, db=mock_db_session)

        assert exc_info.value.status_code == 401

    def test_401_response_includes_www_authenticate_header(
        self, credentials, mock_db_session, monkeypatch
    ):
        def _raise(self, token):
            raise AuthError("Invalid token")

        monkeypatch.setattr(AuthService, "get_user_from_access_token", _raise)

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(credentials=credentials, db=mock_db_session)

        assert exc_info.value.headers["WWW-Authenticate"] == "Bearer"


class TestGetCurrentUserOptional:
    def test_returns_none_when_no_credentials_supplied(self, mock_db_session):
        result = get_current_user_optional(credentials=None, db=mock_db_session)

        assert result is None

    def test_returns_user_for_valid_token(
        self, credentials, mock_db_session, monkeypatch, fake_stock
    ):
        monkeypatch.setattr(
            AuthService, "get_user_from_access_token", lambda self, token: fake_stock
        )

        result = get_current_user_optional(credentials=credentials, db=mock_db_session)

        assert result is fake_stock

    def test_returns_none_for_invalid_token_instead_of_raising(
        self, credentials, mock_db_session, monkeypatch
    ):
        def _raise(self, token):
            raise AuthError("Invalid token")

        monkeypatch.setattr(AuthService, "get_user_from_access_token", _raise)

        result = get_current_user_optional(credentials=credentials, db=mock_db_session)

        assert result is None
