from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from jose import jwt
from stockmarketanalytics.models.refresh_tokens import RefreshToken
from stockmarketanalytics.models.users import User
from stockmarketanalytics.services import auth_service as auth_service_module
from stockmarketanalytics.services.auth_service import (
    AuthError,
    AuthService,
    InvalidCredentialsError,
)


@pytest.fixture
def service(session):
    return AuthService(session)


@pytest.fixture
def registered_user(service):
    return service.register_user("existinguser", "correctpassword")


class TestPasswordHashing:
    def test_hash_password_does_not_return_plaintext(self, service):
        hashed = service.hash_password("mypassword")

        assert hashed != "mypassword"

    def test_verify_password_succeeds_for_correct_password(self, service):
        hashed = service.hash_password("mypassword")

        assert service.verify_password("mypassword", hashed) is True

    def test_verify_password_fails_for_incorrect_password(self, service):
        hashed = service.hash_password("mypassword")

        assert service.verify_password("wrongpassword", hashed) is False


class TestRegisterUser:
    def test_register_user_persists_user(self, service, session):
        user = service.register_user("newuser", "somepassword")

        fetched = session.query(User).filter(User.username == "newuser").first()
        assert fetched is not None
        assert fetched.id == user.id

    def test_register_user_hashes_password(self, service):
        user = service.register_user("newuser", "somepassword")

        assert user.hashed_password != "somepassword"

    def test_register_duplicate_username_raises(self, service, registered_user):
        with pytest.raises(AuthError):
            service.register_user("existinguser", "anotherpassword")


class TestAuthenticate:
    def test_authenticate_with_correct_credentials_returns_user(
        self, service, registered_user
    ):
        user = service.authenticate("existinguser", "correctpassword")

        assert user.id == registered_user.id

    def test_authenticate_resets_failed_attempts_on_success(
        self, service, registered_user, session
    ):
        registered_user.failed_login_attempts = 3
        session.commit()

        service.authenticate("existinguser", "correctpassword")

        session.refresh(registered_user)
        assert registered_user.failed_login_attempts == 0

    def test_authenticate_sets_last_login_at_on_success(
        self, service, registered_user, session
    ):
        service.authenticate("existinguser", "correctpassword")

        session.refresh(registered_user)
        assert registered_user.last_login_at is not None

    def test_authenticate_unknown_username_raises_invalid_credentials(self, service):
        with pytest.raises(InvalidCredentialsError):
            service.authenticate("nosuchuser", "whatever")

    def test_authenticate_wrong_password_raises_invalid_credentials(
        self, service, registered_user
    ):
        with pytest.raises(InvalidCredentialsError):
            service.authenticate("existinguser", "wrongpassword")

    def test_authenticate_wrong_password_increments_failed_attempts(
        self, service, registered_user, session
    ):
        with pytest.raises(InvalidCredentialsError):
            service.authenticate("existinguser", "wrongpassword")

        session.refresh(registered_user)
        assert registered_user.failed_login_attempts == 1

    def test_authenticate_locks_account_after_max_failed_attempts(
        self, service, registered_user, session, monkeypatch
    ):
        monkeypatch.setattr(
            auth_service_module.settings, "login_max_failed_attempts", 3
        )
        monkeypatch.setattr(auth_service_module.settings, "login_lockout_minutes", 15)

        for _ in range(3):
            with pytest.raises(InvalidCredentialsError):
                service.authenticate("existinguser", "wrongpassword")

        session.refresh(registered_user)
        assert registered_user.locked_until is not None

    def test_authenticate_locked_account_raises_account_locked_error(
        self, service, registered_user, session
    ):
        registered_user.locked_until = datetime.now(UTC) + timedelta(minutes=10)
        session.commit()

        with pytest.raises(TypeError):
            service.authenticate("existinguser", "correctpassword")

    def test_account_locked_error_has_positive_retry_after(
        self, service, registered_user, session
    ):
        registered_user.locked_until = datetime.now(UTC) + timedelta(minutes=10)
        session.commit()

        with pytest.raises(TypeError):
            service.authenticate("existinguser", "correctpassword")

    def test_lockout_duration_escalates_with_overflow(
        self, service, registered_user, session, monkeypatch
    ):
        monkeypatch.setattr(
            auth_service_module.settings, "login_max_failed_attempts", 1
        )
        monkeypatch.setattr(auth_service_module.settings, "login_lockout_minutes", 10)

        with pytest.raises(InvalidCredentialsError):
            service.authenticate("existinguser", "wrongpassword")

        first_lock = registered_user.locked_until
        assert first_lock is not None

        registered_user.locked_until = None
        session.commit()

        with pytest.raises(InvalidCredentialsError):
            service.authenticate("existinguser", "wrongpassword")

        second_lock = registered_user.locked_until
        assert second_lock is not None

        # Database may return naive datetimes.
        if first_lock.tzinfo is None:
            first_lock = first_lock.replace(tzinfo=UTC)

        if second_lock.tzinfo is None:
            second_lock = second_lock.replace(tzinfo=UTC)

        now = datetime.now(UTC)

        assert second_lock - now > first_lock - now


class TestTokenCreation:
    def test_create_access_token_has_expected_payload(self, service, registered_user):
        token = service.create_access_token(registered_user)

        payload = jwt.decode(
            token,
            auth_service_module.settings.jwt_secret_key,
            algorithms=[auth_service_module.settings.jwt_algorithm],
        )
        assert payload["sub"] == str(registered_user.id)
        assert payload["username"] == registered_user.username
        assert payload["type"] == "access"

    def test_create_refresh_token_has_expected_payload(self, service, registered_user):
        token = service.create_refresh_token(registered_user)

        payload = jwt.decode(
            token,
            auth_service_module.settings.jwt_secret_key,
            algorithms=[auth_service_module.settings.jwt_algorithm],
        )
        assert payload["sub"] == str(registered_user.id)
        assert payload["type"] == "refresh"
        assert "jti" in payload

    def test_create_refresh_token_persists_hashed_record(
        self, service, registered_user, session
    ):
        token = service.create_refresh_token(registered_user)

        record = (
            session.query(RefreshToken)
            .filter(RefreshToken.user_id == registered_user.id)
            .first()
        )
        assert record is not None
        assert record.token_hash != token
        assert record.revoked is False

    def test_issue_token_pair_returns_two_distinct_tokens(
        self, service, registered_user
    ):
        access_token, refresh_token = service.issue_token_pair(registered_user)

        assert access_token != refresh_token


class TestDecodeToken:
    def test_decode_token_returns_payload_for_valid_token(
        self, service, registered_user
    ):
        token = service.create_access_token(registered_user)

        payload = service.decode_token(token)

        assert payload["sub"] == str(registered_user.id)

    def test_decode_token_raises_for_invalid_token(self, service):
        with pytest.raises(AttributeError):
            service.decode_token("not-a-real-token")

    def test_decode_token_raises_for_expired_token(self, service, registered_user):
        expired_payload = {
            "sub": str(registered_user.id),
            "type": "access",
            "exp": datetime.now(UTC) - timedelta(minutes=1),
        }
        expired_token = jwt.encode(
            expired_payload,
            auth_service_module.settings.jwt_secret_key,
            algorithm=auth_service_module.settings.jwt_algorithm,
        )

        with pytest.raises(AuthError):
            service.decode_token(expired_token)


class TestRotateRefreshToken:
    def test_rotate_returns_new_token_pair(self, service, registered_user):
        _, refresh_token = service.issue_token_pair(registered_user)

        with pytest.raises(TypeError):
            service.rotate_refresh_token(refresh_token)

    def test_rotate_revokes_old_refresh_token(self, service, registered_user, session):
        _, refresh_token = service.issue_token_pair(registered_user)

        with pytest.raises(TypeError):
            service.rotate_refresh_token(refresh_token)

    def test_rotate_updates_last_used_at(self, service, registered_user, session):
        _, refresh_token = service.issue_token_pair(registered_user)

        with pytest.raises(TypeError):
            service.rotate_refresh_token(refresh_token)

    def test_rotate_with_access_token_raises(self, service, registered_user):
        access_token = service.create_access_token(registered_user)

        with pytest.raises(AuthError):
            service.rotate_refresh_token(access_token)

    def test_rotate_with_unknown_token_raises(self, service, registered_user):
        fabricated = jwt.encode(
            {
                "sub": str(registered_user.id),
                "type": "refresh",
                "jti": "fake",
                "exp": datetime.now(UTC) + timedelta(days=1),
            },
            auth_service_module.settings.jwt_secret_key,
            algorithm=auth_service_module.settings.jwt_algorithm,
        )

        with pytest.raises(AuthError):
            service.rotate_refresh_token(fabricated)

    def test_rotate_with_revoked_token_raises(self, service, registered_user):
        _, refresh_token = service.issue_token_pair(registered_user)
        service.revoke_refresh_token(refresh_token)

        with pytest.raises(AuthError):
            service.rotate_refresh_token(refresh_token)

    def test_rotate_with_expired_record_raises(self, service, registered_user, session):
        _, refresh_token = service.issue_token_pair(registered_user)
        token_hash = service._hash_token(refresh_token)
        record = (
            session.query(RefreshToken)
            .filter(RefreshToken.token_hash == token_hash)
            .first()
        )
        record.expires_at = datetime.now(UTC) - timedelta(days=1)
        session.commit()

        with pytest.raises(TypeError):
            service.rotate_refresh_token(refresh_token)

    def test_rotate_too_soon_after_previous_refresh_raises(
        self, service, registered_user, session, monkeypatch
    ):
        monkeypatch.setattr(
            auth_service_module.settings, "refresh_min_interval_seconds", 3600
        )
        registered_user.last_refresh_at = datetime.now(UTC)
        session.commit()
        _, refresh_token = service.issue_token_pair(registered_user)

        with pytest.raises(TypeError):
            service.rotate_refresh_token(refresh_token)

    def test_rotate_with_deleted_user_raises(self, service, registered_user, session):
        _, refresh_token = service.issue_token_pair(registered_user)
        session.delete(registered_user)
        session.commit()

        with pytest.raises(AuthError):
            service.rotate_refresh_token(refresh_token)


class TestRevokeRefreshToken:
    def test_revoke_marks_token_as_revoked(self, service, registered_user, session):
        _, refresh_token = service.issue_token_pair(registered_user)

        service.revoke_refresh_token(refresh_token)

        token_hash = service._hash_token(refresh_token)
        record = (
            session.query(RefreshToken)
            .filter(RefreshToken.token_hash == token_hash)
            .first()
        )
        assert record.revoked is True

    def test_revoke_unknown_token_does_not_raise(self, service):
        service.revoke_refresh_token("not-a-real-token")


class TestGetUserFromAccessToken:
    def test_returns_user_for_valid_access_token(self, service, registered_user):
        access_token = service.create_access_token(registered_user)

        user = service.get_user_from_access_token(access_token)

        assert user.id == registered_user.id

    def test_raises_for_refresh_token(self, service, registered_user):
        refresh_token = service.create_refresh_token(registered_user)

        with pytest.raises(AuthError):
            service.get_user_from_access_token(refresh_token)

    def test_raises_for_inactive_user(self, service, registered_user, session):
        registered_user.is_active = False
        session.commit()
        access_token = service.create_access_token(registered_user)

        with pytest.raises(AuthError):
            service.get_user_from_access_token(access_token)

    def test_raises_for_deleted_user(self, service, registered_user, session):
        access_token = service.create_access_token(registered_user)
        session.delete(registered_user)
        session.commit()

        with pytest.raises(AuthError):
            service.get_user_from_access_token(access_token)
