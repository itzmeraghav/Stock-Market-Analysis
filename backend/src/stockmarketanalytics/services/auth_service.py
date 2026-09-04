from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from sqlalchemy.orm import Session
from stockmarketanalytics.models.refresh_tokens import RefreshToken
from stockmarketanalytics.models.users import User
from stockmarketanalytics.settings import settings

logger = logging.getLogger("auth_service")

_password_hasher = PasswordHasher()


class AuthError(Exception):
    pass


class AccountLockedError(AuthError):
    def __init__(self, retry_after_seconds: int):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Account locked. Try again in {retry_after_seconds} seconds.")


class InvalidCredentialsError(AuthError):
    pass


class AuthService:
    def __init__(self, db: Session):
        self.db = db

    def hash_password(self, password: str) -> str:
        return _password_hasher.hash(password)

    def verify_password(self, password: str, hashed_password: str) -> bool:
        try:
            return _password_hasher.verify(hashed_password, password)
        except VerifyMismatchError:
            return False

    def register_user(self, username: str, password: str) -> User:
        existing = self.db.query(User).filter(User.username == username).first()
        if existing is not None:
            raise AuthError(f"Username already exists: {username}")

        user = User(username=username, hashed_password=self.hash_password(password))
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        logger.info("Registered new user: %s", username)
        return user

    def _is_locked(self, user: User) -> bool:
        return user.locked_until is not None and user.locked_until > datetime.now(UTC)

    def _register_failed_attempt(self, user: User) -> None:
        user.failed_login_attempts += 1

        if user.failed_login_attempts >= settings.login_max_failed_attempts:
            overflow = user.failed_login_attempts - settings.login_max_failed_attempts
            lockout_minutes = settings.login_lockout_minutes * (1 + overflow)
            user.locked_until = datetime.now(UTC) + timedelta(minutes=lockout_minutes)
            logger.warning(
                "Account locked for %s until %s (failed attempt #%d)",
                user.username,
                user.locked_until,
                user.failed_login_attempts,
            )

        self.db.commit()

    def _reset_failed_attempts(self, user: User) -> None:
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login_at = datetime.now(UTC)
        self.db.commit()

    def authenticate(self, username: str, password: str) -> User:
        user = self.db.query(User).filter(User.username == username).first()

        if user is None:
            raise InvalidCredentialsError("Invalid username or password")

        if self._is_locked(user):
            retry_after = int((user.locked_until - datetime.now(UTC)).total_seconds())
            raise AccountLockedError(max(retry_after, 1))

        if not self.verify_password(password, user.hashed_password):
            self._register_failed_attempt(user)
            raise InvalidCredentialsError("Invalid username or password")

        self._reset_failed_attempts(user)
        return user

    def create_access_token(self, user: User) -> str:
        expire = datetime.now(UTC) + timedelta(
            minutes=settings.access_token_expire_minutes
        )
        payload = {
            "sub": str(user.id),
            "username": user.username,
            "type": "access",
            "exp": expire,
        }
        return jwt.encode(
            payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
        )

    def _hash_token(self, token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_refresh_token(self, user: User) -> str:
        expire = datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days)
        jti = secrets.token_urlsafe(32)
        payload = {"sub": str(user.id), "type": "refresh", "jti": jti, "exp": expire}
        token = jwt.encode(
            payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
        )

        record = RefreshToken(
            user_id=user.id,
            token_hash=self._hash_token(token),
            expires_at=expire.replace(tzinfo=None),
        )
        self.db.add(record)
        self.db.commit()
        return token

    def issue_token_pair(self, user: User) -> tuple[str, str]:
        access_token = self.create_access_token(user)
        refresh_token = self.create_refresh_token(user)
        return access_token, refresh_token

    def decode_token(self, token: str) -> dict:
        try:
            return jwt.decode(
                token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
            )
        except jwt.ExpiredSignatureError:
            raise AuthError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthError("Invalid token")

    def rotate_refresh_token(self, refresh_token: str) -> tuple[str, str]:
        payload = self.decode_token(refresh_token)

        if payload.get("type") != "refresh":
            raise AuthError("Provided token is not a refresh token")

        token_hash = self._hash_token(refresh_token)
        record = (
            self.db.query(RefreshToken)
            .filter(RefreshToken.token_hash == token_hash)
            .first()
        )

        if record is None or record.revoked:
            raise AuthError("Refresh token has been revoked or does not exist")

        if record.expires_at < datetime.now(UTC):
            raise AuthError("Refresh token has expired")

        user = self.db.query(User).filter(User.id == record.user_id).first()
        if user is None:
            raise AuthError("User no longer exists")

        if user.last_refresh_at is not None:
            elapsed = (datetime.now(UTC) - user.last_refresh_at).total_seconds()
            if elapsed < settings.refresh_min_interval_seconds:
                wait = int(settings.refresh_min_interval_seconds - elapsed)
                raise AuthError(
                    f"Refresh requested too soon. Wait {wait} more seconds."
                )

        record.revoked = True
        record.last_used_at = datetime.now(UTC)
        user.last_refresh_at = datetime.now(UTC)
        self.db.commit()

        return self.issue_token_pair(user)

    def revoke_refresh_token(self, refresh_token: str) -> None:
        token_hash = self._hash_token(refresh_token)
        record = (
            self.db.query(RefreshToken)
            .filter(RefreshToken.token_hash == token_hash)
            .first()
        )
        if record is not None:
            record.revoked = True
            self.db.commit()

    def get_user_from_access_token(self, token: str) -> User:
        payload = self.decode_token(token)
        if payload.get("type") != "access":
            raise AuthError("Provided token is not an access token")

        user_id = payload.get("sub")
        user = self.db.query(User).filter(User.id == int(user_id)).first()
        if user is None or not user.is_active:
            raise AuthError("User not found or inactive")

        return user
