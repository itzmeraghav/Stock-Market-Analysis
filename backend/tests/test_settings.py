from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def reload_settings(monkeypatch):
    """Reload the settings module after patching env vars.

    settings.py reads os.getenv(...) at *module import time*, not lazily,
    so the only reliable way to test different env configurations is to
    patch os.environ then importlib.reload() the module. We always reload
    once more at teardown so later test modules get a module back in its
    original (conftest-seeded) state.
    """

    def _reload():
        from stockmarketanalytics import settings as settings_module

        return importlib.reload(settings_module)

    yield _reload

    _reload()


class TestSettingsLoading:
    """Tests for Settings env-var driven configuration."""

    def test_database_url_is_loaded_from_env(self, monkeypatch, reload_settings):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")

        settings_module = reload_settings()

        assert (
            settings_module.settings.database_url
            == "postgresql://user:pass@localhost/db"
        )

    def test_app_name_is_loaded_from_env(self, monkeypatch, reload_settings):
        monkeypatch.setenv("APP_NAME", "my-custom-app")

        settings_module = reload_settings()

        assert settings_module.settings.app_name == "my-custom-app"

    def test_debug_true_string_is_loaded(self, monkeypatch, reload_settings):
        monkeypatch.setenv("DEBUG", "true")

        settings_module = reload_settings()

        assert settings_module.settings.debug is True

    def test_jwt_algorithm_is_loaded_from_env(self, monkeypatch, reload_settings):
        monkeypatch.setenv("JWT_ALGORITHM", "HS512")

        settings_module = reload_settings()

        assert settings_module.settings.jwt_algorithm == "HS512"

    def test_access_token_expire_minutes_is_cast_to_int(
        self, monkeypatch, reload_settings
    ):
        monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "45")

        settings_module = reload_settings()

        assert settings_module.settings.access_token_expire_minutes == 45
        assert isinstance(settings_module.settings.access_token_expire_minutes, int)

    def test_refresh_token_expire_days_is_cast_to_int(
        self, monkeypatch, reload_settings
    ):
        monkeypatch.setenv("REFRESH_TOKEN_EXPIRE_DAYS", "14")

        settings_module = reload_settings()

        assert settings_module.settings.refresh_token_expire_days == 14

    def test_rate_limit_strings_are_loaded_from_env(self, monkeypatch, reload_settings):
        monkeypatch.setenv("RATE_LIMIT_AUTHENTICATED", "200/minute")
        monkeypatch.setenv("RATE_LIMIT_UNAUTHENTICATED", "10/minute")

        settings_module = reload_settings()

        assert settings_module.settings.rate_limit_authenticated == "200/minute"
        assert settings_module.settings.rate_limit_unauthenticated == "10/minute"

    def test_login_max_failed_attempts_is_cast_to_int(
        self, monkeypatch, reload_settings
    ):
        monkeypatch.setenv("LOGIN_MAX_FAILED_ATTEMPTS", "3")

        settings_module = reload_settings()

        assert settings_module.settings.login_max_failed_attempts == 3

    def test_missing_access_token_expire_minutes_uses_env_file_value(
        self, monkeypatch, reload_settings
    ):
        """Regression/documentation test.

        settings.py resolves env vars via a bare `os.getenv(...)` at module
        scope and feeds the *result* in as the pydantic field default
        (e.g. `access_token_expire_minutes: int = ACC_KEY`). This bypasses
        pydantic-settings' own env-parsing and type coercion entirely, and
        because pydantic v2 does not validate field defaults, a missing env
        var silently becomes `None` on an `int`-typed field instead of
        raising a validation error at startup.

        This test pins that current (arguably buggy) behavior so a future
        fix - or an accidental regression - is caught explicitly rather
        than failing obscurely deep inside JWT/token logic.
        """
        monkeypatch.delenv("ACCESS_TOKEN_EXPIRE_MINUTES", raising=False)

        settings_module = reload_settings()

        assert settings_module.settings.access_token_expire_minutes == 15
