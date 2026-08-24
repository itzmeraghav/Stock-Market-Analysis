import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
APP_NAME = os.getenv("APP_NAME")
DE = os.getenv("DEBUG")
JWT_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALOG = os.getenv("JWT_ALGORITHM")
ACC_KEY = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")
REF_KEY = os.getenv("REFRESH_TOKEN_EXPIRE_DAYS")
LOG_RATE = os.getenv("LOGIN_RATE_LIMIT_PER_IP")
LOG_MAX = os.getenv("LOGIN_MAX_FAILED_ATTEMPTS")
LOG_OUT = os.getenv("LOGIN_LOCKOUT_MINUTES")
REF_SEC = os.getenv("REFRESH_MIN_INTERVAL_SECONDS")
TOK_GLO = os.getenv("TOKEN_GLOBAL_CAP_PER_HOUR")
RAT_AUTH = os.getenv("RATE_LIMIT_AUTHENTICATED")
RAT_UNAUTH = os.getenv("RATE_LIMIT_UNAUTHENTICATED")


class Settings(BaseSettings):
    database_url: str = DB_URL

    app_name: str = APP_NAME

    debug: bool = DE

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    jwt_secret_key: str = JWT_KEY
    jwt_algorithm: str = JWT_ALOG
    access_token_expire_minutes: int = ACC_KEY
    refresh_token_expire_days: int = REF_KEY

    login_rate_limit_per_ip: str = LOG_RATE
    login_max_failed_attempts: int = LOG_MAX
    login_lockout_minutes: int = LOG_OUT
    refresh_min_interval_seconds: int = REF_SEC
    token_global_cap_per_hour: str = TOK_GLO

    rate_limit_authenticated: str = RAT_AUTH
    rate_limit_unauthenticated: str = RAT_UNAUTH


settings = Settings()
