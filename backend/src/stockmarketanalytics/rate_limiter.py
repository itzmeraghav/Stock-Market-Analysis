from __future__ import annotations

import hashlib

from fastapi import Request, Response
from jose import jwt
from limits import parse_many
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from stockmarketanalytics.settings import settings

# Used for per-route decorators on the auth endpoints (login, refresh) —
# these have their own fixed limit strings and are confirmed to work
# correctly through slowapi's standard decorator path.
limiter = Limiter(key_func=get_remote_address)

# Used for the global per-request authenticated/unauthenticated split,
# applied to every route automatically. Built directly on the `limits`
# library slowapi wraps, because slowapi's own `default_limits` does not
# support a per-request dynamic (callable) limit value — verified against
# the installed slowapi source; only per-route `@limiter.limit()` decorators
# get request-aware dynamic limits, not the global `default_limits` list.
_storage = MemoryStorage()
_strategy = MovingWindowRateLimiter(_storage)


def _is_valid_access_token(token: str) -> bool:
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
    except jwt.InvalidTokenError:
        return False
    return payload.get("type") == "access"


def _bucket_key(request: Request) -> str:
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        if _is_valid_access_token(token):
            token_fingerprint = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
            return f"user:{token_fingerprint}"
    return f"ip:{get_remote_address(request)}"


def _limit_string_for(key: str) -> str:
    if key.startswith("user:"):
        return settings.rate_limit_authenticated
    return settings.rate_limit_unauthenticated


class GeneralRateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        key = _bucket_key(request)
        limit_string = _limit_string_for(key)

        for limit_item in parse_many(limit_string):
            if not _strategy.hit(limit_item, key):
                return Response(
                    content='{"detail":"Rate limit exceeded"}',
                    status_code=429,
                    media_type="application/json",
                    headers={"Retry-After": str(limit_item.get_expiry())},
                )

        return await call_next(request)
