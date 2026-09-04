from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from stockmarketanalytics.data.app_db_context import get_db
from stockmarketanalytics.rate_limiter import limiter
from stockmarketanalytics.schemas.auth_schemas import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserOut,
)
from stockmarketanalytics.services.auth_service import (
    AccountLockedError,
    AuthError,
    AuthService,
    InvalidCredentialsError,
)
from stockmarketanalytics.settings import settings

router = APIRouter(prefix="/auth", tags=["auth"])


DB_DEPENDENCY = Depends(get_db)


@router.post("/register", response_model=UserOut)
def register(request: RegisterRequest, db: Session = DB_DEPENDENCY):
    service = AuthService(db)
    try:
        user = service.register_user(request.username, request.password)
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return UserOut(id=user.id, username=user.username)


@router.post("/login", response_model=TokenResponse)
@limiter.limit(settings.login_rate_limit_per_ip)
def login(request: Request, body: LoginRequest, db: Session = DB_DEPENDENCY):
    service = AuthService(db)
    try:
        user = service.authenticate(body.username, body.password)
    except AccountLockedError as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )
    except InvalidCredentialsError as exc:
        raise HTTPException(status_code=401, detail=str(exc))

    access_token, refresh_token = service.issue_token_pair(user)
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in_minutes=settings.access_token_expire_minutes,
    )


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit(settings.token_global_cap_per_hour)
def refresh(request: Request, body: RefreshRequest, db: Session = DB_DEPENDENCY):
    service = AuthService(db)
    try:
        access_token, refresh_token = service.rotate_refresh_token(body.refresh_token)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc))

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in_minutes=settings.access_token_expire_minutes,
    )


@router.post("/logout")
def logout(body: RefreshRequest, db: Session = DB_DEPENDENCY):
    service = AuthService(db)
    service.revoke_refresh_token(body.refresh_token)
    return {"detail": "Logged out successfully"}
