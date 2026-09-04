from __future__ import annotations

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from stockmarketanalytics.data.app_db_context import get_db
from stockmarketanalytics.models.users import User
from stockmarketanalytics.services.auth_service import AuthError, AuthService

bearer_scheme = HTTPBearer(auto_error=True)
optional_bearer_scheme = HTTPBearer(auto_error=False)

DB_BEARER_DEPENDENCY = Depends(bearer_scheme)
DB_BEARER_OPTIONAL_DEPENDENCY = Depends(optional_bearer_scheme)
DB_DEPENDENCY = Depends(get_db)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = DB_BEARER_DEPENDENCY,
    db: Session = DB_DEPENDENCY,
) -> User:
    service = AuthService(db)
    try:
        return service.get_user_from_access_token(credentials.credentials)
    except AuthError as exc:
        raise HTTPException(
            status_code=401, detail=str(exc), headers={"WWW-Authenticate": "Bearer"}
        )


def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials | None = DB_BEARER_OPTIONAL_DEPENDENCY,
    db: Session = DB_DEPENDENCY,
) -> User | None:
    if credentials is None:
        return None
    service = AuthService(db)
    try:
        return service.get_user_from_access_token(credentials.credentials)
    except AuthError:
        return None
