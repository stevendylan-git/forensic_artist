from typing import List, Optional
from fastapi import Depends, HTTPException, status, Header, Path
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.config import settings
from app.core.security import decode_token
from app.models.user import User, UserRole
from app.models.case import Case

bearer_security = HTTPBearer(auto_error=False)


async def get_current_user(
    auth_credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Extracts, validates JWT access token, and fetches active User.
    """
    if not auth_credentials or not auth_credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = auth_credentials.credentials
    try:
        payload = decode_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user_id: str = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload missing subject identifier",
            headers={"WWW-Authenticate": "Bearer"}
        )

    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )

    return user


def require_roles(*allowed_roles: UserRole):
    """Dependency enforcing role-based access control."""
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        user_role = current_user.role
        if isinstance(user_role, str):
            try:
                user_role = UserRole(user_role)
            except ValueError:
                pass

        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: role '{current_user.role}' lacks required access"
            )
        return current_user

    return role_checker


async def get_case_for_user(
    case_id: str = Path(..., description="Case UUID"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Case:
    """
    Case-level need-to-know authorization:
    - Admin and Supervisor have global case access.
    - Investigator and ForensicArtist must be assigned to the case.
    - Auditor can read case metadata for audit compliance, but has NO face image access.
    """
    stmt = (
        select(Case)
        .where(Case.id == case_id)
        .options(selectinload(Case.assignments))
    )
    result = await db.execute(stmt)
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case with ID {case_id} not found"
        )

    user_role = current_user.role
    if isinstance(user_role, str):
        user_role = UserRole(user_role)

    # Admin and Supervisor can access any case
    if user_role in [UserRole.ADMIN, UserRole.SUPERVISOR]:
        return case

    # Auditor can access case metadata
    if user_role == UserRole.AUDITOR:
        return case

    # Investigator & ForensicArtist need explicit assignment
    assigned_ids = [a.user_id for a in case.assignments]
    if current_user.id not in assigned_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Need-to-know violation: You are not assigned to this case."
        )

    return case


def require_face_composite_access(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Enforces that Auditor role has STRICTLY NO ACCESS to face images.
    """
    user_role = current_user.role
    if isinstance(user_role, str):
        user_role = UserRole(user_role)

    if user_role == UserRole.AUDITOR:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Auditor role restriction: Strict prohibition from viewing face images or composites."
        )
    return current_user
