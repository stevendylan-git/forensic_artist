from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.config import settings
from app.core.security import (
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_totp_secret,
    get_totp_uri,
    verify_totp_code,
)
from app.models.user import User, UserRole
from app.models.audit import AuditAction, AuditOutcome
from app.audit.ledger import AuditLedger
from app.schemas.user import (
    LoginRequest,
    TokenResponse,
    RefreshTokenRequest,
    UserResponse,
    MfaSetupResponse,
    MfaVerifyRequest,
)
from app.api.deps import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication & RBAC"])


@router.post("/login", response_model=TokenResponse)
async def login(
    login_data: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Email + password login with Argon2id password verification and optional TOTP MFA.
    Issues short-lived JWT access token + refresh token.
    """
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    stmt = select(User).where(User.email == login_data.email)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user or not verify_password(login_data.password, user.hashed_password):
        # Explicit forensic audit log of failed login attempt
        request.state.audited = True
        await AuditLedger.append_entry(
            session=db,
            action=AuditAction.LOGIN_FAILED,
            resource="/api/v1/auth/login",
            outcome=AuditOutcome.DENIED,
            user_id=user.id if user else None,
            role=user.role.value if (user and isinstance(user.role, UserRole)) else None,
            ip=client_ip,
            user_agent=user_agent,
            details={"email": login_data.email, "reason": "Bad credentials"}
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )

    # Optional TOTP MFA verification
    if user.mfa_enabled:
        if not login_data.totp_code:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"mfa_required": True, "message": "Two-factor authentication code required"}
            )
        if not verify_totp_code(user.mfa_secret or "", login_data.totp_code):
            request.state.audited = True
            await AuditLedger.append_entry(
                session=db,
                action=AuditAction.LOGIN_FAILED,
                resource="/api/v1/auth/login",
                outcome=AuditOutcome.DENIED,
                user_id=user.id,
                role=user.role.value if isinstance(user.role, UserRole) else str(user.role),
                ip=client_ip,
                user_agent=user_agent,
                details={"email": login_data.email, "reason": "Invalid TOTP MFA code"}
            )
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid two-factor authentication code",
                headers={"WWW-Authenticate": "Bearer"}
            )

    role_str = user.role.value if isinstance(user.role, UserRole) else str(user.role)
    access_token = create_access_token(
        subject=user.id,
        role=role_str,
        claims={"email": user.email, "full_name": user.full_name}
    )
    refresh_token = create_refresh_token(subject=user.id, role=role_str)

    # Explicit audit log of successful login
    request.state.audited = True
    await AuditLedger.append_entry(
        session=db,
        action=AuditAction.LOGIN,
        resource="/api/v1/auth/login",
        outcome=AuditOutcome.SUCCESS,
        user_id=user.id,
        role=role_str,
        ip=client_ip,
        user_agent=user_agent,
        details={"email": user.email, "mfa_used": user.mfa_enabled}
    )
    await db.commit()

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in_seconds=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse.model_validate(user)
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_req: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """Refreshes a short-lived access token using a valid refresh token."""
    try:
        payload = decode_token(refresh_req.refresh_token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Provided token is not a refresh token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user_id = payload.get("sub")
    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"}
        )

    role_str = user.role.value if isinstance(user.role, UserRole) else str(user.role)
    new_access_token = create_access_token(
        subject=user.id,
        role=role_str,
        claims={"email": user.email, "full_name": user.full_name}
    )
    new_refresh_token = create_refresh_token(subject=user.id, role=role_str)

    return TokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in_seconds=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse.model_validate(user)
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Returns profile for currently authenticated user."""
    return UserResponse.model_validate(current_user)


@router.post("/mfa/setup", response_model=MfaSetupResponse)
async def setup_mfa(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Initializes TOTP MFA secret for the current user.
    Returns the secret and otpauth URI to display as QR code.
    """
    secret = generate_totp_secret()
    current_user.mfa_secret = secret
    await db.flush()

    request.state.audited = True
    await AuditLedger.append_entry(
        session=db,
        action=AuditAction.MFA_SETUP,
        resource="/api/v1/auth/mfa/setup",
        outcome=AuditOutcome.SUCCESS,
        user_id=current_user.id,
        role=current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role),
        ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown"),
        details={"status": "pending_verification"}
    )
    await db.commit()

    otpauth_url = get_totp_uri(secret=secret, email=current_user.email)
    return MfaSetupResponse(
        secret=secret,
        otpauth_url=otpauth_url,
        message="Scan this QR code or enter the secret into your authenticator app, then verify with /mfa/verify"
    )


@router.post("/mfa/verify")
async def verify_mfa(
    mfa_req: MfaVerifyRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Verifies a 6-digit TOTP code and permanently activates MFA for the user."""
    if not current_user.mfa_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MFA setup has not been initiated. Call /mfa/setup first."
        )

    is_valid = verify_totp_code(current_user.mfa_secret, mfa_req.code)
    if not is_valid:
        request.state.audited = True
        await AuditLedger.append_entry(
            session=db,
            action=AuditAction.MFA_VERIFY,
            resource="/api/v1/auth/mfa/verify",
            outcome=AuditOutcome.DENIED,
            user_id=current_user.id,
            role=current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role),
            ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
            details={"status": "invalid_code"}
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid two-factor authentication code"
        )

    current_user.mfa_enabled = True
    await db.flush()

    request.state.audited = True
    await AuditLedger.append_entry(
        session=db,
        action=AuditAction.MFA_VERIFY,
        resource="/api/v1/auth/mfa/verify",
        outcome=AuditOutcome.SUCCESS,
        user_id=current_user.id,
        role=current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role),
        ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown"),
        details={"status": "mfa_activated"}
    )
    await db.commit()

    return {"status": "success", "message": "MFA has been successfully verified and activated."}
