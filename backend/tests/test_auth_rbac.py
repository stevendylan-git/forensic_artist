import pytest
import pyotp
from httpx import AsyncClient
from sqlalchemy import select
from app.models.user import User, UserRole
from app.core.security import get_password_hash, verify_password


@pytest.mark.asyncio
async def test_argon2_password_hashing():
    """Verifies Argon2id password hashing and verification."""
    password = "SuperSecretPassword123!"
    hashed = get_password_hash(password)
    assert hashed != password
    assert "$argon2" in hashed
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword123!", hashed) is False


@pytest.mark.asyncio
async def test_demo_users_seeded(db_session):
    """Verifies all 5 required demo roles are seeded in the database."""
    stmt = select(User)
    result = await db_session.execute(stmt)
    users = result.scalars().all()
    roles = {u.role.value if isinstance(u.role, UserRole) else str(u.role) for u in users}

    expected_roles = {
        UserRole.ADMIN.value,
        UserRole.SUPERVISOR.value,
        UserRole.INVESTIGATOR.value,
        UserRole.FORENSIC_ARTIST.value,
        UserRole.AUDITOR.value,
    }
    assert expected_roles.issubset(roles)


@pytest.mark.asyncio
async def test_login_success(client: AsyncClient):
    """Verifies email + password login and token issuance."""
    response = await client.post(
        "/api/v1/auth/login",
        json={
            "email": "investigator@forensic.local",
            "password": "DemoPassword123!"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "investigator@forensic.local"
    assert data["user"]["role"] == "Investigator"


@pytest.mark.asyncio
async def test_login_bad_credentials_logs_audit(client: AsyncClient, db_session):
    """Verifies failed login raises 401 and creates LOGIN_FAILED audit log."""
    response = await client.post(
        "/api/v1/auth/login",
        json={
            "email": "investigator@forensic.local",
            "password": "IncorrectPassword!"
        }
    )
    assert response.status_code == 401

    # Check that LOGIN_FAILED is in audit log
    from app.models.audit import AuditLogEntry, AuditAction
    stmt = select(AuditLogEntry).where(AuditLogEntry.action == AuditAction.LOGIN_FAILED)
    result = await db_session.execute(stmt)
    entries = result.scalars().all()
    assert len(entries) >= 1
    assert entries[-1].outcome.value == "DENIED"


@pytest.mark.asyncio
async def test_token_refresh(client: AsyncClient):
    """Verifies token refresh using valid refresh token."""
    login_resp = await client.post(
        "/api/v1/auth/login",
        json={
            "email": "admin@forensic.local",
            "password": "DemoPassword123!"
        }
    )
    refresh_token = login_resp.json()["refresh_token"]

    refresh_resp = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token}
    )
    assert refresh_resp.status_code == 200
    data = refresh_resp.json()
    assert "access_token" in data
    assert "refresh_token" in data


@pytest.mark.asyncio
async def test_totp_mfa_flow(client: AsyncClient, auth_tokens):
    """Verifies TOTP MFA setup, verification, and enforced two-factor login."""
    token = auth_tokens["artist@forensic.local"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Setup MFA
    setup_resp = await client.post("/api/v1/auth/mfa/setup", headers=headers)
    assert setup_resp.status_code == 200
    setup_data = setup_resp.json()
    secret = setup_data["secret"]
    assert "otpauth://" in setup_data["otpauth_url"]

    # 2. Verify with valid TOTP code
    totp = pyotp.TOTP(secret)
    valid_code = totp.now()

    verify_resp = await client.post(
        "/api/v1/auth/mfa/verify",
        headers=headers,
        json={"code": valid_code}
    )
    assert verify_resp.status_code == 200
    assert verify_resp.json()["status"] == "success"

    # 3. Login without TOTP code now fails with mfa_required
    login_no_mfa = await client.post(
        "/api/v1/auth/login",
        json={
            "email": "artist@forensic.local",
            "password": "DemoPassword123!"
        }
    )
    assert login_no_mfa.status_code == 403
    assert login_no_mfa.json()["detail"]["mfa_required"] is True

    # 4. Login with valid TOTP code succeeds
    login_with_mfa = await client.post(
        "/api/v1/auth/login",
        json={
            "email": "artist@forensic.local",
            "password": "DemoPassword123!",
            "totp_code": totp.now()
        }
    )
    assert login_with_mfa.status_code == 200
    assert "access_token" in login_with_mfa.json()


@pytest.mark.asyncio
async def test_auth_me_endpoint(client: AsyncClient, auth_tokens):
    """Verifies /auth/me returns current authenticated user profile."""
    token = auth_tokens["supervisor@forensic.local"]
    resp = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["email"] == "supervisor@forensic.local"
    assert data["role"] == "Supervisor"
