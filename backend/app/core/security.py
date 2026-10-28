from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError
import pyotp
from app.config import settings

_pwd_hasher = PasswordHasher(
    time_cost=2,
    memory_cost=65536,
    parallelism=1,
    hash_len=32,
    salt_len=16
)


def get_password_hash(password: str) -> str:
    """Hashes a password using Argon2id."""
    return _pwd_hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against an Argon2id hash."""
    try:
        return _pwd_hasher.verify(hashed_password, plain_password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def create_access_token(
    subject: str,
    role: str,
    claims: Optional[Dict[str, Any]] = None,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Generates a short-lived signed JWT access token."""
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": subject,
        "role": role,
        "type": "access",
        "iat": now,
        "exp": expire,
    }
    if claims:
        payload.update(claims)

    encoded_jwt = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def create_refresh_token(
    subject: str,
    role: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Generates a longer-lived signed JWT refresh token."""
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    payload = {
        "sub": subject,
        "role": role,
        "type": "refresh",
        "iat": now,
        "exp": expire,
    }
    encoded_jwt = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Dict[str, Any]:
    """Decodes and validates a JWT token. Raises PyJWTError on invalid/expired token."""
    return jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM]
    )


# --- TOTP MFA Utilities ---

def generate_totp_secret() -> str:
    """Generates a new base32 secret for TOTP authenticator app."""
    return pyotp.random_base32()


def get_totp_uri(secret: str, email: str, issuer_name: str = "ForensicFaceApp") -> str:
    """Generates an otpauth:// URI suitable for QR code scanning."""
    totp = pyotp.TOTP(secret)
    return totp.provisioning_uri(name=email, issuer_name=issuer_name)


def verify_totp_code(secret: str, code: str) -> bool:
    """Verifies a 6-digit TOTP code against the secret (allowing 1 step tolerance)."""
    if not secret or not code:
        return False
    totp = pyotp.TOTP(secret)
    # valid_window=1 allows 30 seconds clock drift forward or backward
    return totp.verify(code.strip(), valid_window=1)
