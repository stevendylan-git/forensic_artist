from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field
from app.models.user import UserRole


class UserBase(BaseModel):
    email: str = Field(..., min_length=3, max_length=255, description="User email address")
    full_name: str
    role: UserRole = UserRole.INVESTIGATOR


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)


class UserResponse(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    is_active: bool
    mfa_enabled: bool
    created_at: Optional[datetime] = None


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str
    totp_code: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in_seconds: int
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class MfaSetupResponse(BaseModel):
    secret: str
    otpauth_url: str
    message: str


class MfaVerifyRequest(BaseModel):
    code: str
