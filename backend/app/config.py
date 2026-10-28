import os
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Forensic Face Composite Generator Settings
    Strictly loads all configuration from environment and backend/.env.
    Rules:
    - Rule 1: GEMINI_API_KEY and secrets live in backend .env only.
    - Rule 2: Model IDs come from backend/app/config.py (GEMINI_TEXT_MODEL, GEMINI_IMAGE_MODEL).
    """
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Google Gemini Models (Rule 2: Single source of truth)
    GEMINI_API_KEY: str = Field(default="", description="Secret Google GenAI API Key")
    GEMINI_TEXT_MODEL: str = Field(default="gemini-2.5-flash", description="Model for elicitation and structured output")
    GEMINI_IMAGE_MODEL: str = Field(default="imagen-3.0-generate-002", description="Model for face rendering")

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://forensic_user:forensic_pass@localhost:5432/forensic_artist",
        description="Async database connection string"
    )

    # Application & Forensic Config
    ENVIRONMENT: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"])
    WATERMARK_TEXT: str = Field(default="AI-GENERATED APPROXIMATION - INVESTIGATIVE LEAD ONLY")

    # Security, Auth & RBAC
    JWT_SECRET_KEY: str = Field(
        default="forensic-face-audit-secret-key-change-in-production-2026",
        description="HMAC secret key for JWT signing"
    )
    JWT_ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=15, description="Short-lived access token duration")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="Refresh token duration")
    AUTO_LOGOUT_MINUTES: int = Field(default=15, description="Client inactivity auto-logout threshold")

    # Storage & Envelope Encryption Config
    STORAGE_LOCAL_DIR: str = Field(default="data/storage", description="Path to local encrypted object store")
    STORAGE_MASTER_KEY: str = Field(
        default="forensic-face-master-kek-32bytes-secret!",
        description="Master Key Encryption Key for envelope encryption (32-bytes or passphrase)"
    )
    IMAGEN_COST_PER_IMAGE: float = Field(default=0.030, description="Estimated USD cost per image generation")
    IMAGE_GEN_CONCURRENCY: int = Field(default=3, description="Concurrency limit semaphore for candidate generation")
    RUN_LIVE_GEMINI_TESTS: bool = Field(default=False, description="Flag gating live tests with Google GenAI API")

    # Client Retry & Timeout Config
    GEMINI_REQUEST_TIMEOUT_SECONDS: float = Field(default=30.0)
    GEMINI_MAX_RETRIES: int = Field(default=3)


settings = Settings()
