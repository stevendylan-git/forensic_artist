from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from app.database import get_db
from app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def get_system_health(db: AsyncSession = Depends(get_db)):
    """
    Health check endpoint reporting:
    - Overall system status
    - Database connectivity
    - Configured Gemini text and image models (without ever exposing keys)
    """
    db_connected = False
    error_msg = None

    try:
        result = await db.execute(text("SELECT 1"))
        if result.scalar() == 1:
            db_connected = True
    except Exception as exc:
        error_msg = f"Database ping failed: {str(exc)}"

    status = "ok" if db_connected else "degraded"

    return HealthResponse(
        status=status,
        environment=settings.ENVIRONMENT,
        database_connected=db_connected,
        gemini_configured=bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "test_placeholder_key_not_real"),
        gemini_text_model=settings.GEMINI_TEXT_MODEL,
        gemini_image_model=settings.GEMINI_IMAGE_MODEL,
        version="1.0.0",
        error=error_msg
    )
