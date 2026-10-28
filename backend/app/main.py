import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import engine, Base
from app.audit.middleware import AuditMiddleware
from app.api.v1.router import api_v1_router
from app.api.v1.health import router as health_router

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("forensic_app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables exist in dev environment
    logger.info("Initializing Forensic Face Composite database schema...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Seed initial demo accounts
    from app.database import AsyncSessionLocal
    from app.services.seed import seed_demo_users
    async with AsyncSessionLocal() as session:
        await seed_demo_users(session)
    logger.info(f"Loaded config: TEXT_MODEL={settings.GEMINI_TEXT_MODEL}, IMAGE_MODEL={settings.GEMINI_IMAGE_MODEL}")
    yield
    # Shutdown
    await engine.dispose()
    logger.info("Database engine connections closed.")


def create_application() -> FastAPI:
    app = FastAPI(
        title="Forensic Face Composite Generator API",
        version="1.0.0",
        description="Forensic evidentiary composite generation with cognitive elicitation, FaceSpec truth source, and hash-chained audit logging.",
        lifespan=lifespan
    )

    # 1. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Forensic Audit Middleware (Rule 5: Applied to all endpoints)
    app.add_middleware(AuditMiddleware)

    # 3. Mount Routers
    from app.api.v1.cases import router as cases_router
    app.include_router(health_router)   # Direct /health
    app.include_router(api_v1_router)    # Versioned /api/v1/*
    app.include_router(cases_router)     # Direct /cases/* convenience mount

    return app


app = create_application()
