import os
import sys
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# Set test environment
os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_suite.db"
os.environ["GEMINI_API_KEY"] = "test_key_for_unit_tests"
os.environ["GEMINI_TEXT_MODEL"] = "gemini-2.5-flash"
os.environ["GEMINI_IMAGE_MODEL"] = "imagen-3.0-generate-002"
os.environ["JWT_SECRET_KEY"] = "test-jwt-secret-key-32-bytes-long-for-testing"

from app.database import Base, get_db
from app.main import app
from app.services.seed import seed_demo_users
from app.core.security import create_access_token
from app.models.user import User, UserRole
from sqlalchemy import event, select

test_engine = create_async_engine(
    "sqlite+aiosqlite:///./test_suite.db",
    echo=False,
    connect_args={"check_same_thread": False, "timeout": 30.0}
)


@event.listens_for(test_engine.sync_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


TestingSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


@pytest_asyncio.fixture(autouse=True)
async def init_test_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    # Seed demo accounts
    async with TestingSessionLocal() as session:
        await seed_demo_users(session)

    yield

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()
    if os.path.exists("./test_suite.db"):
        try:
            os.remove("./test_suite.db")
        except Exception:
            pass


@pytest_asyncio.fixture
async def db_session():
    async with TestingSessionLocal() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client():
    async def override_get_db():
        async with TestingSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def auth_tokens(db_session):
    """Returns a dict mapping role to JWT access token string."""
    tokens = {}
    stmt = select(User)
    result = await db_session.execute(stmt)
    users = result.scalars().all()
    for user in users:
        role_str = user.role.value if isinstance(user.role, UserRole) else str(user.role)
        token = create_access_token(
            subject=user.id,
            role=role_str,
            claims={"email": user.email, "full_name": user.full_name}
        )
        tokens[role_str] = token
        tokens[user.email] = token
    return tokens
