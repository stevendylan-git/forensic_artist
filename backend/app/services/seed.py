import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user import User, UserRole
from app.core.security import get_password_hash

logger = logging.getLogger("seed")

DEMO_USERS = [
    {
        "email": "admin@forensic.local",
        "password": "DemoPassword123!",
        "full_name": "System Administrator",
        "role": UserRole.ADMIN,
    },
    {
        "email": "supervisor@forensic.local",
        "password": "DemoPassword123!",
        "full_name": "Capt. Marcus Vance (Supervisor)",
        "role": UserRole.SUPERVISOR,
    },
    {
        "email": "artist@forensic.local",
        "password": "DemoPassword123!",
        "full_name": "Alex Reed (Forensic Artist)",
        "role": UserRole.FORENSIC_ARTIST,
    },
    {
        "email": "investigator@forensic.local",
        "password": "DemoPassword123!",
        "full_name": "Det. Sarah Miller (Investigator)",
        "role": UserRole.INVESTIGATOR,
    },
    {
        "email": "auditor@forensic.local",
        "password": "DemoPassword123!",
        "full_name": "Elena Rostova (Compliance Auditor)",
        "role": UserRole.AUDITOR,
    },
]


async def seed_demo_users(session: AsyncSession) -> None:
    """
    Seeds initial demo accounts for each role if they do not exist.
    Idempotent and resilient to concurrent initialization.
    """
    for user_info in DEMO_USERS:
        stmt = select(User).where(User.email == user_info["email"])
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()
        if not existing:
            try:
                user = User(
                    email=user_info["email"],
                    hashed_password=get_password_hash(user_info["password"]),
                    full_name=user_info["full_name"],
                    role=user_info["role"],
                    is_active=True,
                    mfa_enabled=False,
                )
                session.add(user)
                await session.commit()
                logger.info(f"Seeded demo user: {user.email} [{user.role.value}]")
            except Exception:
                await session.rollback()
