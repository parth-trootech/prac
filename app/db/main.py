from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.config import Config

# Database configuration (Async)
async_engine = create_async_engine(url=Config.DATABASE_URL, echo=True, future=True)
AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


# Dependency to get the async DB session
async def get_db():
    async with AsyncSessionLocal() as db:
        yield db
        await db.commit()
