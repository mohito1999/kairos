# backend/app/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core.config import settings

# --- ASYNC ENGINE & SESSION FOR FASTAPI ---
async_engine = create_async_engine(
    settings.DATABASE_URL, # Uses postgresql+asyncpg://
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False # Good practice for async sessions
)

# --- SYNC ENGINE & SESSION FOR CELERY ---
# Note: settings.SYNC_DATABASE_URL converts the URL to use psycopg2
sync_engine = create_engine(
    settings.SYNC_DATABASE_URL, # Uses postgresql+psycopg2://
    pool_pre_ping=True,
)

SyncSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=sync_engine
)

# --- DEPENDENCIES ---
async def get_db():
    """FastAPI dependency to get an async database session."""
    async with AsyncSessionLocal() as session:
        yield session

# CRITICAL FIX: This is the definitive context manager for sync sessions.
@contextmanager
def get_sync_db_session() -> Session:
    """Provides a transactional scope around a series of operations for Celery."""
    db = SyncSessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()