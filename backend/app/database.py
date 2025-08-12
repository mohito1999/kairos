# backend/app/database.py 
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings

# --- ASYNC ENGINE & SESSION - DEPRECATED ---
# We are REMOVING the module-level async engine and session factory.
# These will now be managed by the async_context to prevent process-forking issues.
#
# async_engine = create_async_engine(...)
# AsyncSessionLocal = async_sessionmaker(...)


# --- SYNC ENGINE & SESSION FOR CELERY (UNCHANGED) ---
# This part remains exactly the same. It is used by our synchronous tasks
# and is perfectly safe.
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

# Import the new context manager for the FastAPI dependency
from app.core.async_context import get_async_context

async def get_db() -> AsyncSession:
    """
    FastAPI dependency to get an async database session from our new context.
    """
    async_context = get_async_context()
    session_factory = async_context.session_factory
    
    async with session_factory() as session:
        yield session

@contextmanager
def get_sync_db_session() -> Session:
    """
    Provides a transactional scope around a series of operations for Celery.
    (UNCHANGED)
    """
    db = SyncSessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()