import pytest_asyncio
import asyncio
from typing import AsyncGenerator

from httpx import AsyncClient
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import get_db
from app.models.base import Base

# CRITICAL: Use in-memory SQLite for tests to avoid connection conflicts
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Create test engine with specific configuration for async SQLite
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    poolclass=StaticPool, # Required for SQLite
    connect_args={"check_same_thread": False}, # Required for SQLite
)

# Create sessionmaker for tests
TestingSessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False, 
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,  # CRITICAL: Prevents DetachedInstanceError
)

@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Creates a database session for the test.
    This is the SINGLE source of truth for the database session.
    """
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with TestingSessionLocal() as session:
        yield session
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest_asyncio.fixture(scope="function")
async def test_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """
    Creates an HTTP client with the database dependency overridden.
    """
    async def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    finally:
        app.dependency_overrides.clear()

