from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.core.config import settings

# Create an asynchronous engine
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True, # Checks connection vitality before use
)

# Create a configured "Session" class
AsyncSessionLocal = async_sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine
)

# Dependency to get a DB session
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session