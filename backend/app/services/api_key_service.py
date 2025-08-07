import secrets
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Tuple

from app.models.agent_api_key import AgentApiKey
from app.schemas.agent_api_key import AgentApiKeyCreate

# Setup the password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
API_KEY_PREFIX = "kai_"
API_KEY_LENGTH = 32 # The length of the random part of the key

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def generate_api_key() -> Tuple[str, str]:
    """
    Generates a new API key.
    Returns a tuple of (full_key, key_prefix).
    """
    random_part = secrets.token_urlsafe(API_KEY_LENGTH)
    full_key = f"{API_KEY_PREFIX}{random_part}"
    return full_key, full_key[:len(API_KEY_PREFIX) + 4] # e.g., "kai_AbCd..."

async def create_api_key(db: AsyncSession, key_create: AgentApiKeyCreate) -> AgentApiKey:
    """
    Saves a new hashed API key to the database.
    """
    db_key = AgentApiKey(**key_create.model_dump())
    db.add(db_key)
    await db.commit()
    await db.refresh(db_key)
    return db_key