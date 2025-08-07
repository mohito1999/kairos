from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from passlib.context import CryptContext

from app.models.agent_api_key import AgentApiKey
from app.models.agent import Agent

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_api_key(plain_key: str, hashed_key: str) -> bool:
    """Verifies a plain API key against its hashed version."""
    return pwd_context.verify(plain_key, hashed_key)

async def get_agent_from_api_key(db: AsyncSession, api_key: str) -> Agent | None:
    """
    Finds an active agent corresponding to a given API key.
    """
    if not api_key.startswith("kai_"):
        return None
    
    # We can't query by the full key directly, as we only store the hash.
    # This is why we store the non-secret prefix.
    key_prefix = api_key[:8] # "kai_" + 4 random chars
    
    result = await db.execute(
        select(AgentApiKey)
        .where(AgentApiKey.key_prefix == key_prefix, AgentApiKey.is_active == True)
    )
    
    possible_keys = result.scalars().all()

    for db_key in possible_keys:
        if verify_api_key(api_key, db_key.hashed_key):
            # We found the matching key, now get the agent
            agent_result = await db.execute(select(Agent).where(Agent.id == db_key.agent_id))
            return agent_result.scalars().first()
    
    return None