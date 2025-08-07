from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

from app.schemas.agent_api_key import AgentApiKeyPublic
from app.models.user import User
from app.core.dependencies import get_current_user_with_provisioining as get_current_user
from app.database import get_db
from app.services import agent_service, api_key_service
from app.schemas.agent_api_key import AgentApiKeyCreate

router = APIRouter()

@router.post("/", response_model=AgentApiKeyPublic, status_code=status.HTTP_201_CREATED)
async def create_agent_api_key(
    agent_id: uuid.UUID, # We'll likely get this from the request body in the future
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate a new API key for a specified agent.
    The full key is only returned once upon creation.
    """
    # Ensure the agent belongs to the user's organization
    agent = await agent_service.get_agent_by_id(
        db=db, agent_id=agent_id, organization_id=current_user.organization_id
    )
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Generate the key
    full_key, key_prefix = api_key_service.generate_api_key()
    hashed_key = api_key_service.get_password_hash(full_key)

    # Save the hashed key to the DB
    key_create_schema = AgentApiKeyCreate(
        agent_id=agent_id,
        organization_id=current_user.organization_id,
        hashed_key=hashed_key,
        key_prefix=key_prefix
    )
    db_key = await api_key_service.create_api_key(db, key_create=key_create_schema)

    # Return the public version (with the full key) to the user
    return AgentApiKeyPublic(
        id=db_key.id,
        agent_id=db_key.agent_id,
        key_prefix=db_key.key_prefix,
        full_key=full_key, # This is the only time the user will see this
        created_at=db_key.created_at
    )