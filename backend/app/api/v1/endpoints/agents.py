from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import uuid

from app.schemas.agent import Agent, AgentCreate
from app.models.user import User
from app.core.dependencies import get_current_user_with_provisioining as get_current_user
from app.database import get_db
from app.services import agent_service

router = APIRouter()

@router.post("/", response_model=Agent, status_code=status.HTTP_201_CREATED)
async def create_agent(
    agent_in: AgentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new agent for the current user's organization.
    """
    return await agent_service.create_agent(
        db=db, agent_in=agent_in, organization_id=current_user.organization_id
    )

@router.get("/", response_model=List[Agent])
async def list_agents(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List all agents for the current user's organization.
    """
    return await agent_service.get_agents_by_organization(
        db=db, organization_id=current_user.organization_id
    )

@router.get("/{agent_id}", response_model=Agent)
async def read_agent(
    agent_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific agent by its ID.
    """
    agent = await agent_service.get_agent_by_id(
        db=db, agent_id=agent_id, organization_id=current_user.organization_id
    )
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent