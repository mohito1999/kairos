import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

from app.models.agent import Agent
from app.schemas.agent import AgentCreate

async def create_agent(db: AsyncSession, agent_in: AgentCreate, organization_id: uuid.UUID) -> Agent:
    """
    Creates a new agent for a given organization.
    """
    new_agent = Agent(**agent_in.model_dump(), organization_id=organization_id)
    db.add(new_agent)
    await db.commit()
    await db.refresh(new_agent)
    return new_agent

async def get_agents_by_organization(db: AsyncSession, organization_id: uuid.UUID) -> List[Agent]:
    """
    Fetches all agents belonging to a specific organization.
    """
    result = await db.execute(
        select(Agent).filter(Agent.organization_id == organization_id)
    )
    return result.scalars().all()

async def get_agent_by_id(db: AsyncSession, agent_id: uuid.UUID, organization_id: uuid.UUID) -> Agent | None:
    """
    Fetches a single agent by its ID, ensuring it belongs to the correct organization.
    """
    result = await db.execute(
        select(Agent).filter(Agent.id == agent_id, Agent.organization_id == organization_id)
    )
    return result.scalars().first()