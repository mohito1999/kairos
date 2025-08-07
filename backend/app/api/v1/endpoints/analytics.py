from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any
import uuid

from app.models.user import User
from app.core.dependencies import get_current_user_with_provisioining as get_current_user
from app.database import get_db
from app.services import agent_service # We'll add analytics services later

# For now, we will put the query logic directly in the endpoint.
# In a larger app, this would be in a dedicated analytics_service.py.
from sqlalchemy import select, func, case
from app.models.interaction import Interaction
from app.models.outcome import Outcome
from app.models.learned_pattern import LearnedPattern
from app.models.suggested_opportunity import SuggestedOpportunity


router = APIRouter()

@router.get("/performance-over-time/{agent_id}", response_model=Dict[str, Any])
async def get_performance_over_time(
    agent_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Fetches aggregated performance data for an agent to display in a chart.
    """
    # Ensure agent belongs to the user's org
    agent = await agent_service.get_agent_by_id(db, agent_id, current_user.organization_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    # Define the date grouping expression once for clarity
    date_series = func.date_trunc('day', Interaction.created_at).label('date')

    stmt = (
        select(
            date_series,
            func.count(Interaction.id).label('total_interactions'),
            func.sum(
                case((Outcome.is_success == True, 1), else_=0)
            ).label('successful_interactions')
        )
        .select_from(Interaction) # Explicitly state the FROM clause
        .join(Outcome, Interaction.id == Outcome.interaction_id)
        .where(Interaction.agent_id == agent_id)
        .group_by(date_series) # Group by the labeled expression
        .order_by(date_series.desc()) # Order by the labeled expression
        .limit(30)
    )
    
    result = await db.execute(stmt)
    data = result.all()
    
    # Format the data for a charting library like Recharts
    chart_data = [
        {
            "date": row.date.strftime('%Y-%m-%d'),
            "success_rate": (row.successful_interactions / row.total_interactions) * 100 if row.total_interactions > 0 else 0
        }
        for row in data
    ]

    return {"timeseries": chart_data}


@router.get("/patterns/{agent_id}", response_model=List[Dict[str, Any]])
async def list_patterns_for_agent(
    agent_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Fetches all learned patterns for a specific agent.
    """
    agent = await agent_service.get_agent_by_id(db, agent_id, current_user.organization_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    stmt = select(LearnedPattern).where(LearnedPattern.agent_id == agent_id)
    result = await db.execute(stmt)
    patterns = result.scalars().all()
    
    # We can calculate the success rate on the fly
    return [
        {
            "id": p.id,
            "source": p.source,
            "trigger_context_summary": p.trigger_context_summary,
            "suggested_strategy": p.suggested_strategy,
            "status": p.status,
            "impressions": p.impressions,
            "success_rate": (p.success_count / p.impressions) * 100 if p.impressions > 0 else 0
        }
        for p in patterns
    ]


@router.get("/opportunities", response_model=List[Dict[str, Any]])
async def list_opportunities(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Fetches all suggested opportunities for the user's organization.
    """
    stmt = select(SuggestedOpportunity).where(
        SuggestedOpportunity.organization_id == current_user.organization_id,
        SuggestedOpportunity.status == "NEW"
    )
    result = await db.execute(stmt)
    opportunities = result.scalars().all()
    return [o.__dict__ for o in opportunities]