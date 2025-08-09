from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, Optional

from app.models.agent import Agent
from app.core.dependencies import get_agent_from_sdk_auth
from app.background.tasks import process_human_interaction_task

router = APIRouter()

class HumanInteractionPayload(BaseModel):
    recording_url: HttpUrl
    context: Optional[Dict[str, Any]] = None
    outcome: Optional[Dict[str, Any]] = None
    outcome_goal_description: Optional[str] = None

@router.post("/webhook", status_code=status.HTTP_202_ACCEPTED)
async def human_interaction_webhook(
    payload: HumanInteractionPayload,
    agent: Agent = Depends(get_agent_from_sdk_auth),
):
    """
    Accepts a webhook with a recording URL from a human interaction
    and queues it for transcription and analysis.
    """
    process_human_interaction_task.delay(
        agent_id=str(agent.id),
        recording_url=str(payload.recording_url),
        context=payload.context,
        explicit_outcome=payload.outcome,
        outcome_goal=payload.outcome_goal_description
    )
    return {"status": "accepted", "message": "Human interaction queued for processing."}