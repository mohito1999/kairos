from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, Optional

from app.models.agent import Agent
from app.core.dependencies import get_agent_from_sdk_auth
from app.core.celery_app import celery_app

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
    celery_app.send_task(
        'app.background.tasks.process_human_interaction_task',
        args=[
            str(agent.id),
            str(payload.recording_url),
            payload.context,
            payload.outcome,
            payload.outcome_goal_description
        ]
    )
    return {"status": "accepted", "message": "Human interaction queued for processing."}