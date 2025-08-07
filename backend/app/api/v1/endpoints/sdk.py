from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

# Schemas
from app.schemas.sdk import (
    ContextExtractRequest, ContextExtractResponse,
    InteractionAssessRequest, InteractionAssessResponse, OutcomeAssessment,
    InteractionStartRequest, InteractionStartResponse,
    InteractionOutcomeRequest, InteractionOutcomeResponse
)

# Services and dependencies
from app.services import llm_service
from app.database import get_db
from app.core.dependencies import get_agent_from_sdk_auth
from app.models.agent import Agent
from app.models.interaction import Interaction
from app.models.outcome import Outcome

router = APIRouter()

# --- Helper Endpoints ---

@router.post("/context/extract", response_model=ContextExtractResponse)
async def extract_context_from_transcript(
    request: ContextExtractRequest,
    agent: Agent = Depends(get_agent_from_sdk_auth) # Protect this endpoint
):
    system_prompt = "..." # (Same prompt as before)
    user_prompt = f"Here is the transcript:\n\n{request.transcript}"
    extracted_context = await llm_service.get_json_response(...) # (Same logic as before)
    if not extracted_context:
        raise HTTPException(...)
    return ContextExtractResponse(extracted_context=extracted_context)

@router.post("/interactions/assess", response_model=InteractionAssessResponse)
async def assess_interaction_outcome(
    request: InteractionAssessRequest,
    agent: Agent = Depends(get_agent_from_sdk_auth) # Protect this endpoint
):
    system_prompt = "..." # (Same prompt as before)
    user_prompt = f"..." # (Same logic as before)
    assessment_json = await llm_service.get_json_response(...) # (Same logic as before)
    if not assessment_json or "is_success" not in assessment_json:
        raise HTTPException(...)
    validated_assessment = OutcomeAssessment(**assessment_json)
    return validated_assessment


# --- Core Learning Loop Endpoints ---

@router.post("/interactions/start", response_model=InteractionStartResponse)
async def start_interaction(
    request: InteractionStartRequest,
    agent: Agent = Depends(get_agent_from_sdk_auth),
    db: AsyncSession = Depends(get_db)
):
    """
    Logs the start of an interaction and returns an initial strategy.
    For now, it returns a default (empty) strategy.
    """
    # Create and save the interaction record
    new_interaction = Interaction(
        agent_id=agent.id,
        session_id=request.session_id,
        context=request.context,
        full_transcript=request.full_transcript
        # We will add context_embedding and applied_pattern_id later
    )
    db.add(new_interaction)
    await db.commit()
    await db.refresh(new_interaction)

    # In Phase 4, we will add the bandit logic here.
    # For now, we return a placeholder.
    strategy_to_inject = "" 
    pattern_id = None

    return InteractionStartResponse(
        interaction_id=new_interaction.id,
        strategy_to_inject=strategy_to_inject,
        pattern_id=pattern_id
    )

@router.post("/interactions/outcome", response_model=InteractionOutcomeResponse)
async def record_interaction_outcome(
    request: InteractionOutcomeRequest,
    agent: Agent = Depends(get_agent_from_sdk_auth),
    db: AsyncSession = Depends(get_db)
):
    """
    Records the final outcome of an interaction. This is the core feedback signal.
    """
    # Find the original interaction
    interaction = await db.get(Interaction, request.interaction_id)
    if not interaction or interaction.agent_id != agent.id:
        raise HTTPException(status_code=404, detail="Interaction not found")
        
    # TODO: In the future, determine is_success based on the agent's objective
    # For now, we'll assume a boolean 'success' metric is passed.
    is_success = bool(request.metrics.get("success", False))
    
    # Create the outcome record
    new_outcome = Outcome(
        interaction_id=request.interaction_id,
        source="EXPLICIT", # Or "AI_ASSISTED" if it came from that endpoint
        metrics=request.metrics,
        is_success=is_success
    )
    db.add(new_outcome)
    await db.commit()

    # In Phase 4, we will push this to a Celery queue for learning.
    # For now, we just save it.

    return InteractionOutcomeResponse()