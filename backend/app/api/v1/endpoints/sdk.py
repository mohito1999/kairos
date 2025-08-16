import uuid
import json
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

# Schemas
from app.schemas.sdk import (
    ContextExtractRequest, ContextExtractResponse,
    InteractionAssessRequest, InteractionAssessResponse, OutcomeAssessment,
    InteractionStartRequest, InteractionStartResponse,
    InteractionOutcomeRequest, InteractionOutcomeResponse
)

# Services, dependencies, and models
from app.services import llm_service, embedding_service
from app.database import get_db
from app.core.dependencies import get_agent_from_sdk_auth
from app.models.agent import Agent
from app.models.interaction import Interaction
from app.models.outcome import Outcome
from app.models.learned_pattern import LearnedPattern
from app.core.celery_app import celery_app

router = APIRouter()

# --- Helper Endpoints ---

@router.post("/context/extract", response_model=ContextExtractResponse)
async def extract_context_from_transcript(
    request: ContextExtractRequest,
    agent: Agent = Depends(get_agent_from_sdk_auth)
):
    """
    Extracts a structured context object from a conversation transcript.
    """
    # This is the full, engineered prompt
    system_prompt = """
    You are an expert AI assistant specializing in silent conversation analysis. 
    Your task is to analyze the provided transcript and extract key entities into a single, flat JSON object.
    If the user provides a specific schema, adhere to it strictly.
    If no schema is provided, use your best judgment to extract common, relevant entities like names, topics, user intent, user sentiment (positive, neutral, negative), and key user requests.
    Only return a valid JSON object. Do not include any explanations or conversational text.
    """
    
    schema_instruction = ""
    if request.schema_definition:
        schema_str = json.dumps(request.schema_definition, indent=2)
        schema_instruction = f"\n\nUse this specific JSON schema for extraction:\n{schema_str}"

    user_prompt = f"Here is the transcript:\n\n---\n{request.transcript}\n---\n{schema_instruction}"

    extracted_context = await llm_service.get_json_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model="mistralai/mistral-7b-instruct:free" # Use a fast model for this
    )
    
    if not extracted_context:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract context from the transcript."
        )

    return ContextExtractResponse(extracted_context=extracted_context)


@router.post("/interactions/assess", response_model=InteractionAssessResponse)
async def assess_interaction_outcome(
    request: InteractionAssessRequest,
    agent: Agent = Depends(get_agent_from_sdk_auth)
):
    """
    Provides an AI-driven assessment of whether a conversation met its goal.
    """
    # This is the full, engineered prompt
    system_prompt = """
    You are a meticulous and objective AI evaluator. Your task is to determine if a conversation successfully met a specific goal.
    Analyze the provided transcript and the success goal, then respond with a JSON object containing your assessment.
    The JSON object MUST have the following keys and data types:
    - "is_success": boolean (True if the goal was met, False otherwise).
    - "confidence_score": float (Your confidence in the assessment, from 0.0 to 1.0).
    - "reason": string (A brief, one-sentence explanation for your decision, from an objective third-person perspective).
    - "failure_type": string or null (If not successful, choose ONE from this list: [UNRESOLVED_ISSUE, USER_FRUSTRATION, AGENT_CONFUSION, GOAL_NOT_MET]. Otherwise, null).
    Only return a valid JSON object.
    """
    
    user_prompt = f"""
    SUCCESS GOAL: "{request.goal}"
    
    CONVERSATION TRANSCRIPT:
    ---
    {request.transcript}
    ---
    """
    
    assessment_json = await llm_service.get_json_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model="openai/gpt-4o-mini" # Use a powerful model for this reasoning task
    )

    if not assessment_json or "is_success" not in assessment_json:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get a valid assessment from the LLM."
        )

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
    Logs the start of an interaction and returns the best strategy
    using a contextual Thompson Sampling bandit.
    """
    strategy_to_inject = "" 
    pattern_id = None
    context_embedding = None

    if request.context:
        # Generate embedding for the incoming context for similarity search
        context_str = json.dumps(request.context)
        context_embedding = await embedding_service.get_embedding(context_str)
        
        if context_embedding:
            # pgvector query to find the 5 most similar patterns
            # Note: <-> is the cosine distance operator from pgvector
            stmt = (
                select(LearnedPattern)
                .where(LearnedPattern.agent_id == agent.id, LearnedPattern.status.in_(["ACTIVE", "CANDIDATE"]))
                .order_by(LearnedPattern.trigger_embedding.l2_distance(context_embedding))
                .limit(5)
            )
            result = await db.execute(stmt)
            candidate_patterns = result.scalars().all()
            
            if candidate_patterns:
                best_pattern, max_sample = None, -1
                for pattern in candidate_patterns:
                    alpha = 1 + pattern.success_count
                    beta = 1 + (pattern.impressions - pattern.success_count)
                    sample = np.random.beta(alpha, beta)
                    if sample > max_sample:
                        max_sample, best_pattern = sample, pattern
                
                if best_pattern:
                    strategy_to_inject = best_pattern.suggested_strategy
                    pattern_id = best_pattern.id
                    print(f"Bandit selected pattern {pattern_id} for the current context.")

    # Create and save the interaction record
    new_interaction = Interaction(
        agent_id=agent.id,
        session_id=request.session_id,
        context=request.context,
        full_transcript=request.full_transcript,
        applied_pattern_id=pattern_id,
        context_embedding=context_embedding
    )
    db.add(new_interaction)
    await db.commit()
    await db.refresh(new_interaction)

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
    interaction = await db.get(Interaction, request.interaction_id)
    if not interaction or interaction.agent_id != agent.id:
        raise HTTPException(status_code=404, detail="Interaction not found")
        
    # Determine is_success based on a "success" key in the metrics
    is_success = bool(request.metrics.get("success", False))
    
    new_outcome = Outcome(
        interaction_id=request.interaction_id,
        source="EXPLICIT",
        metrics=request.metrics,
        is_success=is_success
    )
    db.add(new_outcome)
    await db.commit()

    # Trigger the background learning task
    celery_app.send_task(
        'app.background.tasks.process_live_outcome_task',
        args=[str(request.interaction_id)]
    )

    return InteractionOutcomeResponse()