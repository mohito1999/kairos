import uuid
import json
import numpy as np
import pandas as pd
from io import StringIO
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

import asyncio
import sqlalchemy as sa
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.core.config import settings
from app.core.celery_app import celery_app
from app.database import get_sync_db_session

# Models
from app.models.agent import Agent
from app.models.historical_upload import HistoricalUpload
from app.models.historical_interaction import HistoricalInteraction
from app.models.human_interaction import HumanInteraction
from app.models.learned_pattern import LearnedPattern, PatternStatusEnum
from app.models.interaction import Interaction
from app.models.outcome import Outcome
from app.models.suggested_opportunity import SuggestedOpportunity

# Services
from app.services import embedding_service, llm_service
from app.services.transcription_service import transcription_service

# Scientific Libraries
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind

def run_async_task(async_func, *args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(async_func(*args, **kwargs))
    finally:
        tasks = asyncio.all_tasks(loop=loop)
        for task in tasks: task.cancel()
        if tasks: loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.close()
        asyncio.set_event_loop(None)

async def is_duplicate_pattern(session: AsyncSession, new_pattern_strategy: str, agent_id: uuid.UUID) -> bool:
    stmt = select(LearnedPattern).where(LearnedPattern.agent_id == agent_id, LearnedPattern.status == PatternStatusEnum.ACTIVE)
    existing_patterns = (await session.execute(stmt)).scalars().all()
    if not existing_patterns: return False
    
    new_embedding_list = await embedding_service.get_embeddings([new_pattern_strategy])
    if not new_embedding_list or not new_embedding_list[0]: return False
    
    new_embedding = new_embedding_list[0]
    existing_embeddings = await embedding_service.get_embeddings([p.suggested_strategy for p in existing_patterns])
    for emb in existing_embeddings:
        if emb and cosine_similarity([new_embedding], [emb])[0][0] > 0.95:
            return True
    return False

def is_quality_pattern(pattern_json: dict) -> bool:
    strategy = pattern_json.get("suggested_strategy", "").lower()
    trigger = pattern_json.get("trigger_context_summary", "").lower()
    if len(strategy.split()) < 4 or len(trigger.split()) < 3: return False
    return True

async def _async_run_contrastive_engine(
    db: AsyncSession, 
    agent_id: uuid.UUID, 
    interactions: List[HistoricalInteraction],
    source: str,
    source_upload_id: Optional[uuid.UUID] = None
):
    """
    This is the refactored, reusable core of the Contrastive Engine.
    It takes a list of interactions and discovers patterns from them.
    """
    failed_interactions = [i for i in interactions if not i.is_success and i.original_response]
    if len(failed_interactions) < 5:
        print(f"Engine: Not enough failed interactions ({len(failed_interactions)}) to run.")
        return

    # A. OPPORTUNITY DISCOVERY
    failed_transcripts = [i.original_response for i in failed_interactions]
    failed_embeddings_list = await embedding_service.get_embeddings(failed_transcripts)
    valid_embeddings = [emb for emb in failed_embeddings_list if emb]
    valid_interactions = [inter for i, inter in enumerate(failed_interactions) if failed_embeddings_list[i]]
    if len(valid_embeddings) < 5: return

    dbscan = DBSCAN(eps=0.5, min_samples=3, metric="cosine")
    clusters = dbscan.fit_predict(np.array(valid_embeddings))
    
    battlegrounds = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id != -1: battlegrounds.setdefault(cluster_id, []).append(valid_interactions[i])

    print(f"Engine: Discovered {len(battlegrounds)} battlegrounds.")
    patterns_to_create = []

    # B. CONTRASTIVE ANALYSIS
    for cluster_id, losing_interactions in battlegrounds.items():
        losing_centroid_embeddings = [emb for inter in losing_interactions for emb in await embedding_service.get_embeddings([inter.original_response]) if emb]
        if not losing_centroid_embeddings: continue
        losing_centroid = np.mean(losing_centroid_embeddings, axis=0)
        
        successful_interactions = [i for i in interactions if i.is_success]
        if not successful_interactions: continue

        successful_transcripts = [i.original_response for i in successful_interactions]
        successful_embeddings = await embedding_service.get_embeddings(successful_transcripts)
        
        winning_interactions = [successful_interactions[i] for i, emb in enumerate(successful_embeddings) if emb and cosine_similarity([losing_centroid], [emb])[0][0] > 0.8]

        if len(winning_interactions) < 2 or len(losing_interactions) < 2: continue

        # C. PATTERN CREATION
        positive_snippets = [i.original_response for i in winning_interactions[:5]]
        negative_snippets = [i.original_response for i in losing_interactions[:5]]
        
        system_prompt = "You are a sales coach... Distill the winning tactic into a JSON object with keys 'trigger_context_summary' and 'suggested_strategy'."
        user_prompt = f"FAILED SNIPPETS:\n{json.dumps(negative_snippets)}\n\nSUCCESSFUL SNIPPETS:\n{json.dumps(positive_snippets)}\n\nWhat is the battleground and the specific, winning strategy?"
        
        pattern_json = await llm_service.get_json_response(system_prompt, user_prompt, model="openai/gpt-4o")

        if pattern_json and is_quality_pattern(pattern_json) and not await is_duplicate_pattern(db, pattern_json["suggested_strategy"], agent_id):
            patterns_to_create.append(LearnedPattern(
                agent_id=agent_id, source=source, status="CANDIDATE", source_upload_id=source_upload_id,
                battleground_context={"cluster_id": int(cluster_id)}, positive_examples={"transcripts": positive_snippets},
                negative_examples={"transcripts": negative_snippets}, trigger_context_summary=pattern_json["trigger_context_summary"],
                suggested_strategy=pattern_json["suggested_strategy"],
            ))
    
    if patterns_to_create:
        print(f"Engine: Discovered {len(patterns_to_create)} new candidate patterns from source '{source}'.")
        db.add_all(patterns_to_create)
        await db.commit()
        # If this came from a historical upload, we trigger validation
        if source_upload_id:
            upload = await db.get(HistoricalUpload, source_upload_id)
            if upload:
                upload.status = "VALIDATING"
                await db.commit()
                celery_app.send_task('app.background.tasks.validate_patterns_task', args=[str(source_upload_id)])

async def _async_process_historical_upload(upload_id: str, file_content_bytes: bytes, data_mapping: dict):
    engine = create_async_engine(settings.DATABASE_URL)
    AsyncSessionLocal_Task = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with AsyncSessionLocal_Task() as db:
        upload = await db.get(HistoricalUpload, uuid.UUID(upload_id))
        if not upload: return

        try:
            csv_content = file_content_bytes.decode('utf-8', errors='ignore')
            df = pd.read_csv(StringIO(csv_content))
            
            transcript_col = data_mapping.get("conversation_transcript")
            if not transcript_col or transcript_col not in df.columns:
                raise ValueError(f"Transcript column '{transcript_col}' not found.")
            
            outcome_col = data_mapping.get("outcome_column")
            outcome_goal = data_mapping.get("outcome_goal_description")
            context_cols = {k: v for k, v in data_mapping.items() if k.startswith("context_")}

            interactions_to_create = []
            for _, row in df.iterrows():
                context = {k.replace("context_", ""): row.get(v) for k, v in context_cols.items() if v in df.columns}
                response_text = str(row.get(transcript_col, ""))
                is_success, raw_outcome = False, ""
                if outcome_col and outcome_col in df.columns:
                    raw_outcome_val = str(row.get(outcome_col, ""))
                    is_success = raw_outcome_val.strip().lower() in ['true', 'success', '1', 'yes', 'resolved']
                    raw_outcome = raw_outcome_val
                elif outcome_goal and response_text:
                    is_success = (await llm_service.get_json_response(
                        "You are an AI evaluator...", f"GOAL: \"{outcome_goal}\"\nTRANSCRIPT:\n{response_text}"
                    )).get("is_success", False)
                    raw_outcome = "judged_by_ai"
                
                interactions_to_create.append(HistoricalInteraction(
                    upload_id=upload.id, original_context=context, original_response=response_text,
                    is_success=is_success, extracted_outcome={"value": raw_outcome}
                ))
            
            db.add_all(interactions_to_create)
            upload.status = "SPLITTING_DATA"
            upload.total_interactions = len(df)
            upload.processed_interactions = len(df)
            await db.commit()
            
            celery_app.send_task('app.background.tasks.split_historical_data_task', args=[upload_id])
        except Exception as e:
            print(f"Error processing upload {upload.id}: {e}")
            upload.status = "FAILED"; await db.commit()
    await engine.dispose()

async def _async_split_historical_data(upload_id: str):
    engine = create_async_engine(settings.DATABASE_URL)
    AsyncSessionLocal_Task = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with AsyncSessionLocal_Task() as db:
        upload = await db.get(HistoricalUpload, uuid.UUID(upload_id))
        if not upload: return

        stmt = select(HistoricalInteraction).where(HistoricalInteraction.upload_id == upload.id, HistoricalInteraction.is_success == True)
        successful_interactions = (await db.execute(stmt)).scalars().all()
        
        if len(successful_interactions) < 10:
            training_ids, holdout_ids = [i.id for i in successful_interactions], []
        else:
            training_ids, holdout_ids = train_test_split([i.id for i in successful_interactions], test_size=0.3, random_state=42)

        upload.interaction_id_split = {"training_set": [str(uid) for uid in training_ids], "holdout_set": [str(uid) for uid in holdout_ids]}
        upload.status = "EXTRACTING_PATTERNS"
        await db.commit()
        
        celery_app.send_task('app.background.tasks.extract_patterns_from_history_task', args=[upload_id])
    await engine.dispose()

async def _async_extract_patterns_from_history(upload_id: str):
    """Orchestrator for historical data. Fetches data and calls the core engine."""
    engine = create_async_engine(settings.DATABASE_URL)
    AsyncSessionLocal_Task = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with AsyncSessionLocal_Task() as db:
        upload = await db.get(HistoricalUpload, uuid.UUID(upload_id))
        if not upload or not upload.interaction_id_split: return

        training_set_ids = [uuid.UUID(id_str) for id_str in upload.interaction_id_split.get("training_set", [])]
        if not training_set_ids: upload.status = "COMPLETED"; await db.commit(); return
        
        training_interactions = (await db.execute(select(HistoricalInteraction).where(HistoricalInteraction.id.in_(training_set_ids)))).scalars().all()
        
        await _async_run_contrastive_engine(
            db=db, 
            agent_id=upload.agent_id,
            interactions=training_interactions,
            source="HISTORICAL_CONTRASTIVE",
            source_upload_id=upload.id
        )
    await engine.dispose()

async def _async_discover_patterns_from_live_data(agent_id: str):
    """Orchestrator for live data. Fetches data and calls the core engine."""
    engine = create_async_engine(settings.DATABASE_URL)
    AsyncSessionLocal_Task = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with AsyncSessionLocal_Task() as db:
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        
        stmt = select(Interaction).options(selectinload(Interaction.outcome)).where(
            Interaction.agent_id == uuid.UUID(agent_id),
            Interaction.created_at >= seven_days_ago
        )
        recent_interactions = (await db.execute(stmt)).scalars().all()

        if len(recent_interactions) < 20:
            print(f"Live Learning: Not enough interactions ({len(recent_interactions)}) for agent {agent_id}.")
            return

        pseudo_historical_interactions = [
            HistoricalInteraction(
                original_context=inter.context,
                original_response=inter.full_transcript,
                is_success=inter.outcome.is_success if inter.outcome else False
            ) for inter in recent_interactions if inter.outcome
        ]
        
        await _async_run_contrastive_engine(
            db=db,
            agent_id=uuid.UUID(agent_id),
            interactions=pseudo_historical_interactions,
            source="LIVE_DISCOVERED"
        )
    await engine.dispose()

async def _async_validate_patterns(upload_id: str):
    """
    The async core of the pattern validation task. This function performs a
    rigorous statistical test for each candidate pattern against the holdout set.
    """
    engine = create_async_engine(settings.DATABASE_URL)
    AsyncSessionLocal_Task = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with AsyncSessionLocal_Task() as db:
        # 1. Load Upload and Holdout Data
        upload = await db.get(HistoricalUpload, uuid.UUID(upload_id))
        if not upload or not upload.interaction_id_split:
            print(f"Validation Error: Upload {upload_id} or its split data not found.")
            return

        holdout_ids_str = upload.interaction_id_split.get("holdout_set", [])
        if not holdout_ids_str:
            print(f"No holdout set for upload {upload_id}. Auto-promoting CANDIDATE patterns to VALIDATED.")
            stmt = update(LearnedPattern).where(
                LearnedPattern.source_upload_id == upload.id,
                LearnedPattern.status == 'CANDIDATE'
            ).values(status='VALIDATED')
            await db.execute(stmt)
            upload.status = "COMPLETED"
            await db.commit()
            await engine.dispose()
            return
            
        holdout_ids = [uuid.UUID(id_str) for id_str in holdout_ids_str]
        holdout_interactions = (await db.execute(
            select(HistoricalInteraction).where(HistoricalInteraction.id.in_(holdout_ids))
        )).scalars().all()

        # 2. Get all Candidate Patterns for this Upload
        candidate_patterns = (await db.execute(select(LearnedPattern).where(
            LearnedPattern.source_upload_id == upload.id, 
            LearnedPattern.status == 'CANDIDATE'
        ))).scalars().all()
        
        print(f"Found {len(candidate_patterns)} candidate patterns to validate against a holdout set of {len(holdout_interactions)}.")
        if not candidate_patterns:
            upload.status = "COMPLETED"
            await db.commit()
            await engine.dispose()
            return

        # Pre-fetch all necessary embeddings for the holdout set to avoid repeated calls
        holdout_transcripts = [i.original_response for i in holdout_interactions]
        holdout_embeddings = await embedding_service.get_embeddings(holdout_transcripts)
        
        # Create a map for easy lookup
        interaction_embedding_map = {
            holdout_interactions[i].id: emb 
            for i, emb in enumerate(holdout_embeddings) if emb
        }

        # 3. Iterate through each Candidate Pattern and Validate
        for pattern in candidate_patterns:
            # The "battleground" is defined by the centroid of the failures that created the pattern
            losing_examples = pattern.negative_examples.get("transcripts", [])
            if not losing_examples:
                pattern.status = "REJECTED"
                print(f"Pattern {pattern.id} rejected: No negative examples to define battleground.")
                continue

            losing_embeddings = await embedding_service.get_embeddings(losing_examples)
            battleground_centroid = np.mean([emb for emb in losing_embeddings if emb], axis=0)

            # 4. Find Relevant Interactions in Holdout Set (Context Matching)
            relevant_holdout_interactions = []
            for interaction in holdout_interactions:
                emb = interaction_embedding_map.get(interaction.id)
                if emb is not None:
                    # Check how similar this interaction is to the battleground
                    similarity = cosine_similarity([battleground_centroid], [emb])[0][0]
                    if similarity > 0.8: # Threshold for being "in the same situation"
                        relevant_holdout_interactions.append(interaction)
            
            if len(relevant_holdout_interactions) < 10: # Need at least 10 examples for a meaningful test
                pattern.status = "REJECTED"
                print(f"Pattern {pattern.id} rejected: Insufficient relevant examples in holdout set ({len(relevant_holdout_interactions)} found).")
                continue

            # 5. Form Treatment vs. Control Groups
            treatment_group = []
            control_group = []
            strategy_embedding_list = await embedding_service.get_embeddings([pattern.suggested_strategy])
            if not strategy_embedding_list or not strategy_embedding_list[0]:
                pattern.status = "REJECTED"
                print(f"Pattern {pattern.id} rejected: Could not generate embedding for its own strategy.")
                continue
            strategy_embedding = strategy_embedding_list[0]

            for interaction in relevant_holdout_interactions:
                interaction_embedding = interaction_embedding_map.get(interaction.id)
                if interaction_embedding:
                    similarity_to_strategy = cosine_similarity([strategy_embedding], [interaction_embedding])[0][0]
                    if similarity_to_strategy > 0.85: # High threshold to confirm strategy was used
                        treatment_group.append(interaction)
                    else:
                        control_group.append(interaction)

            if len(treatment_group) < 3 or len(control_group) < 3:
                pattern.status = "REJECTED"
                print(f"Pattern {pattern.id} rejected: Could not form both treatment ({len(treatment_group)}) and control ({len(control_group)}) groups.")
                continue

            # 6. Perform Statistical Test
            treatment_outcomes = [1 if i.is_success else 0 for i in treatment_group]
            control_outcomes = [1 if i.is_success else 0 for i in control_group]
            
            # Ensure there is variance to avoid statistical errors
            if len(set(treatment_outcomes)) < 2 or len(set(control_outcomes)) < 2:
                pattern.status = "REJECTED"
                print(f"Pattern {pattern.id} rejected: No variance in outcomes for statistical test.")
                continue

            t_stat, p_value = ttest_ind(treatment_outcomes, control_outcomes, equal_var=False)
            uplift = np.mean(treatment_outcomes) - np.mean(control_outcomes)

            print(f"Pattern {pattern.id}: Uplift = {uplift*100:+.2f}%, P-Value = {p_value:.4f}")

            if p_value < 0.05 and uplift > 0: # Stricter p-value for production
                pattern.status = "VALIDATED"
                pattern.uplift_score = uplift
                pattern.p_value = p_value
                print(f"  -> RESULT: VALIDATED")
            else:
                pattern.status = "REJECTED"
                print(f"  -> RESULT: REJECTED")

        upload.status = "COMPLETED"
        await db.commit()
        print(f"\nValidation complete for upload {upload_id}. Final status: COMPLETED.")

    await engine.dispose()

async def _async_process_human_interaction(agent_id: str, recording_url: str, context: Optional[Dict[str, Any]], explicit_outcome: Optional[Dict[str, Any]], outcome_goal: Optional[str]):
    engine = create_async_engine(settings.DATABASE_URL)
    AsyncSessionLocal_Task = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with AsyncSessionLocal_Task() as db:
        transcript = await transcription_service.transcribe_audio_from_url(recording_url)
        if "Error transcribing" in transcript: return

        is_success = bool(explicit_outcome.get("success", False)) if explicit_outcome else None
        if outcome_goal and transcript:
            is_success = (await llm_service.get_json_response(
                "You are an AI evaluator...", f"GOAL: \"{outcome_goal}\"\nTRANSCRIPT:\n{transcript}"
            )).get("is_success", False)
        
        agent = await db.get(Agent, uuid.UUID(agent_id))
        if not agent: return

        db.add(HumanInteraction(
            agent_id=agent.id, organization_id=agent.organization_id, recording_url=recording_url,
            context=context, transcript=transcript, is_success=is_success, status="PROCESSED"
        ))
        await db.commit()
    await engine.dispose()

# --- CELERY TASK DEFINITIONS ---
@celery_app.task(name='app.background.tasks.process_historical_upload_task')
def process_historical_upload_task(upload_id: str, file_content_bytes: bytes, data_mapping: dict):
    run_async_task(_async_process_historical_upload, upload_id, file_content_bytes, data_mapping)

@celery_app.task(name='app.background.tasks.split_historical_data_task')
def split_historical_data_task(upload_id: str):
    run_async_task(_async_split_historical_data, upload_id)

@celery_app.task(name='app.background.tasks.extract_patterns_from_history_task')
def extract_patterns_from_history_task(upload_id: str):
    run_async_task(_async_extract_patterns_from_history, upload_id)

@celery_app.task(name='app.background.tasks.validate_patterns_task')
def validate_patterns_task(upload_id: str):
    run_async_task(_async_validate_patterns, upload_id)

@celery_app.task(name='app.background.tasks.process_human_interaction_task')
def process_human_interaction_task(agent_id: str, recording_url: str, context: Optional[Dict[str, Any]], explicit_outcome: Optional[Dict[str, Any]], outcome_goal: Optional[str]):
    run_async_task(_async_process_human_interaction, agent_id, recording_url, context, explicit_outcome, outcome_goal)

@celery_app.task(name='app.background.tasks.process_live_outcome_task')
def process_live_outcome_task(interaction_id: str):
    with get_sync_db_session() as db:
        stmt = select(Interaction).options(selectinload(Interaction.outcome)).where(Interaction.id == uuid.UUID(interaction_id))
        interaction = db.execute(stmt).scalars().first()
        if not interaction or not interaction.outcome or not interaction.applied_pattern_id: return
        
        pattern = db.get(LearnedPattern, interaction.applied_pattern_id)
        if pattern:
            pattern.impressions = (pattern.impressions or 0) + 1
            if interaction.outcome.is_success:
                pattern.success_count = (pattern.success_count or 0) + 1
        db.commit()

@celery_app.task(name='app.background.tasks.discover_patterns_from_live_data_task')
def discover_patterns_from_live_data_task(agent_id: str):
    """Celery wrapper for the live learning engine."""
    print(f"Starting live pattern discovery for agent ID: {agent_id}")
    run_async_task(_async_discover_patterns_from_live_data, agent_id)

@celery_app.task(name='app.background.tasks.generate_opportunities_task')
def generate_opportunities_task(organization_id: str):
    print(f"Opportunity generation for org {organization_id} is not yet implemented.")
    pass