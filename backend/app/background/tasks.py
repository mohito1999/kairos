import uuid
import json
import numpy as np
import pandas as pd
from io import StringIO
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import asyncio
import sqlalchemy as sa
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from app.core.celery_app import celery_app
from app.database import get_sync_db_session, AsyncSessionLocal

# Import all models
from app.models.agent import Agent
from app.models.historical_upload import HistoricalUpload
from app.models.historical_interaction import HistoricalInteraction
from app.models.human_interaction import HumanInteraction
from app.models.learned_pattern import LearnedPattern
from app.models.interaction import Interaction
from app.models.outcome import Outcome
from app.models.suggested_opportunity import SuggestedOpportunity

# Import services
from app.services import embedding_service, llm_service
from app.services.transcription_service import transcription_service

# Scientific Libraries
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity


# --- HELPER FUNCTIONS ---

async def is_duplicate_pattern(new_pattern_strategy: str, agent_id: uuid.UUID) -> bool:
    """Checks if a new pattern is semantically similar to an existing ACTIVE one."""
    try:
        async with AsyncSessionLocal() as db:
            stmt = select(LearnedPattern).where(LearnedPattern.agent_id == agent_id, LearnedPattern.status == "ACTIVE")
            result = await db.execute(stmt)
            existing_patterns = result.scalars().all()

            if not existing_patterns: return False

            new_embedding_list = await embedding_service.get_embeddings([new_pattern_strategy])
            if not new_embedding_list or not new_embedding_list[0]:
                print("Warning: Could not generate embedding for new pattern. Cannot check for duplicates.")
                return False # Fail open

            new_embedding = new_embedding_list[0]
            existing_strategy_texts = [p.suggested_strategy for p in existing_patterns]
            existing_strategy_embeddings = await embedding_service.get_embeddings(existing_strategy_texts)
            
            similarity_threshold = 0.95
            for i, existing_embedding in enumerate(existing_strategy_embeddings):
                if not existing_embedding: continue
                similarity = cosine_similarity([new_embedding], [existing_embedding])[0][0]
                if similarity > similarity_threshold:
                    print(f"New pattern is a likely duplicate of existing pattern '{existing_strategy_texts[i]}' (similarity: {similarity:.2f}). Skipping.")
                    return True
            return False
    except Exception as e:
        print(f"Error checking for duplicate pattern: {e}")
        return False # Fail open

def is_quality_pattern(pattern_json: dict) -> bool:
    """Performs quality checks on a generated pattern."""
    strategy = pattern_json.get("suggested_strategy", "").lower()
    trigger = pattern_json.get("trigger_context_summary", "").lower()

    generic_phrases = ["be helpful", "assist the customer", "provide support", "be nice", "respond appropriately"]
    if any(phrase in strategy for phrase in generic_phrases):
        print(f"Quality Check Failed: Strategy '{strategy}' is too generic.")
        return False
    if len(strategy.split()) < 4:
        print(f"Quality Check Failed: Strategy '{strategy}' is too short.")
        return False
    if len(trigger.split()) < 3:
        print(f"Quality Check Failed: Trigger '{trigger}' is too short.")
        return False
    return True

# --- PRIMARY CELERY TASKS: HISTORICAL DATA PIPELINE ---

@celery_app.task(name='app.background.tasks.process_historical_upload_task')
def process_historical_upload_task(upload_id: str, file_content_bytes: bytes, data_mapping: dict):
    """
    Processes a historical data file with flexible context mapping,
    optional LLM-as-a-judge for outcome determination, and robust parsing.
    This is the first task in the historical upload pipeline.
    """
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        async def _judge_outcome(transcript: str, goal: str) -> bool:
            if not transcript or not goal:
                return False
            
            system_prompt = """
            You are a meticulous and objective AI evaluator. Your task is to determine if a conversation successfully met a specific goal.
            Analyze the provided transcript and the success goal, then respond with a JSON object containing your assessment.
            The JSON object MUST have a key "is_success" with a boolean value.
            Only return a valid JSON object.
            """
            user_prompt = f"SUCCESS GOAL: \"{goal}\"\n\nCONVERSATION TRANSCRIPT:\n---\n{transcript}\n---"
            
            try:
                assessment = await llm_service.get_json_response(
                    system_prompt=system_prompt, user_prompt=user_prompt, model="openai/gpt-4o-mini"
                )
                return assessment.get("is_success", False)
            except Exception as e:
                print(f"Error in AI outcome judgment: {e}")
                return False

        with get_sync_db_session() as db:
            print(f"Starting to process historical upload with ID: {upload_id}")
            upload = db.get(HistoricalUpload, uuid.UUID(upload_id))
            if not upload:
                print(f"Upload {upload_id} not found.")
                return

            try:
                csv_content = file_content_bytes.decode('utf-8', errors='ignore')
                df = pd.read_csv(StringIO(csv_content))
                print(f"Pandas DataFrame created with {len(df)} rows. Columns: {list(df.columns)}")

                interactions_to_create = []
                
                outcome_col = data_mapping.get("outcome_column")
                outcome_goal = data_mapping.get("outcome_goal_description")
                transcript_col = data_mapping.get("conversation_transcript")
                context_cols = {key: value for key, value in data_mapping.items() if key.startswith("context_")}

                if outcome_col and outcome_col not in df.columns:
                    print(f"Warning: Outcome column '{outcome_col}' not found in CSV. Outcomes may be marked as False.")
                    outcome_col = None
                if transcript_col and transcript_col not in df.columns:
                    print(f"FATAL: Transcript column '{transcript_col}' not found in CSV. Cannot process.")
                    upload.status = "FAILED"
                    db.commit()
                    return

                for index, row in df.iterrows():
                    context = {
                        key.replace("context_", ""): row.get(value)
                        for key, value in context_cols.items() if value in df.columns
                    }
                    response_text = str(row.get(transcript_col, ""))
                    
                    is_success = False
                    raw_outcome = ""
                    if outcome_col:
                        raw_outcome_value = str(row.get(outcome_col, ""))
                        cleaned_outcome = raw_outcome_value.strip().strip('"').strip("'").lower()
                        is_success = cleaned_outcome in ['true', 'success', '1', 'yes', 'resolved']
                        raw_outcome = raw_outcome_value
                    elif outcome_goal and response_text:
                        print(f"Row {index}: Using AI Judge for outcome...")
                        is_success = loop.run_until_complete(_judge_outcome(response_text, outcome_goal))
                        raw_outcome = "judged_by_ai"
                    
                    interactions_to_create.append(
                        HistoricalInteraction(
                            upload_id=upload.id,
                            original_context=context,
                            original_response=response_text,
                            is_success=is_success,
                            extracted_outcome={"value_from_file": raw_outcome}
                        )
                    )
                
                db.add_all(interactions_to_create)
                upload.status = "PARSED"
                upload.total_interactions = len(df)
                upload.processed_interactions = len(df)
                db.commit()

                print(f"Successfully parsed and saved {len(df)} interactions. Triggering pattern extraction.")
                celery_app.send_task(
                    'app.background.tasks.extract_patterns_from_history_task',
                    args=[upload_id]
                )

            except Exception as e:
                print(f"FATAL Error processing file for upload {upload.id}: {e}")
                upload.status = "FAILED"
                db.commit()
    except Exception as e:
        print(f"FATAL ERROR in upload processing task: {e}")
    finally:
        try:
            loop.close()
        except Exception as cleanup_error:
            print(f"Error during loop cleanup: {cleanup_error}")
        finally:
            asyncio.set_event_loop(None)

@celery_app.task(name='app.background.tasks.extract_patterns_from_history_task')
def extract_patterns_from_history_task(upload_id: str):
    """
    The Contrastive Expertise Distillation Engine.
    This task discovers "battlegrounds" from failures and then performs
    contrastive analysis to find winning strategies.
    """
    print(f"Starting CONTRASTIVE pattern extraction for upload ID: {upload_id}")
    
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(
            _async_extract_patterns_from_history(upload_id)
        )
    except Exception as e:
        print(f"FATAL ERROR during async execution of pattern extraction: {e}")
    finally:
        if not asyncio.get_event_loop().is_running():
             asyncio.get_event_loop().close()

async def _async_extract_patterns_from_history(upload_id: str):
    """The async core of the pattern extraction task."""
    async with AsyncSessionLocal() as db:
        upload_stmt = select(HistoricalUpload).where(HistoricalUpload.id == uuid.UUID(upload_id))
        upload = (await db.execute(upload_stmt)).scalars().first()
        if not upload:
            print(f"Upload {upload_id} not found."); return

        # --- A. OPPORTUNITY DISCOVERY: Cluster failures to find battlegrounds ---
        failed_stmt = select(HistoricalInteraction).where(
            HistoricalInteraction.upload_id == upload.id,
            HistoricalInteraction.is_success == False,
            HistoricalInteraction.original_response.is_not(None),
            HistoricalInteraction.original_response != ''
        )
        failed_interactions = (await db.execute(failed_stmt)).scalars().all()

        if len(failed_interactions) < 10:
            print("Not enough failed interactions to discover significant battlegrounds.")
            upload.status = "COMPLETED"
            await db.commit()
            return

        failed_transcripts = [inter.original_response for inter in failed_interactions]
        failed_embeddings = await embedding_service.get_embeddings(failed_transcripts)
        
        valid_embeddings = [emb for emb in failed_embeddings if emb]
        valid_interactions = [inter for i, inter in enumerate(failed_interactions) if failed_embeddings[i]]
        
        if len(valid_embeddings) < 5:
            print("Could not generate enough embeddings from failed interactions.")
            upload.status = "COMPLETED"
            await db.commit()
            return

        dbscan = DBSCAN(eps=0.5, min_samples=5, metric="cosine")
        clusters = dbscan.fit_predict(np.array(valid_embeddings))

        battlegrounds: Dict[int, List[HistoricalInteraction]] = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id != -1:
                battlegrounds.setdefault(cluster_id, []).append(valid_interactions[i])
        
        print(f"Discovered {len(battlegrounds)} potential battlegrounds from {len(failed_interactions)} failed interactions.")

        patterns_to_create = []
        # --- B. CONTRASTIVE ANALYSIS: For each battleground, find winners vs. losers ---
        for cluster_id, losing_interactions in battlegrounds.items():
            print(f"\n--- Analyzing Battleground #{cluster_id} ---")
            
            losing_centroid_embeddings = [emb for inter in losing_interactions for emb in await embedding_service.get_embeddings([inter.original_response]) if emb]
            if not losing_centroid_embeddings: continue
            losing_centroid = np.mean(losing_centroid_embeddings, axis=0)
            
            all_successful_stmt = select(HistoricalInteraction).where(
                HistoricalInteraction.upload_id == upload.id, HistoricalInteraction.is_success == True
            )
            all_successful_interactions = (await db.execute(all_successful_stmt)).scalars().all()
            if not all_successful_interactions: continue

            successful_transcripts = [i.original_response for i in all_successful_interactions]
            successful_embeddings = await embedding_service.get_embeddings(successful_transcripts)

            winning_interactions = []
            for i, emb in enumerate(successful_embeddings):
                if emb is not None:
                    similarity = cosine_similarity([losing_centroid], [emb])[0][0]
                    if similarity > 0.8:
                        winning_interactions.append(all_successful_interactions[i])

            print(f"Found {len(winning_interactions)} winning examples and {len(losing_interactions)} losing examples for this battleground.")

            if len(winning_interactions) < 3 or len(losing_interactions) < 3:
                print("Insufficient data for contrastive analysis. Skipping battleground.")
                continue

            # --- C. PATTERN CREATION: Use LLM to synthesize the winning strategy ---
            positive_snippets = [i.original_response for i in winning_interactions[:5]]
            negative_snippets = [i.original_response for i in losing_interactions[:5]]
            
            system_prompt = """
            You are an expert sales and customer service coach. You will be shown examples of agent conversations that led to failure, and conversations that led to success, all within the same high-stakes situation.
            Your task is to identify the core behavioral difference and distill the winning tactic into a clear, actionable strategy for a voice AI agent.
            Respond in a valid JSON format with "trigger_context_summary" and "suggested_strategy".
            - 'trigger_context_summary': A concise, one-sentence description of the situation or "battleground".
            - 'suggested_strategy': A specific, tactical instruction for the AI agent. Do NOT be generic.
            """
            user_prompt = f"""
            This is the situation:
            - CONTEXT: A customer interaction where the outcome is uncertain.

            These agent responses led to FAILURE:
            ---
            {json.dumps(negative_snippets, indent=2)}
            ---

            These agent responses led to SUCCESS in the same situation:
            ---
            {json.dumps(positive_snippets, indent=2)}
            ---

            Analyze the difference. What is the battleground, and what is the specific, winning strategy?
            """

            pattern_json = await llm_service.get_json_response(system_prompt=system_prompt, user_prompt=user_prompt, model="openai/gpt-4o")

            if pattern_json and is_quality_pattern(pattern_json) and not await is_duplicate_pattern(pattern_json["suggested_strategy"], upload.agent_id):
                print(f"Synthesized a new, high-quality pattern for battleground #{cluster_id}.")
                new_pattern = LearnedPattern(
                    agent_id=upload.agent_id,
                    source="HISTORICAL_CONTRASTIVE",
                    status="CANDIDATE",
                    source_upload_id=upload.id, # Link back to the upload
                    battleground_context={"cluster_id": cluster_id},
                    positive_examples={"transcripts": positive_snippets},
                    negative_examples={"transcripts": negative_snippets},
                    trigger_context_summary=pattern_json["trigger_context_summary"],
                    suggested_strategy=pattern_json["suggested_strategy"],
                )
                patterns_to_create.append(new_pattern)

        # --- Finalize and Save ---
        if patterns_to_create:
            print(f"\nDiscovered {len(patterns_to_create)} new candidate patterns. Saving to database.")
            db.add_all(patterns_to_create)
            upload.status = "VALIDATING"
            await db.commit()
            print("Triggering validation task...")
            celery_app.send_task('app.background.tasks.validate_patterns_task', args=[upload_id])
        else:
            print("\nNo new, high-quality patterns were discovered in this run.")
            upload.status = "COMPLETED"
            upload.processing_completed_timestamp = datetime.now(timezone.utc)
            await db.commit()

@celery_app.task(name='app.background.tasks.validate_patterns_task')
def validate_patterns_task(upload_id: str):
    """
    Validates candidate patterns. In a real system, this would involve statistical checks.
    For now, it promotes CANDIDATE patterns to ACTIVE to complete the pipeline.
    """
    print(f"Starting validation task for upload ID: {upload_id}")
    with get_sync_db_session() as db:
        upload = db.get(HistoricalUpload, uuid.UUID(upload_id))
        if not upload or upload.status != "VALIDATING":
            print(f"Upload {upload_id} not found or not in VALIDATING state. Aborting.")
            return

        stmt = update(LearnedPattern).where(
            LearnedPattern.source_upload_id == upload.id,
            LearnedPattern.status == 'CANDIDATE'
        ).values(status='ACTIVE')
        
        result = db.execute(stmt)
        db.commit()
        
        print(f"{result.rowcount} patterns for upload {upload_id} have been promoted to ACTIVE.")
        
        upload.status = "COMPLETED"
        upload.processing_completed_timestamp = datetime.now(timezone.utc)
        db.commit()
        print(f"Upload {upload_id} processing is now complete.")

# --- OTHER TASKS ---

@celery_app.task(name='app.background.tasks.process_live_outcome_task')
def process_live_outcome_task(interaction_id: str):
    print(f"Processing outcome for interaction ID: {interaction_id}")
    with get_sync_db_session() as db:
        stmt = select(Interaction).options(selectinload(Interaction.outcome)).where(Interaction.id == uuid.UUID(interaction_id))
        interaction = db.execute(stmt).scalars().first()

        if not interaction or not interaction.outcome: return
        
        if interaction.applied_pattern_id:
            pattern = db.get(LearnedPattern, interaction.applied_pattern_id)
            if pattern:
                pattern.impressions = (pattern.impressions or 0) + 1
                if interaction.outcome.is_success:
                    pattern.success_count = (pattern.success_count or 0) + 1
                print(f"Updated stats for pattern {pattern.id}")
        db.commit()

@celery_app.task(name='app.background.tasks.discover_patterns_from_live_data_task')
def discover_patterns_from_live_data_task(agent_id: str):
    print(f"Starting live pattern discovery for agent ID: {agent_id}")
    # This will eventually call a function similar to _async_extract_patterns_from_history
    # but using live Interaction data instead of HistoricalInteraction data.
    # For now, it remains a placeholder.
    pass

@celery_app.task(name='app.background.tasks.generate_opportunities_task')
def generate_opportunities_task(organization_id: str):
    print(f"Starting opportunity discovery for organization ID: {organization_id}")
    with get_sync_db_session() as db:
        # Placeholder for complex analysis logic
        db.commit()
        print(f"Opportunity discovery complete for organization ID: {organization_id}")

@celery_app.task(name='app.background.tasks.process_human_interaction_task')
def process_human_interaction_task(
    agent_id: str,
    recording_url: str,
    context: Optional[Dict[str, Any]],
    explicit_outcome: Optional[Dict[str, Any]],
    outcome_goal: Optional[str]
):
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(
            _async_process_human_interaction(agent_id, recording_url, context, explicit_outcome, outcome_goal)
        )
    except Exception as e:
        print(f"FATAL ERROR during async execution of human interaction processing: {e}")
    finally:
        if not asyncio.get_event_loop().is_running():
             asyncio.get_event_loop().close()

async def _async_process_human_interaction(agent_id: str, recording_url: str, context: Optional[Dict[str, Any]], explicit_outcome: Optional[Dict[str, Any]], outcome_goal: Optional[str]):
    async with AsyncSessionLocal() as db:
        print(f"Transcribing audio for agent {agent_id} from {recording_url}")
        transcript = await transcription_service.transcribe_audio_from_url(recording_url)
        if "Error transcribing" in transcript:
            print(f"Transcription failed for agent {agent_id}. Aborting.")
            return

        is_success = None
        if explicit_outcome:
            is_success = bool(explicit_outcome.get("success", False))
        elif outcome_goal and transcript:
            system_prompt = """
            You are a meticulous and objective AI evaluator. Your task is to determine if a conversation successfully met a specific goal.
            Analyze the provided transcript and the success goal, then respond with a JSON object containing your assessment.
            The JSON object MUST have a key "is_success" with a boolean value.
            """
            user_prompt = f"SUCCESS GOAL: \"{outcome_goal}\"\n\nCONVERSATION TRANSCRIPT:\n---\n{transcript}\n---"
            print("Performing AI-assisted outcome assessment...")
            try:
                assessment = await llm_service.get_json_response(system_prompt=system_prompt, user_prompt=user_prompt)
                is_success = assessment.get("is_success", False)
            except Exception as e:
                print(f"Error in AI outcome assessment: {e}")
                is_success = False
        
        agent = await db.get(Agent, uuid.UUID(agent_id))
        if not agent:
            print(f"Agent {agent_id} not found, cannot save interaction.")
            return

        new_interaction_record = HumanInteraction(
            agent_id=uuid.UUID(agent_id),
            organization_id=agent.organization_id,
            recording_url=recording_url,
            context=context,
            transcript=transcript,
            is_success=is_success,
            status="PROCESSED"
        )
        db.add(new_interaction_record)
        await db.commit()
        print(f"Human interaction {new_interaction_record.id} saved and processed.")