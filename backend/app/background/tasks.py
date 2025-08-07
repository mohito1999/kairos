import uuid
import pandas as pd
import tempfile
import shutil
import os
import asyncio
from io import StringIO
from sqlalchemy import select
from sqlalchemy.orm import selectinload, Session
from contextlib import contextmanager
from datetime import datetime, timezone 

from app.core.celery_app import celery_app
from app.database import SyncSessionLocal, AsyncSessionLocal

# Import all models needed for the tasks
from app.models.historical_upload import HistoricalUpload
from app.models.historical_interaction import HistoricalInteraction
from app.models.learned_pattern import LearnedPattern
from app.models.interaction import Interaction
from app.models.outcome import Outcome
from app.models.suggested_opportunity import SuggestedOpportunity

# --- SYNCHRONOUS CELERY TASKS ---

@contextmanager
def get_sync_db_session() -> Session:
    """Provides a transactional scope around a series of operations."""
    db = SyncSessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@celery_app.task
def extract_patterns_from_history_task(upload_id: str):
    """
    Analyzes a completed historical upload to find and create patterns.
    Uses synchronous SQLAlchemy operations.
    """
    print(f"Starting pattern extraction for upload ID: {upload_id}")
    with get_sync_db_session() as db:
        stmt = select(HistoricalUpload).options(selectinload(HistoricalUpload.interactions)).where(HistoricalUpload.id == uuid.UUID(upload_id))
        upload = db.execute(stmt).scalars().first()
        if not upload: return

        successful_interactions = [inter for inter in upload.interactions if inter.is_success]
        patterns_found = {}
        for inter in successful_interactions:
            customer_type = (inter.original_context or {}).get("customer_type")
            if customer_type and customer_type not in patterns_found:
                patterns_found[customer_type] = inter.original_response

        for context_summary, strategy in patterns_found.items():
            new_pattern = LearnedPattern(agent_id=upload.agent_id, source="HISTORICAL", trigger_context_summary=context_summary, suggested_strategy=strategy, status="ACTIVE", impressions=1, success_count=1)
            db.add(new_pattern)
        
        print(f"Pattern extraction complete for upload ID: {upload_id}")


@celery_app.task
def process_historical_upload_task(upload_id: str, file_content_bytes: bytes, data_mapping: dict):
    """
    Processes a historical data file from in-memory bytes.
    Parses the CSV, creates HistoricalInteraction records, and triggers pattern extraction.
    """
    async def _run():
        async with AsyncSessionLocal() as db:
            print(f"Starting to process historical upload with ID: {upload_id}")
            upload = await db.get(HistoricalUpload, uuid.UUID(upload_id))
            if not upload:
                print(f"Upload {upload_id} not found.")
                return

            try:
                # Use pandas to read the CSV content from the in-memory bytes
                csv_content = file_content_bytes.decode('utf-8')
                df = pd.read_csv(StringIO(csv_content))

                interactions_to_create = []
                for index, row in df.iterrows():
                    # Dynamically build the context object from the mapping
                    context = {}
                    for key, value in data_mapping.items():
                        if key.startswith("context_") and value in row:
                            context_key = key.replace("context_", "")
                            context[context_key] = row[value]
                    
                    # Determine outcome
                    is_success = False
                    outcome_column = data_mapping.get("outcome")
                    if outcome_column and outcome_column in row:
                        # Simple success check, can be made more robust
                        is_success = str(row[outcome_column]).lower() in ['true', 'success', '1', 'yes']
                    
                    # Get transcript/response
                    transcript_column = data_mapping.get("conversation_transcript")
                    response_text = row.get(transcript_column, "")

                    interactions_to_create.append(
                        HistoricalInteraction(
                            upload_id=upload.id,
                            original_context=context,
                            original_response=response_text,
                            is_success=is_success,
                            extracted_outcome={"outcome": str(row.get(outcome_column, ''))}
                        )
                    )
                
                # Bulk insert all interactions for efficiency
                db.add_all(interactions_to_create)

                upload.status = "COMPLETED"
                upload.total_interactions = len(df)
                upload.processed_interactions = len(df)
                upload.processing_completed_timestamp = datetime.now(timezone.utc)
                print(f"Successfully parsed {len(df)} interactions from file.")

            except Exception as e:
                print(f"Error processing file for upload {upload.id}: {e}")
                upload.status = "FAILED"
            
            await db.commit()
            
            if upload.status == "COMPLETED":
                print(f"Triggering pattern extraction for upload ID: {upload_id}")
                extract_patterns_from_history_task.delay(upload_id)

    # We must import StringIO for this task
    from io import StringIO
    return asyncio.run(_run())


@celery_app.task
def discover_patterns_from_live_data_task(agent_id: str):
    """
    Analyzes recent live interactions for a given agent to find new patterns.
    Uses synchronous SQLAlchemy operations.
    """
    print(f"Starting live pattern discovery for agent ID: {agent_id}")
    with get_sync_db_session() as db:
        stmt = select(Interaction).join(Outcome).where(
            Interaction.agent_id == uuid.UUID(agent_id), Outcome.is_success == True
        ).limit(1000)
        successful_interactions = db.execute(stmt).scalars().all()

        if len(successful_interactions) < 20:
            print("Not enough recent successful interactions to discover new patterns.")
            return

        context_groups = {}
        for inter in successful_interactions:
            if inter.context and 'occasion' in inter.context:
                key = inter.context['occasion']
                if key not in context_groups: context_groups[key] = []
                if inter.full_transcript and "Agent:" in inter.full_transcript:
                    agent_response = inter.full_transcript.split("Agent:")[-1].strip()
                    context_groups[key].append(agent_response)
        
        for occasion, responses in context_groups.items():
            if not responses: continue
            most_common_response = max(set(responses), key=responses.count)
            
            existing_pattern = db.execute(select(LearnedPattern).where(
                LearnedPattern.agent_id == uuid.UUID(agent_id),
                LearnedPattern.trigger_context_summary == occasion
            )).scalars().first()
            if existing_pattern: continue

            print(f"Discovered new candidate pattern for occasion: {occasion}")
            new_pattern = LearnedPattern(
                agent_id=uuid.UUID(agent_id), source="LIVE_DISCOVERED",
                trigger_context_summary=occasion, suggested_strategy=most_common_response,
                status="CANDIDATE"
            )
            db.add(new_pattern)
        
        print(f"Live pattern discovery complete for agent ID: {agent_id}")


@celery_app.task
def process_live_outcome_task(interaction_id: str):
    """
    Updates the performance statistics for the pattern used in a live interaction.
    Uses synchronous SQLAlchemy operations.
    """
    print(f"Processing outcome for interaction ID: {interaction_id}")
    with get_sync_db_session() as db:
        stmt = select(Interaction).options(selectinload(Interaction.outcome)).where(Interaction.id == uuid.UUID(interaction_id))
        interaction = db.execute(stmt).scalars().first()

        if not interaction or not interaction.outcome: return
        
        if interaction.applied_pattern_id:
            pattern = db.get(LearnedPattern, interaction.applied_pattern_id)
            if pattern:
                pattern.impressions += 1
                if interaction.outcome.is_success:
                    pattern.success_count += 1
                print(f"Updated stats for pattern {pattern.id}")


@celery_app.task
def generate_opportunities_task(organization_id: str):
    """
    Analyzes failed interactions to suggest new opportunities.
    Uses synchronous SQLAlchemy operations.
    """
    print(f"Starting opportunity discovery for organization ID: {organization_id}")
    with get_sync_db_session() as db:
        # This is a placeholder for the complex analysis logic.
        # A real implementation would fetch failed interactions for the org,
        # cluster their transcripts, and use an LLM to identify latent needs.
        
        # Example: if we found a new opportunity
        # new_opportunity = SuggestedOpportunity(
        #     organization_id=uuid.UUID(organization_id),
        #     title="Consider a 'Driving Range' offering",
        #     description="Analysis of failed calls shows 15% of users ask about a driving range.",
        #     suggested_action="Add a 'Driving Range Bundle' to your agent's core knowledge.",
        #     source="LATENT_NEED"
        # )
        # db.add(new_opportunity)
        
        print(f"Opportunity discovery complete for organization ID: {organization_id}")