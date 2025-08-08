import uuid
import json
import pandas as pd
from io import StringIO
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.celery_app import celery_app
from contextlib import contextmanager
from app.database import get_sync_db

get_sync_db_session = contextmanager(get_sync_db)

# Import all models
from app.models.historical_upload import HistoricalUpload
from app.models.historical_interaction import HistoricalInteraction
from app.models.learned_pattern import LearnedPattern
from app.models.interaction import Interaction
from app.models.outcome import Outcome
from app.models.suggested_opportunity import SuggestedOpportunity

# Import services (note: these will need sync versions or careful handling)
from app.services import embedding_service, llm_service
import numpy as np
from sklearn.cluster import DBSCAN
import asyncio # We need asyncio only to run the async services

# --- SYNCHRONOUS CELERY TASKS ---


@celery_app.task
def extract_patterns_from_history_task(upload_id: str):
    """
    Analyzes historical data using a more robust pattern finding logic.
    """
    print(f"Starting ADVANCED pattern extraction for upload ID: {upload_id}")
    with get_sync_db_session() as db:
        stmt = select(HistoricalInteraction).options(selectinload(HistoricalInteraction.upload)).where(
            HistoricalInteraction.upload_id == uuid.UUID(upload_id),
            HistoricalInteraction.is_success == True
        )
        successful_interactions = db.execute(stmt).scalars().all()
        
        print(f"Found {len(successful_interactions)} successful interactions to analyze.")

        if not successful_interactions:
            return

        # --- NEW, MORE ROBUST LOGIC ---
        # Instead of complex clustering on a tiny dataset, we will use a simpler,
        # more direct "group by" approach for the initial pattern discovery.
        # This is more reliable for small amounts of data.

        context_groups = {}
        for inter in successful_interactions:
            # We will group by the `customer_type` context key, if it exists.
            customer_type = (inter.original_context or {}).get("customer_type")
            if customer_type:
                if customer_type not in context_groups:
                    context_groups[customer_type] = []
                context_groups[customer_type].append(inter)
        
        print(f"Grouped interactions into {len(context_groups)} context groups: {list(context_groups.keys())}")

        if not context_groups:
            print("Could not group interactions by a common context key. Exiting.")
            return

        # AI Service calls are async, so we run them in an event loop
        async def perform_ai_abstraction():
            patterns_to_create = []
            for context_key, interactions_in_group in context_groups.items():
                # Get a sample of successful responses from this group
                sample_responses = [inter.original_response for inter in interactions_in_group[:5]]
                
                system_prompt = """
                You are a data analyst. You will be given a list of successful agent responses that were all used in very similar situations. 
                Your task is to identify the core strategy being used and to summarize the situation (the trigger).
                Respond in a valid JSON format with two keys: "trigger_context_summary" and "suggested_strategy".
                """
                user_prompt = f"""
                Here are some successful agent responses for a similar context:
                ---
                {json.dumps(sample_responses, indent=2)}
                ---
                Based on these, what is the core strategy, and what is a one-sentence summary of the context that triggers it?
                """
                
                pattern_json = await llm_service.get_json_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model="openai/gpt-4o"
                )

                if pattern_json and "trigger_context_summary" in pattern_json and "suggested_strategy" in pattern_json:
                    print(f"LLM abstracted new pattern for context: {context_key}")
                    
                    # We will create the object but not add it to the session yet
                    patterns_to_create.append(
                        LearnedPattern(
                            agent_id=interactions_in_group[0].upload.agent_id, # Get agent_id from the interaction
                            source="HISTORICAL",
                            trigger_context_summary=pattern_json["trigger_context_summary"],
                            suggested_strategy=pattern_json["suggested_strategy"],
                            status="ACTIVE",
                            impressions=len(interactions_in_group),
                            success_count=len(interactions_in_group)
                        )
                    )
            return patterns_to_create

        # Run the async AI part
        new_patterns = asyncio.run(perform_ai_abstraction())

        # Now, back in the sync world, save the results
        if new_patterns:
            print(f"Saving {len(new_patterns)} new patterns to the database.")
            db.add_all(new_patterns)
            db.commit()
        else:
            print("No new patterns were generated by the LLM.")



@celery_app.task
def process_historical_upload_task(upload_id: str, file_content_bytes: bytes, data_mapping: dict):
    """
    Processes a historical data file using sync operations with robust parsing.
    """
    with get_sync_db_session() as db:
        print(f"Starting to process historical upload with ID: {upload_id}")
        upload = db.get(HistoricalUpload, uuid.UUID(upload_id))
        if not upload: return

        try:
            csv_content = file_content_bytes.decode('utf-8')
            df = pd.read_csv(StringIO(csv_content))

            print(f"Pandas DataFrame created with {len(df)} rows. Columns: {list(df.columns)}")

            interactions_to_create = []
            
            # --- ROBUST MAPPING AND PARSING LOGIC ---
            outcome_col = data_mapping.get("outcome")
            transcript_col = data_mapping.get("conversation_transcript")
            context_cols = {key.replace("context_", ""): value for key, value in data_mapping.items() if key.startswith("context_")}

            for index, row in df.iterrows():
                # Build context object
                context = {k: row.get(v) for k, v in context_cols.items() if v in row}
                
                # Robustly determine outcome
                raw_outcome = ""
                if outcome_col and outcome_col in row:
                    raw_outcome = str(row[outcome_col]).strip().lower()
                
                is_success = raw_outcome in ['true', 'success', '1', 'yes']
                
                # Get transcript/response
                response_text = ""
                if transcript_col and transcript_col in row:
                    response_text = str(row[transcript_col])

                # DEBUG LOGGING: Print what we've parsed for each row
                print(f"Row {index}: Raw Outcome='{raw_outcome}', Parsed Success={is_success}, Context={context}")

                interactions_to_create.append(
                    HistoricalInteraction(
                        upload_id=upload.id,
                        original_context=context,
                        original_response=response_text,
                        is_success=is_success,
                        # Correctly populate extracted_outcome
                        extracted_outcome={"value_from_file": raw_outcome}
                    )
                )
            
            db.add_all(interactions_to_create)

            upload.status = "COMPLETED"
            upload.total_interactions = len(df)
            upload.processed_interactions = len(df)
            upload.processing_completed_timestamp = datetime.now(timezone.utc)
            db.commit()

            print(f"Successfully parsed {len(df)} interactions. Triggering pattern extraction.")
            extract_patterns_from_history_task.delay(upload_id)

        except Exception as e:
            print(f"FATAL Error processing file for upload {upload.id}: {e}")
            upload.status = "FAILED"
            db.commit()

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
        
        db.commit()
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
        db.commit()


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
        db.commit()
        print(f"Opportunity discovery complete for organization ID: {organization_id}")
