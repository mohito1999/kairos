import uuid
import json
import pandas as pd
from io import StringIO
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import Dict, Any, Optional

from app.core.celery_app import celery_app
# Correctly import our definitive session manager
from app.database import get_sync_db_session

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
# CRITICAL FIX: Import the singleton instance, not the class
from app.services.transcription_service import transcription_service
import numpy as np
from sklearn.cluster import DBSCAN
import asyncio

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
    Processes a historical data file with flexible context mapping,
    optional LLM-as-a-judge for outcome determination, and robust parsing.
    """
    # This async sub-function will handle the AI calls for outcome judging
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
                system_prompt=system_prompt, user_prompt=user_prompt, model="openai/gpt-4o"
            )
            return assessment.get("is_success", False)
        except Exception:
            return False

    # The main task logic remains synchronous
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
            
            # --- PRE-PROCESS THE DATA MAPPING ---
            outcome_col = data_mapping.get("outcome_column")
            outcome_goal = data_mapping.get("outcome_goal_description")
            transcript_col = data_mapping.get("conversation_transcript")
            context_cols = {key: value for key, value in data_mapping.items() if key.startswith("context_")}

            # --- VALIDATE MAPPED COLUMNS EXIST IN THE DATAFRAME ---
            if outcome_col and outcome_col not in df.columns:
                print(f"Warning: Outcome column '{outcome_col}' not found in CSV. Outcomes will be marked as False.")
                outcome_col = None # Invalidate it
            if transcript_col and transcript_col not in df.columns:
                print(f"Warning: Transcript column '{transcript_col}' not found in CSV.")
                transcript_col = None

            # --- ITERATE AND PROCESS EACH ROW ---
            for index, row in df.iterrows():
                # 1. Dynamically build the context object (already correct)
                context = {
                    key.replace("context_", ""): row.get(value)
                    for key, value in context_cols.items() if value in df.columns
                }
                
                # 2. Get transcript/response text (CORRECTED)
                response_text = str(row.get(transcript_col, "")) if transcript_col else ""
                
                # 3. Determine the outcome using the "either/or" logic (CORRECTED)
                is_success = False
                raw_outcome = ""
                if outcome_col:
                    # Method A: Use the outcome column with the safe .get() method
                    raw_outcome_value = str(row.get(outcome_col, ""))
                    cleaned_outcome = raw_outcome_value.strip().strip('"').strip("'").lower()
                    is_success = cleaned_outcome in ['true', 'success', '1', 'yes', 'resolved']
                    raw_outcome = raw_outcome_value # Keep original for logging
                elif outcome_goal and response_text:
                    # Method B: Use LLM-as-a-judge
                    print(f"Row {index}: Using AI Judge for outcome...")
                    is_success = asyncio.run(_judge_outcome(response_text, outcome_goal))
                    raw_outcome = "judged_by_ai"
                
                # DEBUG LOGGING
                print(f"Row {index}: Raw Outcome='{raw_outcome}', Parsed Success={is_success}, Context={context}")

                interactions_to_create.append(
                    HistoricalInteraction(
                        upload_id=upload.id,
                        original_context=context,
                        original_response=response_text,
                        is_success=is_success,
                        extracted_outcome={"value_from_file": raw_outcome}
                    )
                )
            
            # Bulk insert all interactions for efficiency
            db.add_all(interactions_to_create)

            upload.status = "COMPLETED"
            upload.total_interactions = len(df)
            upload.processed_interactions = len(df)
            upload.processing_completed_timestamp = datetime.now(timezone.utc)
            db.commit()

            print(f"Successfully parsed and saved {len(df)} interactions. Triggering pattern extraction.")
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


@celery_app.task
def process_human_interaction_task(
    agent_id: str,
    recording_url: str,
    context: Optional[Dict[str, Any]],
    explicit_outcome: Optional[Dict[str, Any]],
    outcome_goal: Optional[str]
):
    """
    Processes a recorded human interaction: transcribes, determines outcome, and saves it.
    Uses a sync DB session but calls async AI services.
    """
    
    # This async sub-function will handle the AI calls
    async def get_transcript_and_assessment():
        print(f"Transcribing audio for agent {agent_id} from {recording_url}")
        transcript = await transcription_service.transcribe_audio_from_url(recording_url)
        
        if "Error transcribing" in transcript:
            return transcript, None # Return error and no assessment

        # If we need to judge the outcome, do it here
        is_success = None
        if explicit_outcome:
            is_success = bool(explicit_outcome.get("success", False))
        elif outcome_goal:
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
            user_prompt = f"SUCCESS GOAL: \"{outcome_goal}\"\n\nCONVERSATION TRANSCRIPT:\n---\n{transcript}\n---"
            
            print("Performing AI-assisted outcome assessment...")
            assessment = await llm_service.get_json_response(
                system_prompt=system_prompt, user_prompt=user_prompt, model="openai/gpt-4o"
            )
            is_success = assessment.get("is_success", False)
        
        return transcript, is_success

    # Run the async part to get the data we need
    transcript, is_success = asyncio.run(get_transcript_and_assessment())

    # Now, back in the sync world, interact with the database
    if "Error transcribing" in transcript:
        print(f"Failed to transcribe audio. Aborting database operations.")
        return

    with get_sync_db_session() as db:
        try:
            agent = db.get(Agent, uuid.UUID(agent_id))
            if not agent:
                print(f"Agent {agent_id} not found in database. Cannot save interaction.")
                return

            # Create the record in our new human_interactions table
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
            db.commit()
            print(f"Human interaction {new_interaction_record.id} saved and processed successfully.")
        except Exception as e:
            print(f"Database error while saving human interaction: {e}")