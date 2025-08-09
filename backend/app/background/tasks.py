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
# CRITICAL FIX: Import the singleton instance, not the class
from app.services.transcription_service import transcription_service
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import asyncio


# --- HELPER FUNCTION FOR DYNAMIC GROUPING ---

def find_best_grouping_key(interactions: list) -> str | None:
    """
    Analyzes a list of interactions to find the best categorical key for grouping.
    Returns the key (e.g., 'customer_type') or None if no good key is found.
    """
    # A prioritized list of common categorical keys we should look for.
    POTENTIAL_GROUPING_KEYS = ["customer_type", "inquiry_type", "category", "department", "priority", "region", "lead_source"]
    
    best_key = None
    max_coverage = 0.2  # Require at least 20% of data to be covered by meaningful groups

    for key in POTENTIAL_GROUPING_KEYS:
        groups = {}
        for inter in interactions:
            # Ensure context is a dict before calling .get()
            context = inter.original_context if isinstance(inter.original_context, dict) else {}
            value = context.get(key)
            if value and isinstance(value, (str, int)): # Only group by simple, hashable types
                groups.setdefault(value, []).append(inter)
        
        # A "good" key is one that creates multiple, non-trivial groups.
        # Let's say a meaningful group has at least 2 interactions.
        meaningful_groups = [g for g in groups.values() if len(g) >= 2]
        
        # We also want at least 2 distinct groups to be formed.
        if len(meaningful_groups) >= 2:
            # Calculate how much of the total data this key successfully groups.
            num_grouped_interactions = sum(len(g) for g in meaningful_groups)
            coverage = num_grouped_interactions / len(interactions)
            
            if coverage > max_coverage:
                max_coverage = coverage
                best_key = key
                
    if best_key:
        print(f"Found best grouping key: '{best_key}' with {max_coverage:.0%} coverage.")
    else:
        print("No suitable grouping key found. Proceeding to clustering.")
        
    return best_key

def get_clustering_params(data_size: int) -> dict:
    """
    Dynamically determines DBSCAN parameters based on the dataset size.
    """
    if data_size < 10:
        # For very small datasets, be lenient.
        # A cluster can be just 2 points that are reasonably close.
        return {"eps": 0.5, "min_samples": 2}
    elif data_size < 100:
        # For medium datasets, be slightly stricter.
        return {"eps": 0.45, "min_samples": 3}
    else: # For large datasets (100+)
        # Be much stricter to find only high-density, high-confidence clusters.
        return {"eps": 0.4, "min_samples": 5}

async def is_duplicate_pattern(new_pattern_strategy: str, agent_id: uuid.UUID) -> bool:
    """
    Checks if a new pattern is a semantic duplicate of an existing one for the agent.
    This function is fully self-contained and manages its own async DB session.
    """
    async with AsyncSessionLocal() as db: # Create its own async session
        # 1. Fetch all existing active patterns for this agent
        stmt = select(LearnedPattern).where(
            LearnedPattern.agent_id == agent_id,
            LearnedPattern.status == "ACTIVE"
        )
        result = await db.execute(stmt)
        existing_patterns = result.scalars().all()

        if not existing_patterns:
            return False

        # 2. Get embeddings for the new and existing strategies
        new_embedding_list = await embedding_service.get_embeddings([new_pattern_strategy])
        if not new_embedding_list or not new_embedding_list[0]:
            return False # Can't compare if embedding fails

        new_embedding = new_embedding_list[0]
        
        existing_strategy_texts = [p.suggested_strategy for p in existing_patterns]
        existing_strategy_embeddings = await embedding_service.get_embeddings(existing_strategy_texts)

        # 3. Calculate cosine similarity and check against a threshold
        similarity_threshold = 0.95
        
        for existing_embedding in existing_strategy_embeddings:
            if not existing_embedding: continue
            
            similarity = cosine_similarity([new_embedding], [existing_embedding])[0][0]
            
            if similarity > similarity_threshold:
                print(f"New pattern is a likely duplicate (similarity: {similarity:.2f}). Skipping.")
                return True

        return False


def is_quality_pattern(pattern_json: dict) -> bool:
    """
    Performs a series of checks to ensure a generated pattern is high-quality.
    Returns True if the pattern is good, False otherwise.
    """
    strategy = pattern_json.get("suggested_strategy", "").lower()
    trigger = pattern_json.get("trigger_context_summary", "").lower()

    # Check 1: Is the strategy too generic or vague?
    generic_phrases = [
        "be helpful", "assist the customer", "provide support", 
        "handle the request", "answer the question", "respond appropriately"
    ]
    if any(phrase in strategy for phrase in generic_phrases):
        print(f"Quality Check Failed: Strategy '{strategy}' is too generic.")
        return False

    # Check 2: Is the strategy too short to be meaningful?
    if len(strategy.split()) < 4: # Must be at least 4 words
        print(f"Quality Check Failed: Strategy '{strategy}' is too short.")
        return False
        
    # Check 3: Is the trigger summary too short or generic?
    if len(trigger.split()) < 3: # Must be at least 3 words
        print(f"Quality Check Failed: Trigger '{trigger}' is too short.")
        return False

    # All checks passed
    return True


# --- SYNCHRONOUS CELERY TASKS ---


@celery_app.task
def extract_patterns_from_history_task(upload_id: str):
    """
    Analyzes historical data using the full Hybrid Engine:
    1. A "group-by" sweep for obvious patterns.
    2. A "clustering" deep search for non-obvious patterns.
    """
    print(f"Starting HYBRID pattern extraction for upload ID: {upload_id}")
    
    with get_sync_db_session() as db:
        # Fetch the upload record to get the agent_id
        upload = db.get(HistoricalUpload, uuid.UUID(upload_id))
        if not upload: 
            print(f"Upload {upload_id} not found.")
            return

        # Fetch all successful interactions for this upload
        stmt = select(HistoricalInteraction).where(
            HistoricalInteraction.upload_id == uuid.UUID(upload_id),
            HistoricalInteraction.is_success == True
        )
        successful_interactions = db.execute(stmt).scalars().all()
        
        print(f"Found {len(successful_interactions)} successful interactions to analyze.")
        if not successful_interactions: return

        # This will store patterns from both stages
        patterns_to_create = []
        
        # --- STAGE 1: The "Sweep" (Group-By) ---
        print("Stage 1: Starting Dynamic Group-By Sweep for obvious patterns.")
        
        context_groups = {}
        grouped_interaction_ids = set()

        # Find the best key to group by from the available data
        best_grouping_key = find_best_grouping_key(successful_interactions)

        if best_grouping_key:
            for inter in successful_interactions:
                # Ensure context is a dict before calling .get()
                context = inter.original_context if isinstance(inter.original_context, dict) else {}
                value = context.get(best_grouping_key)
                if value:
                    context_groups.setdefault(value, []).append(inter)
            
            # Filter out groups that are too small to be meaningful
            meaningful_groups = {key: val for key, val in context_groups.items() if len(val) >= 2}
            
            # Mark which interactions have been successfully grouped
            for group in meaningful_groups.values():
                for inter in group:
                    grouped_interaction_ids.add(inter.id)

            context_groups = meaningful_groups # Re-assign to only include meaningful groups

        print(f"Grouped {len(grouped_interaction_ids)} interactions into {len(context_groups)} context groups using key '{best_grouping_key}'.")

        # Process grouped interactions with AI analysis
        async def perform_group_ai_analysis():
            try:
                group_patterns = []
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

                        if is_quality_pattern(pattern_json):
                            print(f"LLM abstracted new pattern for context: {context_key}")

                        
                            if not await is_duplicate_pattern(
                                new_pattern_strategy=pattern_json["suggested_strategy"],
                                agent_id=upload.agent_id
                            ):
                                group_patterns.append(
                                    LearnedPattern(
                                        agent_id=upload.agent_id,
                                        source="HISTORICAL",
                                        trigger_context_summary=pattern_json["trigger_context_summary"],
                                        suggested_strategy=pattern_json["suggested_strategy"],
                                        status="ACTIVE",
                                        impressions=len(interactions_in_group),
                                        success_count=len(interactions_in_group)
                                    )
                                )
                return group_patterns
            
            except Exception as e:
                print(f"FATAL ERROR during AI analysis: {e}")
                return []

        # Run the group-by analysis
        if context_groups:
            group_patterns = asyncio.run(perform_group_ai_analysis())
            patterns_to_create.extend(group_patterns)
            print(f"Stage 1 generated {len(group_patterns)} patterns from grouped data.")
        else:
            print("Stage 1: No interactions could be grouped by customer_type context key.")

        # Determine ungrouped interactions for Stage 2
        ungrouped_interactions = [inter for inter in successful_interactions if inter.id not in grouped_interaction_ids]
        print(f"Stage 1 complete. {len(ungrouped_interactions)} interactions remain ungrouped for deep search.")

        # --- STAGE 2: The "Deep Search" (Clustering) ---
        if len(ungrouped_interactions) < 3:
            print("Not enough remaining interactions for deep search clustering.")
        else:
            print(f"Stage 2: Starting Deep Search on {len(ungrouped_interactions)} interactions.")
            
            # AI Service calls are async, so we run them in an event loop
            async def perform_ai_analysis():
                try:
                    contexts_to_embed = [json.dumps(inter.original_context) for inter in ungrouped_interactions]
                    embeddings = await embedding_service.get_embeddings(contexts_to_embed)
                    
                    valid_embeddings = [emb for emb in embeddings if emb]
                    valid_interactions = [inter for i, inter in enumerate(ungrouped_interactions) if embeddings[i]]
                    
                    if len(valid_embeddings) < 2: # We can now attempt with as few as 2
                        print("Not enough valid embeddings for clustering.")
                        return []
                    
                    # --- ADAPTIVE PARAMETERS LOGIC ---
                    # Get the optimal parameters for our current dataset size
                    params = get_clustering_params(len(valid_embeddings))
                    print(f"Using adaptive clustering parameters for {len(valid_embeddings)} items: {params}")

                    # Perform clustering with the dynamic parameters
                    clustering = DBSCAN(
                        eps=params["eps"], 
                        min_samples=params["min_samples"], 
                        metric="cosine"
                    ).fit(np.array(valid_embeddings))
                    labels = clustering.labels_
                    
                    clusters = {}
                    for i, label in enumerate(labels):
                        if label != -1: # Ignore noise
                            if label not in clusters: clusters[label] = []
                            clusters[label].append(valid_interactions[i])

                    print(f"Deep Search found {len(clusters)} potential pattern clusters.")
                    
                    discovered_patterns = []
                    for label, interactions_in_cluster in clusters.items():
                        sample_responses = [inter.original_response for inter in interactions_in_cluster[:5]]
                        
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

                            if is_quality_pattern(pattern_json):
                                print(f"LLM abstracted new pattern for cluster {label}")

                                if not await is_duplicate_pattern(
                                    new_pattern_strategy=pattern_json["suggested_strategy"],
                                    agent_id=upload.agent_id
                                ):
                                    discovered_patterns.append(
                                        LearnedPattern(
                                            agent_id=upload.agent_id, 
                                            source="HISTORICAL_DISCOVERED",
                                            trigger_context_summary=pattern_json["trigger_context_summary"],
                                            suggested_strategy=pattern_json["suggested_strategy"], 
                                            status="ACTIVE",
                                            impressions=len(interactions_in_cluster), 
                                            success_count=len(interactions_in_cluster)
                                        )
                                    )
                    return discovered_patterns
                
                except Exception as e:
                    print(f"FATAL ERROR during AI analysis: {e}")
                    # In a production system, we'd add more specific error logging here (e.g., Sentry)
                    return []

            # Run the async part
            clustered_patterns = asyncio.run(perform_ai_analysis())
            patterns_to_create.extend(clustered_patterns)
            print(f"Stage 2 generated {len(clustered_patterns)} patterns from clustering.")

        # --- FALLBACK STRATEGY ---
        # If after all the advanced analysis, we still have no patterns,
        # let's try a simple, frequency-based fallback.
        if not patterns_to_create and len(successful_interactions) >= 3:
            print("No patterns found via hybrid engine. Attempting simple frequency-based fallback...")
            
            # Find the single most common successful response in the entire dataset.
            all_responses = [inter.original_response for inter in successful_interactions if inter.original_response]
            if all_responses:
                most_common_response = max(set(all_responses), key=all_responses.count)
                
                # We need to create a generic trigger summary for this.
                trigger_summary = "A general, high-performing response"

                # We still need to run our quality and deduplication checks
                fallback_pattern_json = {
                    "trigger_context_summary": trigger_summary,
                    "suggested_strategy": most_common_response
                }

                # Run our checks on this fallback pattern
                if is_quality_pattern(fallback_pattern_json):
                    # We need to run the deduplication check in an event loop
                    async def run_fallback_dedup_check():
                        if not await is_duplicate_pattern(most_common_response, upload.agent_id):
                            return True
                        return False
                    
                    is_unique = asyncio.run(run_fallback_dedup_check())

                    if is_unique:
                        print(f"Fallback successful. Creating a general pattern based on the most frequent successful response.")
                        fallback_pattern = LearnedPattern(
                            agent_id=upload.agent_id,
                            source="HISTORICAL_FALLBACK",
                            trigger_context_summary=trigger_summary,
                            suggested_strategy=most_common_response,
                            status="ACTIVE",
                            impressions=all_responses.count(most_common_response),
                            success_count=all_responses.count(most_common_response)
                        )
                        patterns_to_create.append(fallback_pattern)


        # --- FINAL STEP: Save all discovered patterns ---
        if patterns_to_create:
            print(f"Saving a total of {len(patterns_to_create)} new patterns to the database.")
            db.add_all(patterns_to_create)
            db.commit()
        else:
            print("No new patterns were generated in this run.")


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