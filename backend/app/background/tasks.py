from app.core.celery_app import celery_app
import uuid
import asyncio
import pandas as pd
from io import StringIO
from sqlalchemy import select, func as sqlfunc
# Database and Model imports
from app.database import AsyncSessionLocal
from app.models.historical_upload import HistoricalUpload
from app.models.historical_interaction import HistoricalInteraction
from app.models.learned_pattern import LearnedPattern
from app.models.interaction import Interaction
from app.models.outcome import Outcome
# Service imports
from app.services import embedding_service, llm_service
from sklearn.cluster import DBSCAN

# --- NEW TASK ---
@celery_app.task
def extract_patterns_from_history(upload_id: str):
    """
    Analyzes a completed historical upload to find and create patterns.
    """
    print(f"Starting pattern extraction for upload ID: {upload_id}")
    
    async def run_extraction():
        db = AsyncSessionLocal()
        try:
            # For this MVP, we'll implement a simplified logic:
            # 1. Fetch all successful historical interactions for this upload.
            # 2. Assume for now that context is a simple dict with a 'customer_type' key.
            # 3. Find the most common response for each customer_type.
            # 4. Create a new "Learned Pattern" for each one.
            
            # A real implementation would involve complex clustering on embeddings.
            
            # Get the agent_id from the upload
            upload = await db.get(HistoricalUpload, uuid.UUID(upload_id))
            if not upload:
                print(f"Upload {upload_id} not found.")
                return

            print(f"Found {len(upload.interactions)} interactions to analyze.")
            
            successful_interactions = [
                inter for inter in upload.interactions if inter.is_success
            ]
            
            # Simplified pattern logic
            patterns_found = {} # e.g., {"honeymooner": "Suggested strategy..."}
            for inter in successful_interactions:
                customer_type = inter.original_context.get("customer_type")
                if customer_type and customer_type not in patterns_found:
                    # This is a simplification. A real system would find the MODE
                    # or use an LLM to summarize common successful responses.
                    patterns_found[customer_type] = inter.original_response

            # Create LearnedPattern records
            for context_summary, strategy in patterns_found.items():
                print(f"Found new pattern for customer_type: {context_summary}")
                new_pattern = LearnedPattern(
                    agent_id=upload.agent_id,
                    source="HISTORICAL",
                    trigger_context_summary=context_summary,
                    suggested_strategy=strategy,
                    status="ACTIVE", # Promote historical patterns directly
                    impressions=1, # Start with some pseudo-counts
                    success_count=1
                )
                db.add(new_pattern)
            
            await db.commit()
            print(f"Pattern extraction complete for upload ID: {upload_id}")

        finally:
            await db.close()

    asyncio.run(run_extraction())
    return {"status": "success", "task": "extract_patterns_from_history", "upload_id": upload_id}


# --- MODIFIED EXISTING TASK ---
@celery_app.task
def process_historical_upload(upload_id: str):
    """
    Celery task to process a historical data file.
    NOW it will trigger the pattern extraction task upon completion.
    """
    print(f"Starting to process historical upload with ID: {upload_id}")
    
    # Placeholder for file parsing logic. In a real app, this would
    # read from S3, parse a CSV/JSON, and create HistoricalInteraction objects.
    # We will simulate this by creating a few dummy interactions.
    
    async def simulate_processing():
        db = AsyncSessionLocal()
        try:
            upload = await db.get(HistoricalUpload, uuid.UUID(upload_id))
            if not upload: return

            # SIMULATION: Create dummy historical interactions
            dummy_interactions = [
                HistoricalInteraction(upload_id=upload.id, original_context={"customer_type": "honeymoon"}, original_response="For a romantic trip, try our Couple's Paradise package.", is_success=True),
                HistoricalInteraction(upload_id=upload.id, original_context={"customer_type": "family"}, original_response="The Adventurous Family package is great for kids.", is_success=True),
                HistoricalInteraction(upload_id=upload.id, original_context={"customer_type": "business"}, original_response="Our standard package is fine.", is_success=False),
            ]
            db.add_all(dummy_interactions)

            upload.status = "COMPLETED"
            upload.total_interactions = len(dummy_interactions)
            upload.processed_interactions = len(dummy_interactions)
            await db.commit()
            print(f"Successfully processed and updated status for upload ID: {upload_id}")

            # --- NEW ORCHESTRATION STEP ---
            # Trigger the next step in the pipeline.
            extract_patterns_from_history.delay(upload_id)

        finally:
            await db.close()

    asyncio.run(simulate_processing())
    return {"status": "success", "upload_id": upload_id}


@celery_app.task
def discover_patterns_from_live_data(agent_id: str):
    """
    Analyzes recent live interactions for a given agent to find new patterns.
    This would be run periodically (e.g., nightly).
    """
    print(f"Starting live pattern discovery for agent ID: {agent_id}")
    
    async def run_discovery():
        db = AsyncSessionLocal()
        try:
            # This is a simplified version of the logic. A full implementation
            # would use more advanced clustering on context embeddings.
            
            # 1. Fetch recent successful interactions.
            stmt = (
                select(Interaction)
                .join(Outcome, Interaction.id == Outcome.interaction_id)
                .where(Interaction.agent_id == uuid.UUID(agent_id), Outcome.is_success == True)
                # In a real app, we'd filter by a recent time window.
                .limit(1000) 
            )
            result = await db.execute(stmt)
            successful_interactions = result.scalars().all()

            if len(successful_interactions) < 20: # Don't run on too little data
                print("Not enough recent successful interactions to discover new patterns.")
                return

            # 2. Simplified logic: Group interactions by a common context key (e.g., 'occasion')
            #    and see if any particular agent response for that context is unusually successful.
            context_groups = {}
            for inter in successful_interactions:
                if inter.context and 'occasion' in inter.context:
                    key = inter.context['occasion']
                    if key not in context_groups:
                        context_groups[key] = []
                    # We need the agent's response, which is in the transcript.
                    # This is a simplification; a real app would parse the transcript better.
                    if inter.full_transcript and "Agent:" in inter.full_transcript:
                        agent_response = inter.full_transcript.split("Agent:")[-1].strip()
                        context_groups[key].append(agent_response)
            
            # 3. For each group, find the most common successful response (the mode).
            for occasion, responses in context_groups.items():
                if not responses: continue
                
                most_common_response = max(set(responses), key=responses.count)
                
                # 4. Check if a pattern for this already exists. If not, create a new candidate.
                existing_pattern_stmt = select(LearnedPattern).where(
                    LearnedPattern.agent_id == uuid.UUID(agent_id),
                    LearnedPattern.trigger_context_summary == occasion
                )
                existing_pattern_result = await db.execute(existing_pattern_stmt)
                if existing_pattern_result.scalars().first():
                    continue # A pattern for this already exists, skip for now.

                print(f"Discovered new candidate pattern for occasion: {occasion}")
                new_pattern = LearnedPattern(
                    agent_id=uuid.UUID(agent_id),
                    source="LIVE_DISCOVERED",
                    trigger_context_summary=occasion,
                    suggested_strategy=most_common_response,
                    status="CANDIDATE" # New patterns must be tested
                )
                db.add(new_pattern)
            
            await db.commit()
            print(f"Live pattern discovery complete for agent ID: {agent_id}")

        finally:
            await db.close()

    asyncio.run(run_discovery())
    return {"status": "success", "task": "discover_patterns_from_live_data", "agent_id": agent_id}


@celery_app.task
def process_live_outcome(interaction_id: str):
    """
    Updates the performance statistics for the pattern used in a live interaction.
    This is the feedback loop for the bandit model.
    """
    print(f"Processing outcome for interaction ID: {interaction_id}")
    
    async def run_update():
        db = AsyncSessionLocal()
        try:
            # 1. Fetch the interaction and its outcome
            interaction = await db.get(Interaction, uuid.UUID(interaction_id))
            if not interaction or not interaction.outcome:
                print(f"Interaction or outcome not found for ID: {interaction_id}")
                return
            
            # 2. Check if a pattern was used
            if interaction.applied_pattern_id:
                pattern = await db.get(LearnedPattern, interaction.applied_pattern_id)
                if pattern:
                    # 3. Update the pattern's stats
                    pattern.impressions += 1
                    if interaction.outcome.is_success:
                        pattern.success_count += 1
                    
                    await db.commit()
                    print(f"Updated stats for pattern {pattern.id}: "
                          f"Successes={pattern.success_count}, Impressions={pattern.impressions}")
        finally:
            await db.close()

    asyncio.run(run_update())
    return {"status": "success", "task": "process_live_outcome", "interaction_id": interaction_id}