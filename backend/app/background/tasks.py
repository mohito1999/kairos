# backend/app/background/tasks.py
import uuid
import json
import asyncio
import pandas as pd
from io import StringIO
from datetime import datetime, timezone
import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import Dict, Any, Optional

from app.core.celery_app import celery_app
# Correctly import our definitive session manager and the async session for helpers
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

# --- NEW IMPORTS FOR HYBRID ENGINE ---
# Scientific Libraries
import numpy as np
import statsmodels.api as sm
from scipy.stats import ttest_ind
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


# --- NEW HELPER FUNCTIONS FOR HYBRID ENGINE (A1) ---

def find_best_grouping_key(interactions: list[HistoricalInteraction]) -> str | None:
    """
    Analyzes a list of interactions to find the best categorical key for grouping.
    Returns the key (e.g., 'customer_type') or None if no good key is found.
    This is the first part of the "Sweep" in Stage 1.
    """
    # A prioritized list of common categorical keys we should look for.
    POTENTIAL_GROUPING_KEYS = [
        "customer_type", "inquiry_type", "category", "department", 
        "priority", "region", "lead_source", "product_tier"
    ]
    
    best_key = None
    # Require at least 20% of data to be covered by meaningful groups to be considered a "good" key.
    max_coverage = 0.20  

    for key in POTENTIAL_GROUPING_KEYS:
        groups = {}
        for inter in interactions:
            # Ensure context is a dict before calling .get()
            context = inter.original_context if isinstance(inter.original_context, dict) else {}
            value = context.get(key)
            if value and isinstance(value, (str, int)): # Only group by simple, hashable types
                groups.setdefault(value, []).append(inter)
        
        # A "good" key is one that creates multiple, non-trivial groups.
        # Let's define a meaningful group as having at least 2 interactions.
        meaningful_groups = [g for g in groups.values() if len(g) >= 2]
        
        # We also want at least 2 distinct groups to have been formed.
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
        print("No suitable grouping key found. Will proceed to Stage 2 clustering.")
        
    return best_key


def get_clustering_params(data_size: int) -> dict:
    """
    Dynamically determines DBSCAN parameters for Stage 2 based on the dataset size.
    Stricter parameters are used for larger datasets to find only high-density clusters.
    """
    if data_size < 10:
        # For very small datasets, be lenient. A cluster can be just 2 points.
        return {"eps": 0.5, "min_samples": 2}
    elif data_size < 100:
        # For medium datasets, be slightly stricter.
        return {"eps": 0.45, "min_samples": 3}
    else: # For large datasets (100+)
        # Be much stricter to find only high-confidence clusters.
        return {"eps": 0.4, "min_samples": 5}


async def is_duplicate_pattern(new_pattern_strategy: str, agent_id: uuid.UUID) -> bool:
    try:
        async with AsyncSessionLocal() as db:
            stmt = select(LearnedPattern).where(LearnedPattern.agent_id == agent_id, LearnedPattern.status == "ACTIVE")
            result = await db.execute(stmt)
            existing_patterns = result.scalars().all()

            if not existing_patterns: return False

            new_embedding_list = await embedding_service.get_embeddings([new_pattern_strategy])
            if not new_embedding_list or not new_embedding_list[0]:
                print("Warning: Could not generate embedding for new pattern. Cannot check for duplicates.")
                return False

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
    """
    Performs a series of quality checks to ensure a generated pattern is useful.
    This is our LLM Quality Gate guardrail.
    """
    strategy = pattern_json.get("suggested_strategy", "").lower()
    trigger = pattern_json.get("trigger_context_summary", "").lower()

    # Check 1: Is the strategy too generic or vague?
    generic_phrases = [
        "be helpful", "assist the customer", "provide support", "be nice",
        "handle the request", "answer the question", "respond appropriately"
    ]
    if any(phrase in strategy for phrase in generic_phrases):
        print(f"Quality Check Failed: Strategy '{strategy}' is too generic.")
        return False

    # Check 2: Is the strategy too short to be meaningful? (Must be at least 4 words)
    if len(strategy.split()) < 4:
        print(f"Quality Check Failed: Strategy '{strategy}' is too short.")
        return False
        
    # Check 3: Is the trigger summary too short? (Must be at least 3 words)
    if len(trigger.split()) < 3:
        print(f"Quality Check Failed: Trigger '{trigger}' is too short.")
        return False

    # All checks passed
    return True


def smart_contextual_subclustering(context_embeddings: np.ndarray, min_k=2, max_k=5, min_cluster_size=3):
    """
    Tries multiple clustering approaches (K-Means with optimal k) and picks the best result
    based on silhouette score to force differentiation in contexts.
    """
    if len(context_embeddings) < (min_k * min_cluster_size):
        print(f"  Contextual Pass: Not enough data ({len(context_embeddings)} items) for meaningful sub-clustering. Skipping.")
        return None

    # We will test K-Means for a range of k values
    possible_k_values = range(min_k, min(max_k + 1, len(context_embeddings) // min_cluster_size))
    
    best_labels = None
    best_score = -1.0  # Silhouette score ranges from -1 to 1
    best_k = 0

    print(f"  Contextual Pass: Testing k={list(possible_k_values)} for K-Means.")
    for k in possible_k_values:
        if k <= 1: continue

        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        try:
            cluster_labels = kmeans.fit_predict(context_embeddings)
        except Exception as e:
            print(f"    K-Means failed for k={k}: {e}")
            continue

        # Guardrail: ensure no cluster is smaller than our minimum size
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        if np.min(counts) < min_cluster_size:
            print(f"    k={k} resulted in a cluster smaller than {min_cluster_size}. Discarding.")
            continue
            
        score = silhouette_score(context_embeddings, cluster_labels, metric='cosine')
        print(f"    k={k}, silhouette score: {score:.3f}")

        if score > best_score:
            best_score = score
            best_labels = cluster_labels
            best_k = k
    
    if best_labels is not None:
        print(f"  Contextual Pass: Selected optimal k={best_k} with silhouette score: {best_score:.3f}")
        return best_labels
    
    print("  Contextual Pass: No suitable sub-cluster division found.")
    return None

async def perform_causal_validation(pattern: LearnedPattern, holdout_ids: list[uuid.UUID], db: sa.orm.Session):
    """
    Performs a simplified causal analysis using Propensity Score Matching.
    Returns uplift and p-value.
    """
    # 1. Identify Treatment and Control groups in the holdout set
    pattern_trigger_embedding = await embedding_service.get_embedding(pattern.trigger_context_summary)
    if not pattern_trigger_embedding: return None

    holdout_interactions = db.execute(select(HistoricalInteraction).where(HistoricalInteraction.id.in_(holdout_ids))).scalars().all()
    
    # We need to find interactions where the pattern's TRIGGER was met.
    # To do this, we find contexts in the holdout set that are similar to the pattern's trigger.
    holdout_contexts = [json.dumps(i.original_context, sort_keys=True) for i in holdout_interactions]
    holdout_embeddings = await embedding_service.get_embeddings(holdout_contexts)

    similarity_threshold = 0.70 # Similarity threshold to be considered "matching" the trigger
    
    relevant_interactions = []
    for i, embedding in enumerate(holdout_embeddings):
        if not embedding: continue
        similarity = cosine_similarity([pattern_trigger_embedding], [embedding])[0][0]
        if similarity > similarity_threshold:
            relevant_interactions.append(holdout_interactions[i])

    if len(relevant_interactions) < 10: 
        print(f"  Validation failed: Only found {len(relevant_interactions)} interactions in holdout set matching the trigger.")
        return None

    # Now, within these relevant interactions, which ones also used a similar STRATEGY?
    pattern_strategy_embedding = await embedding_service.get_embedding(pattern.suggested_strategy)
    if not pattern_strategy_embedding: return None

    treatment_group, control_group = [], []
    relevant_responses = [i.original_response for i in relevant_interactions]
    relevant_response_embeddings = await embedding_service.get_embeddings(relevant_responses)

    for i, embedding in enumerate(relevant_response_embeddings):
        if not embedding: continue
        similarity = cosine_similarity([pattern_strategy_embedding], [embedding])[0][0]
        if similarity > similarity_threshold:
            treatment_group.append(relevant_interactions[i]) # Used the strategy
        else:
            control_group.append(relevant_interactions[i]) # Did NOT use the strategy

    print(f"  Found {len(treatment_group)} in Treatment Group, {len(control_group)} in Control Group.")

    if not treatment_group or not control_group:
        print("  Validation failed: Could not form both treatment and control groups.")
        return None
        
    # 2. Perform a T-test for a simple statistical comparison
    treatment_outcomes = [1 if i.is_success else 0 for i in treatment_group]
    control_outcomes = [1 if i.is_success else 0 for i in control_group]

    if np.mean(treatment_outcomes) <= np.mean(control_outcomes):
        return {"uplift": np.mean(treatment_outcomes) - np.mean(control_outcomes), "p_value": 1.0}
        
    t_stat, p_value = ttest_ind(treatment_outcomes, control_outcomes, equal_var=False)

    return {
        "uplift": np.mean(treatment_outcomes) - np.mean(control_outcomes),
        "p_value": p_value
    }


# --- CELERY TASKS ---

@celery_app.task(name='app.background.tasks.extract_patterns_from_history_task')
def extract_patterns_from_history_task(upload_id: str):
    """
    Analyzes historical data using the definitive, context-aware Hybrid Intelligence Engine.
    This task implements our final multi-stage discovery process to find specific,
    nuanced, and high-quality behavioral patterns.
    """
    print(f"Starting DEFINITIVE HYBRID pattern extraction for upload ID: {upload_id}")
    
    # Create a single event loop for the entire task
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        with get_sync_db_session() as db:
            upload = db.get(HistoricalUpload, uuid.UUID(upload_id))
            if not upload or not upload.interaction_id_split:
                print(f"Upload {upload_id} not found or has no interaction split.")
                return

            # --- KEY CHANGE: Only use the training set for discovery ---
            training_set_ids_str = upload.interaction_id_split.get("training_set", [])
            if not training_set_ids_str:
                print("No training set interactions found for this upload.")
                upload.status = "COMPLETED"
                db.commit()
                return
            
            training_set_ids = [uuid.UUID(id_str) for id_str in training_set_ids_str]

            stmt = select(HistoricalInteraction).where(
                HistoricalInteraction.id.in_(training_set_ids)
            )
            successful_interactions = db.execute(stmt).scalars().all()
            
            print(f"Found {len(successful_interactions)} successful interactions in the TRAINING SET to analyze.")
            if len(successful_interactions) < 5:
                print("Not enough successful interactions to find meaningful patterns.")
                upload.status = "COMPLETED"
                db.commit()
                return

            patterns_to_create = []
            processed_interaction_ids = set()

            # --- STAGE 1: The "Sweep" (Dynamic Grouping & Behavioral Consensus Check) ---
            
            best_grouping_key = find_best_grouping_key(successful_interactions)
            
            if best_grouping_key:
                print(f"Stage 1: Found best grouping key: '{best_grouping_key}'. Analyzing group consensus...")
                
                # Create the initial groups based on the identified key
                context_groups = {}
                for inter in successful_interactions:
                    context = inter.original_context if isinstance(inter.original_context, dict) else {}
                    value = context.get(best_grouping_key)
                    if value:
                        context_groups.setdefault(value, []).append(inter)

                # Use loop.run_until_complete instead of asyncio.run()
                async def perform_consensus_check_and_analysis():
                    try:
                        group_patterns = []
                        all_responses = [inter.original_response for inter in successful_interactions if inter.original_response]
                        if not all_responses: return []
                        
                        all_embeddings = await embedding_service.get_embeddings(all_responses)
                        response_embedding_map = {resp: emb for resp, emb in zip(all_responses, all_embeddings) if emb}

                        for context_key, interactions_in_group in context_groups.items():
                            if len(interactions_in_group) < 3: continue
                            
                            valid_group_embeddings = [emb for emb in [response_embedding_map.get(i.original_response) for i in interactions_in_group] if emb]
                            if len(valid_group_embeddings) < 2: continue
                            
                            similarity_matrix = cosine_similarity(valid_group_embeddings)
                            avg_similarity = np.mean(similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)])
                            
                            print(f"Group '{context_key}': Avg. internal similarity = {avg_similarity:.2f}")

                            if 0.75 < avg_similarity <= 0.95:
                                print(f"STRONG CONSENSUS found for group '{context_key}' (similarity: {avg_similarity:.2f}). Extracting pattern.")
                                sample_responses = [inter.original_response for inter in interactions_in_group[:5]]
                                
                                system_prompt = """
                                You are a data analyst. You will be given a list of successful agent responses that were all used in very similar situations. 
                                Your task is to identify the core strategy being used and to summarize the situation (the trigger).
                                Respond in a valid JSON format with two keys: "trigger_context_summary" and "suggested_strategy".
                                The trigger should be a concise description of the situation, e.g., "When a customer asks for a discount".
                                The strategy should be an actionable instruction for an AI agent, e.g., "Offer a 10% discount but mention it's a limited-time offer".
                                """
                                user_prompt = f"Context Key: '{best_grouping_key}' is '{context_key}'.\nSuccessful agent responses:\n---\n{json.dumps(sample_responses, indent=2)}\n---"
                                
                                pattern_json = await llm_service.get_json_response(system_prompt=system_prompt, user_prompt=user_prompt, model="openai/gpt-4o-mini")

                                if pattern_json and "suggested_strategy" in pattern_json and is_quality_pattern(pattern_json):
                                    if not await is_duplicate_pattern(pattern_json["suggested_strategy"], upload.agent_id):
                                        group_patterns.append(LearnedPattern(
                                            agent_id=upload.agent_id, 
                                            source="HISTORICAL_GROUPED",
                                            source_upload_id=upload.id,
                                            trigger_context_summary=pattern_json["trigger_context_summary"],
                                            suggested_strategy=pattern_json["suggested_strategy"], 
                                            status="CANDIDATE",
                                            impressions=len(interactions_in_group), 
                                            success_count=len(interactions_in_group)
                                        ))
                                        for inter in interactions_in_group:
                                            processed_interaction_ids.add(inter.id)
                            else:
                                print(f"Group '{context_key}': Similarity ({avg_similarity:.2f}) is outside optimal range (0.75, 0.95]. Passing group to later stages.")
                        
                        return group_patterns
                    
                    except Exception as e:
                        print(f"FATAL ERROR during consensus check: {e}")
                        import traceback
                        traceback.print_exc()
                        return []

                # Run the async analysis for Stage 1 using the persistent loop
                group_patterns = loop.run_until_complete(perform_consensus_check_and_analysis())
                patterns_to_create.extend(group_patterns)
                print(f"Stage 1 generated {len(group_patterns)} high-confidence patterns from grouped data.")
            else:
                print("Stage 1: No suitable grouping key found. Will proceed to Stage 2 clustering.")

            # --- STAGE 2: Context-Aware Deep Search with Smart Sub-Clustering ---
            ungrouped_interactions = [inter for inter in successful_interactions if inter.id not in processed_interaction_ids]
            print(f"Stage 1 complete. {len(ungrouped_interactions)} interactions remain for deep search.")

            if len(ungrouped_interactions) >= 5:
                print(f"Stage 2: Starting Context-Aware Deep Search on {len(ungrouped_interactions)} interactions.")
                
                async def perform_context_aware_deep_search():
                    try:
                        # --- Pass 1: Cluster by BEHAVIOR ---
                        responses_to_embed = [inter.original_response for inter in ungrouped_interactions if inter.original_response]
                        if not responses_to_embed: return []

                        behavior_embeddings = await embedding_service.get_embeddings(responses_to_embed)
                        valid_behavior_embeddings = [emb for emb in behavior_embeddings if emb]
                        valid_behavior_interactions = [inter for i, inter in enumerate(ungrouped_interactions) if behavior_embeddings and i < len(behavior_embeddings) and behavior_embeddings[i]]
                        
                        if len(valid_behavior_embeddings) < 3:
                            print("Stage 2: Not enough valid response embeddings for behavioral clustering.")
                            return []
                        
                        behavior_params = get_clustering_params(len(valid_behavior_embeddings))
                        print(f"  Behavioral Pass: Clustering {len(valid_behavior_embeddings)} items with params {behavior_params}")
                        behavior_clustering = DBSCAN(eps=behavior_params["eps"], min_samples=behavior_params["min_samples"], metric="cosine").fit(np.array(valid_behavior_embeddings))
                        
                        behavior_clusters = {label: [] for label in set(behavior_clustering.labels_) if label != -1}
                        for i, label in enumerate(behavior_clustering.labels_):
                            if label != -1: 
                                behavior_clusters[label].append(valid_behavior_interactions[i])

                        print(f"  Behavioral Pass: Found {len(behavior_clusters)} broad behavioral families.")
                        
                        discovered_patterns = []

                        # --- Pass 2: Sub-Cluster by CONTEXT using Smart Contextual Sub-Clustering ---
                        for label, interactions_in_cluster in behavior_clusters.items():
                            print(f"\n--- Analyzing Behavioral Family #{label} with {len(interactions_in_cluster)} interactions ---")
                            
                            # Skip if cluster is too small for meaningful sub-clustering
                            if len(interactions_in_cluster) < 4:  # Min needed for K-Means
                                print(f"Behavioral Family #{label} is too small for sub-clustering ({len(interactions_in_cluster)} < 4). Skipping.")
                                continue
                            
                            contexts_to_embed = [json.dumps(inter.original_context, sort_keys=True) for inter in interactions_in_cluster if inter.original_context]
                            if len(contexts_to_embed) < 4:
                                print(f"Behavioral Family #{label}: Not enough valid contexts for sub-clustering.")
                                continue

                            context_embeddings = await embedding_service.get_embeddings(contexts_to_embed)
                            valid_context_embeddings = [emb for emb in context_embeddings if emb]
                            valid_context_interactions = [inter for i, inter in enumerate(interactions_in_cluster) if context_embeddings and i < len(context_embeddings) and context_embeddings[i]]

                            if len(valid_context_embeddings) < 4:
                                print(f"Behavioral Family #{label}: Not enough valid context embeddings for sub-clustering.")
                                continue
                            
                            # --- THE NEW CORE LOGIC: Smart Contextual Sub-Clustering ---
                            print(f"  Contextual Pass: Using smart sub-clustering on {len(valid_context_embeddings)} contexts")
                            context_labels = smart_contextual_subclustering(np.array(valid_context_embeddings))

                            if context_labels is not None:
                                sub_clusters = {sub_label: [] for sub_label in set(context_labels)}
                                for i, sub_label in enumerate(context_labels):
                                    sub_clusters[sub_label].append(valid_context_interactions[i])
                                
                                print(f"  Contextual Pass: Differentiated into {len(sub_clusters)} specific sub-patterns.")

                                for sub_label, interactions_in_sub_cluster in sub_clusters.items():
                                    # --- GUARDRAIL: Minimum Viable Cluster Size ---
                                    MINIMUM_CLUSTER_SIZE = 3
                                    if len(interactions_in_sub_cluster) < MINIMUM_CLUSTER_SIZE:
                                        print(f"    Sub-cluster #{sub_label} is too small ({len(interactions_in_sub_cluster)} interactions). Discarding.")
                                        continue

                                    print(f"    --- Synthesizing Pattern from Sub-Cluster #{sub_label} ({len(interactions_in_sub_cluster)} interactions) ---")
                                    sample_contexts = [inter.original_context for inter in interactions_in_sub_cluster[:10]]
                                    sample_response = interactions_in_sub_cluster[0].original_response
                                    
                                    # Using the refined, more demanding prompt
                                    system_prompt = """
                                    You are an expert data strategist. Your task is to find a specific, tactical pattern from the provided data.
                                    A group of conversations has been clustered based on a similar successful AGENT BEHAVIOR. 
                                    Analyze the diverse contexts to find the underlying, abstract trigger that justifies this agent behavior.
                                    IMPORTANT: Do NOT provide generic advice like "be helpful" or "listen to the customer". Focus on concrete, repeatable actions.
                                    Respond in a valid JSON format with "trigger_context_summary" and "suggested_strategy".
                                    """
                                    user_prompt = f"""
                                    The successful agent strategy was a variant of this: "{sample_response}"

                                    This strategy worked in these diverse situations (contexts):
                                    ---
                                    {json.dumps(sample_contexts, indent=2)}
                                    ---
                                    Based on the contexts, what is the specific, non-obvious trigger for this strategy?
                                    What is a concise, tactical instruction for the agent's strategy?
                                    """
                                    
                                    pattern_json = await llm_service.get_json_response(system_prompt=system_prompt, user_prompt=user_prompt, model="openai/gpt-4o-mini")

                                    if not pattern_json or "suggested_strategy" not in pattern_json:
                                        print(f"    Sub-cluster #{sub_label}: LLM failed to return valid JSON. Skipping.")
                                        continue

                                    print(f"    Sub-cluster #{sub_label}: LLM generated strategy: '{pattern_json['suggested_strategy']}'")
                                    
                                    # Quality gate check
                                    if not is_quality_pattern(pattern_json):
                                        print(f"    Sub-cluster #{sub_label}: Pattern failed quality gate. Skipping.")
                                        continue

                                    # Duplicate check
                                    if await is_duplicate_pattern(pattern_json["suggested_strategy"], upload.agent_id):
                                        print(f"    Sub-cluster #{sub_label}: Pattern failed de-duplication gate. Skipping.")
                                        continue
                                    
                                    print(f"    Sub-cluster #{sub_label}: PASSED ALL GATES. Creating new pattern.")
                                    discovered_patterns.append(LearnedPattern(
                                        agent_id=upload.agent_id, 
                                        source="HISTORICAL_DISCOVERED",
                                        source_upload_id=upload.id,
                                        trigger_context_summary=pattern_json["trigger_context_summary"],
                                        suggested_strategy=pattern_json["suggested_strategy"], 
                                        status="CANDIDATE",
                                        impressions=len(interactions_in_sub_cluster), 
                                        success_count=len(interactions_in_sub_cluster)
                                    ))
                            else:
                                print(f"  Contextual Pass: Could not find a meaningful way to differentiate contexts for Behavioral Family #{label}. No patterns generated for this behavioral family.")
                        
                        return discovered_patterns
                    
                    except Exception as e:
                        print(f"FATAL ERROR during context-aware deep search: {e}")
                        import traceback
                        traceback.print_exc()
                        return []

                # Run the async analysis for Stage 2 using the persistent loop
                clustered_patterns = loop.run_until_complete(perform_context_aware_deep_search())
                patterns_to_create.extend(clustered_patterns)
                print(f"Stage 2 generated {len(clustered_patterns)} emergent patterns from context-aware clustering.")
            else:
                print("Not enough remaining interactions for deep search clustering.")

            # --- STAGE 3: FALLBACK STRATEGY ---
            if not patterns_to_create and len(successful_interactions) >= 5:
                print("Stage 3: No specific patterns found in main pipeline. Running Final Pass as a fallback.")
                async def perform_final_pass_analysis():
                    final_patterns = []
                    all_responses = [inter.original_response for inter in successful_interactions if inter.original_response]
                    if all_responses:
                        most_common_response = max(set(all_responses), key=all_responses.count)
                        if not await is_duplicate_pattern(most_common_response, upload.agent_id):
                            print("Final Pass created a fallback pattern from the most frequent successful response.")
                            final_patterns.append(LearnedPattern(
                                agent_id=upload.agent_id,
                                source="HISTORICAL_FALLBACK",
                                trigger_context_summary="A frequently used, generally successful response from historical data.",
                                suggested_strategy=most_common_response,
                                status="CANDIDATE",
                                impressions=all_responses.count(most_common_response),
                                success_count=all_responses.count(most_common_response)
                            ))
                    return final_patterns
                
                final_patterns = loop.run_until_complete(perform_final_pass_analysis())
                patterns_to_create.extend(final_patterns)

            # --- FINAL CHANGE: Chain the new validation task ---
            if patterns_to_create:
                print(f"Saving a total of {len(patterns_to_create)} new CANDIDATE patterns.")
                db.add_all(patterns_to_create)
            else:
                print("No new patterns were generated in this run.")

            # We don't mark as COMPLETED yet. We queue the validation task.
            upload.status = "VALIDATING"
            db.commit()
            
            print(f"Pattern extraction complete. Triggering validation task.")
            celery_app.send_task(
                'app.background.tasks.validate_patterns_task',
                args=[upload_id]
            )

    except Exception as e:
        print(f"FATAL ERROR in pattern extraction task: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always clean up the event loop properly
        try:
            # Cancel any remaining tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            
            # Wait for all cancelled tasks to complete
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
            # Close the loop
            loop.close()
        except Exception as cleanup_error:
            print(f"Error during loop cleanup: {cleanup_error}")
        finally:
            # Reset the event loop policy to avoid issues with subsequent tasks
            asyncio.set_event_loop(None)

@celery_app.task(name='app.background.tasks.process_historical_upload_task')
def process_historical_upload_task(upload_id: str, file_content_bytes: bytes, data_mapping: dict):
    """
    Processes a historical data file with flexible context mapping,
    optional LLM-as-a-judge for outcome determination, and robust parsing.
    This is the first task in the historical upload pipeline.
    """
    
    # Create a persistent event loop for this task too
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
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
                    system_prompt=system_prompt, user_prompt=user_prompt, model="openai/gpt-4o-mini"
                )
                return assessment.get("is_success", False)
            except Exception as e:
                print(f"Error in AI outcome judgment: {e}")
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

                # --- PRE-PROCESS THE DATA MAPPING ---
                outcome_col = data_mapping.get("outcome_column")
                outcome_goal = data_mapping.get("outcome_goal_description")
                transcript_col = data_mapping.get("conversation_transcript")
                context_cols = {key: value for key, value in data_mapping.items() if key.startswith("context_")}

                # --- VALIDATE MAPPED COLUMNS EXIST IN THE DATAFRAME ---
                if outcome_col and outcome_col not in df.columns:
                    print(f"Warning: Outcome column '{outcome_col}' not found in CSV. Outcomes will be marked as False.")
                    outcome_col = None
                if transcript_col and transcript_col not in df.columns:
                    print(f"Warning: Transcript column '{transcript_col}' not found in CSV.")
                    transcript_col = None

                # --- NEW LOGIC: Split successful interactions into training and holdout sets ---
                all_interactions = []
                successful_interaction_ids = []

                # --- ITERATE AND PROCESS EACH ROW ---
                for index, row in df.iterrows():
                    # 1. Dynamically build the context object
                    context = {
                        key.replace("context_", ""): row.get(value)
                        for key, value in context_cols.items() if value in df.columns
                    }
                    
                    # 2. Get transcript/response text
                    response_text = str(row.get(transcript_col, "")) if transcript_col else ""
                    
                    # 3. Determine the outcome using the "either/or" logic
                    is_success = False
                    raw_outcome = ""
                    if outcome_col:
                        # Method A: Use the outcome column
                        raw_outcome_value = str(row.get(outcome_col, ""))
                        cleaned_outcome = raw_outcome_value.strip().strip('"').strip("'").lower()
                        is_success = cleaned_outcome in ['true', 'success', '1', 'yes', 'resolved']
                        raw_outcome = raw_outcome_value
                    elif outcome_goal and response_text:
                        # Method B: Use LLM-as-a-judge with persistent loop
                        print(f"Row {index}: Using AI Judge for outcome...")
                        is_success = loop.run_until_complete(_judge_outcome(response_text, outcome_goal))
                        raw_outcome = "judged_by_ai"
                    
                    interaction_obj = HistoricalInteraction(
                        upload_id=upload.id,
                        original_context=context,
                        original_response=response_text,
                        is_success=is_success,
                        extracted_outcome={"value_from_file": raw_outcome}
                    )
                    all_interactions.append(interaction_obj)
                    if is_success:
                        # We need the ID before it's committed, so we assign one
                        interaction_obj.id = uuid.uuid4()
                        successful_interaction_ids.append(interaction_obj.id)
                
                # Bulk insert all interactions for efficiency
                db.add_all(all_interactions)

                # Now, perform the 70/30 split
                training_ids, holdout_ids = [], []
                if len(successful_interaction_ids) >= 5: # Only split if there's enough data
                    training_ids, holdout_ids = train_test_split(
                        successful_interaction_ids,
                        test_size=0.3,
                        random_state=42
                    )
                else: # Otherwise, everything is for training
                    training_ids = successful_interaction_ids
                
                print(f"Split successful interactions: {len(training_ids)} for training, {len(holdout_ids)} for holdout validation.")

                # Save the split to the database
                upload.interaction_id_split = {
                    "training_set": [str(uid) for uid in training_ids],
                    "holdout_set": [str(uid) for uid in holdout_ids]
                }

                upload.status = "PARSED"
                upload.total_interactions = len(df)
                upload.processed_interactions = len(df)
                db.commit()

                print(f"Successfully parsed and saved {len(df)} interactions. Triggering pattern extraction.")
                # Use send_task for decoupling
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
        import traceback
        traceback.print_exc()
    finally:
        # Clean up the event loop
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
        except Exception as cleanup_error:
            print(f"Error during loop cleanup: {cleanup_error}")
        finally:
            asyncio.set_event_loop(None)


@celery_app.task(name='app.background.tasks.discover_patterns_from_live_data_task')
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


@celery_app.task(name='app.background.tasks.process_live_outcome_task')
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


@celery_app.task(name='app.background.tasks.generate_opportunities_task')
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


@celery_app.task(name='app.background.tasks.process_human_interaction_task')
def process_human_interaction_task(
    agent_id: str,
    recording_url: str,
    context: Optional[Dict[str, Any]],
    explicit_outcome: Optional[Dict[str, Any]],
    outcome_goal: Optional[str]
):
    """
    Processes a recorded human interaction: transcribes, determines outcome, and saves it.
    Uses a persistent event loop for async operations.
    """
    
    # Create a persistent event loop for this task
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
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
                Only return a valid JSON object.
                """
                user_prompt = f"SUCCESS GOAL: \"{outcome_goal}\"\n\nCONVERSATION TRANSCRIPT:\n---\n{transcript}\n---"
                
                print("Performing AI-assisted outcome assessment...")
                try:
                    assessment = await llm_service.get_json_response(
                        system_prompt=system_prompt, user_prompt=user_prompt, model="openai/gpt-4o-mini"
                    )
                    is_success = assessment.get("is_success", False)
                except Exception as e:
                    print(f"Error in AI outcome assessment: {e}")
                    is_success = False
            
            return transcript, is_success

        # Run the async part to get the data we need using persistent loop
        transcript, is_success = loop.run_until_complete(get_transcript_and_assessment())

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

    except Exception as e:
        print(f"FATAL ERROR in human interaction processing task: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up the event loop
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
        except Exception as cleanup_error:
            print(f"Error during loop cleanup: {cleanup_error}")
        finally:
            asyncio.set_event_loop(None)

@celery_app.task(name='app.background.tasks.validate_patterns_task')
def validate_patterns_task(upload_id: str):
    """
    Validates CANDIDATE patterns against the holdout set using causal inference
    to determine if they are statistically significant.
    """
    print(f"Starting validation for patterns from upload ID: {upload_id}")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        with get_sync_db_session() as db:
            upload = db.get(HistoricalUpload, uuid.UUID(upload_id))
            if not upload or not upload.interaction_id_split: return

            holdout_set_ids_str = upload.interaction_id_split.get("holdout_set", [])
            if not holdout_set_ids_str:
                print("No holdout set. Marking upload as complete."); upload.status = "COMPLETED"; db.commit(); return

            holdout_set_ids = [uuid.UUID(id_str) for id_str in holdout_set_ids_str]
            
            # For now, we associate patterns with uploads by adding the upload_id to the source.
            # A future improvement would be a dedicated column.
            candidate_patterns = db.execute(select(LearnedPattern).where(
                LearnedPattern.source_upload_id == uuid.UUID(upload_id),
                LearnedPattern.status == 'CANDIDATE'
            )).scalars().all()

            if not candidate_patterns:
                print("No candidate patterns found to validate."); upload.status = "COMPLETED"; db.commit(); return
            
            print(f"Found {len(candidate_patterns)} candidate patterns to validate against {len(holdout_set_ids)} holdout interactions.")
            
            # --- The Causal Validation Loop ---
            for pattern in candidate_patterns:
                print(f"\n--- Validating Pattern {pattern.id} ---")
                
                # This is a complex async operation, so we run it in our loop.
                validation_results = loop.run_until_complete(
                    perform_causal_validation(pattern, holdout_set_ids, db)
                )

                if validation_results:
                    pattern.uplift_score = validation_results["uplift"]
                    pattern.p_value = validation_results["p_value"]
                    
                    # Check for statistical significance (p-value < 0.05)
                    if validation_results["p_value"] < 0.05 and validation_results["uplift"] > 0:
                        pattern.status = "VALIDATED"
                        print(f"  RESULT: Pattern VALIDATED. Uplift: {pattern.uplift_score:+.2%}, p-value: {pattern.p_value:.4f}")
                    else:
                        pattern.status = "REJECTED"
                        print(f"  RESULT: Pattern REJECTED. Insufficient statistical evidence.")
                else:
                    pattern.status = "REJECTED"
                    print("  RESULT: Pattern REJECTED. Could not perform validation (not enough data).")

            upload.status = "COMPLETED"
            db.commit()

    except Exception as e:
        print(f"FATAL ERROR during validation task: {e}")
        traceback.print_exc()
        # Rollback any partial changes
        with get_sync_db_session() as db:
            upload = db.get(HistoricalUpload, uuid.UUID(upload_id))
            if upload:
                upload.status = "VALIDATION_FAILED"
                db.commit()
    finally:
        # Event loop cleanup
        try:
            pending = asyncio.all_tasks(loop); [task.cancel() for task in pending]
            if pending: loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
        except Exception as cleanup_error:
            print(f"Error during loop cleanup: {cleanup_error}")
        finally:
            asyncio.set_event_loop(None)

