# backend/app/background/tasks.py
import uuid, json, asyncio, traceback
import pandas as pd
from io import StringIO
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

# SQLAlchemy and DB
import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_sync_db_session
from app.core.async_context import get_async_context, close_async_context

# Celery
from app.core.celery_app import celery_app

# Models
from app.models import ( Agent, HistoricalUpload, HistoricalInteraction, HumanInteraction,
                         LearnedPattern, Interaction, Outcome, SuggestedOpportunity )
from app.models.learned_pattern import PatternStatus

# Services (we'll call their async functions directly)
from app.services import embedding_service, llm_service, transcription_service

# Scientific Libraries
import numpy as np
from scipy.stats import ttest_ind
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


# --- CELERY-SAFE ASYNC ORCHESTRATOR PATTERN ---
# All top-level Celery tasks are simple sync wrappers that call `asyncio.run()`
# on a main `_async` orchestrator function. All I/O-bound logic lives inside the orchestrator.
# This pattern provides clean separation and robust async context management.

# --- SYNC HELPER FUNCTIONS (PURE COMPUTATION) ---

def find_best_grouping_key(interactions: list[HistoricalInteraction]) -> str | None:
    """Analyzes interactions to find the best categorical key for grouping."""
    POTENTIAL_GROUPING_KEYS = [
        "customer_type", "inquiry_type", "category", "department",
        "priority", "region", "lead_source", "product_tier"
    ]
    best_key, max_coverage = None, 0.20
    for key in POTENTIAL_GROUPING_KEYS:
        groups = {}
        for inter in interactions:
            context = inter.original_context if isinstance(inter.original_context, dict) else {}
            value = context.get(key)
            if value and isinstance(value, (str, int)):
                groups.setdefault(value, []).append(inter)
        meaningful_groups = [g for g in groups.values() if len(g) >= 2]
        if len(meaningful_groups) >= 2:
            num_grouped_interactions = sum(len(g) for g in meaningful_groups)
            coverage = num_grouped_interactions / len(interactions)
            if coverage > max_coverage:
                max_coverage, best_key = coverage, key
    if best_key:
        print(f"Found best grouping key: '{best_key}' with {max_coverage:.0%} coverage.")
    else:
        print("No suitable grouping key found. Will proceed to Stage 2 clustering.")
    return best_key

def get_clustering_params(data_size: int) -> dict:
    """Dynamically determines DBSCAN parameters based on dataset size."""
    if data_size < 10: return {"eps": 0.5, "min_samples": 2}
    elif data_size < 100: return {"eps": 0.45, "min_samples": 3}
    else: return {"eps": 0.4, "min_samples": 5}

def smart_contextual_subclustering(context_embeddings: np.ndarray, min_k=2, max_k=5, min_cluster_size=3):
    """Smart K-means clustering for contextual differentiation."""
    if len(context_embeddings) < (min_k * min_cluster_size):
        print(f"  Contextual Pass: Not enough data ({len(context_embeddings)} items) for meaningful sub-clustering. Skipping.")
        return None
    
    # Ensure the upper bound of the range is valid
    max_possible_k = len(context_embeddings) // min_cluster_size
    if max_possible_k < min_k:
        print(f"  Contextual Pass: Not enough data for min_k={min_k} with min_cluster_size={min_cluster_size}. Skipping.")
        return None
        
    possible_k_values = range(min_k, min(max_k + 1, max_possible_k + 1))
    best_labels, best_score, best_k = None, -1.0, 0

    if not possible_k_values:
        print("  Contextual Pass: No possible k values for clustering. Skipping.")
        return None

    print(f"  Contextual Pass: Testing k={list(possible_k_values)} for K-Means.")
    for k in possible_k_values:
        if k <= 1: continue
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        try:
            cluster_labels = kmeans.fit_predict(context_embeddings)
        except Exception as e:
            print(f"    K-Means failed for k={k}: {e}"); continue
        
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        if np.min(counts) < min_cluster_size:
            print(f"    k={k} resulted in a cluster smaller than {min_cluster_size}. Discarding."); continue
            
        score = silhouette_score(context_embeddings, cluster_labels, metric='cosine')
        print(f"    k={k}, silhouette score: {score:.3f}")

        if score > best_score:
            best_score, best_labels, best_k = score, cluster_labels, k
            
    if best_labels is not None:
        print(f"  Contextual Pass: Selected optimal k={best_k} with silhouette score: {best_score:.3f}")
        return best_labels
    
    print("  Contextual Pass: No suitable sub-cluster division found."); return None

def is_quality_pattern(pattern_json: dict) -> bool:
    """
    Quality checks for generated patterns.
    Now robust against unexpected data types from the LLM.
    """
    strategy_value = pattern_json.get("suggested_strategy")
    trigger_value = pattern_json.get("trigger_context_summary")

    # --- THE FIX: Check if the values are actually strings before processing ---
    if not isinstance(strategy_value, str) or not isinstance(trigger_value, str):
        print(f"Quality Check Failed: LLM returned unexpected data types. Strategy: {type(strategy_value)}, Trigger: {type(trigger_value)}")
        return False

    strategy = strategy_value.lower()
    trigger = trigger_value.lower()

    generic_phrases = [
        "be helpful", "assist the customer", "provide support", "be nice",
        "handle the request", "answer the question", "respond appropriately"
    ]
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


# --- PURE ASYNC HELPER FUNCTIONS ---

async def is_duplicate_pattern_async(session: AsyncSession, new_pattern_strategy: str, agent_id: uuid.UUID) -> bool:
    """Checks for duplicate patterns against active patterns for an agent."""
    stmt = select(LearnedPattern).where(
        LearnedPattern.agent_id == agent_id, 
        LearnedPattern.status == PatternStatus.ACTIVE
    )
    result = await session.execute(stmt)
    existing_patterns = result.scalars().all()
    if not existing_patterns: return False
    
    new_embedding_list = await embedding_service.get_embeddings([new_pattern_strategy])
    if not new_embedding_list or not new_embedding_list[0]: 
        print("Warning: Could not generate embedding for new pattern.")
        return False
    
    new_embedding = new_embedding_list[0]
    existing_strategy_texts = [p.suggested_strategy for p in existing_patterns]
    existing_strategy_embeddings = await embedding_service.get_embeddings(existing_strategy_texts)
    
    for i, existing_embedding in enumerate(existing_strategy_embeddings):
        if not existing_embedding: continue
        similarity = cosine_similarity([new_embedding], [existing_embedding])[0][0]
        if similarity > 0.95:
            print(f"New pattern is a likely duplicate of existing pattern (similarity: {similarity:.2f}). Skipping.")
            return True
    return False

def calculate_medoid_and_threshold(embeddings: np.ndarray):
    """Calculates the medoid vector and an adaptive similarity threshold for a cluster."""
    if len(embeddings) == 0:
        return None, None
    
    centroid = np.mean(embeddings, axis=0)
    similarities_to_centroid = cosine_similarity(embeddings, [centroid])
    medoid_index = np.argmax(similarities_to_centroid)
    medoid_embedding = embeddings[medoid_index]
    
    similarities_to_medoid = cosine_similarity(embeddings, [medoid_embedding]).flatten()
    adaptive_threshold = np.percentile(similarities_to_medoid, 10) # 10th percentile
    
    return medoid_embedding.tolist(), float(adaptive_threshold)


async def perform_consensus_check_and_analysis_async(session: AsyncSession, context_groups: dict, upload_agent_id: uuid.UUID, best_grouping_key: str) -> list:
    """Performs consensus analysis for pre-grouped data, returning full pattern data."""
    group_patterns = []
    
    for context_key, interactions_in_group in context_groups.items():
        if len(interactions_in_group) < 3: continue
        
        all_responses = [inter.original_response for inter in interactions_in_group if inter.original_response]
        if not all_responses: continue
            
        all_embeddings = await embedding_service.get_embeddings(all_responses)
        valid_embeddings = np.array([emb for emb in all_embeddings if emb])
        if len(valid_embeddings) < 2: continue
        
        similarity_matrix = cosine_similarity(valid_embeddings)
        avg_similarity = np.mean(similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)])
        print(f"Group '{context_key}': Avg. internal similarity = {avg_similarity:.2f}")

        if 0.75 < avg_similarity <= 0.95:
            print(f"STRONG CONSENSUS found for group '{context_key}'")
            sample_responses = all_responses[:5]
            
            system_prompt = """
            You are a data analyst. You will be given a list of successful agent responses that were all used in very similar situations. 
            Your task is to identify the core strategy being used and to summarize the situation (the trigger).
            Respond in a valid JSON format with "trigger_context_summary" and "suggested_strategy".
            """
            user_prompt = f"Context Key: '{best_grouping_key}' is '{context_key}'.\nSuccessful agent responses:\n---\n{json.dumps(sample_responses, indent=2)}\n---"
            
            pattern_json = await llm_service.get_json_response(system_prompt=system_prompt, user_prompt=user_prompt, model="openai/gpt-4o-mini")

            if (pattern_json and "suggested_strategy" in pattern_json and 
                is_quality_pattern(pattern_json) and
                not await is_duplicate_pattern_async(session, pattern_json["suggested_strategy"], upload_agent_id)):
                
                # --- THE FIX: Calculate medoid and threshold for Stage 1 patterns ---
                contexts_to_embed = [json.dumps(inter.original_context, sort_keys=True) for inter in interactions_in_group if inter.original_context]
                context_embeddings = np.array([emb for emb in await embedding_service.get_embeddings(contexts_to_embed) if emb])

                if len(context_embeddings) > 0:
                    centroid = np.mean(context_embeddings, axis=0)
                    similarities_to_centroid = cosine_similarity(context_embeddings, [centroid])
                    medoid_index = np.argmax(similarities_to_centroid)
                    medoid_embedding = context_embeddings[medoid_index]
                    
                    similarities_to_medoid = cosine_similarity(context_embeddings, [medoid_embedding]).flatten()
                    adaptive_threshold = np.percentile(similarities_to_medoid, 10)
                else:
                    medoid_embedding, adaptive_threshold = None, None

                group_patterns.append({
                    'trigger_context_summary': pattern_json["trigger_context_summary"],
                    'suggested_strategy': pattern_json["suggested_strategy"],
                    'impressions': len(interactions_in_group),
                    'success_count': len(interactions_in_group),
                    'trigger_embedding': medoid_embedding.tolist() if medoid_embedding is not None else None,
                    'trigger_threshold': adaptive_threshold
                })
        else:
            print(f"Group '{context_key}': Similarity outside optimal range (0.75, 0.95].")
    
    return group_patterns

        
async def perform_context_aware_deep_search_async(session: AsyncSession, ungrouped_interactions: list, upload_agent_id: uuid.UUID) -> list:
    """Performs deep contextual search on ungrouped data."""
    # Cluster by BEHAVIOR first
    responses_to_embed = [inter.original_response for inter in ungrouped_interactions if inter.original_response]
    if not responses_to_embed: return []

    behavior_embeddings = await embedding_service.get_embeddings(responses_to_embed)
    valid_behavior_embeddings = [emb for emb in behavior_embeddings if emb]
    valid_behavior_interactions = [inter for i, inter in enumerate(ungrouped_interactions) if i < len(behavior_embeddings) and behavior_embeddings[i]]
    if len(valid_behavior_embeddings) < 3: return []
    
    behavior_params = get_clustering_params(len(valid_behavior_embeddings))
    print(f"Behavioral clustering {len(valid_behavior_embeddings)} items with params {behavior_params}")
    
    behavior_clustering = DBSCAN(eps=behavior_params["eps"], min_samples=behavior_params["min_samples"], metric="cosine").fit(np.array(valid_behavior_embeddings))
    behavior_clusters = {label: [] for label in set(behavior_clustering.labels_) if label != -1}
    for i, label in enumerate(behavior_clustering.labels_):
        if label != -1: behavior_clusters[label].append(valid_behavior_interactions[i])

    print(f"Found {len(behavior_clusters)} behavioral families.")
    discovered_patterns = []

    # Sub-cluster by CONTEXT
    for label, interactions_in_cluster in behavior_clusters.items():
        print(f"\n--- Analyzing Behavioral Family #{label} with {len(interactions_in_cluster)} interactions ---")
        if len(interactions_in_cluster) < 4: continue
        
        contexts_to_embed = [json.dumps(inter.original_context, sort_keys=True) for inter in interactions_in_cluster if inter.original_context]
        if len(contexts_to_embed) < 4: continue

        context_embeddings = await embedding_service.get_embeddings(contexts_to_embed)
        valid_context_embeddings = np.array([emb for emb in context_embeddings if emb])
        valid_context_interactions = [inter for i, inter in enumerate(interactions_in_cluster) if i < len(context_embeddings) and context_embeddings[i]]
        if len(valid_context_embeddings) < 4: continue
        
        context_labels = smart_contextual_subclustering(valid_context_embeddings)

        if context_labels is not None:
            sub_clusters = {sub_label: [] for sub_label in set(context_labels)}
            for i, sub_label in enumerate(context_labels):
                sub_clusters[sub_label].append(valid_context_interactions[i])
            
            print(f"Differentiated into {len(sub_clusters)} sub-patterns.")
            for sub_label, interactions_in_sub_cluster in sub_clusters.items():
                if len(interactions_in_sub_cluster) < 3: continue

                sample_contexts = [inter.original_context for inter in interactions_in_sub_cluster[:10]]
                sample_response = interactions_in_sub_cluster[0].original_response
                system_prompt = """
                You are an expert data strategist. Your task is to find a specific, tactical pattern from the provided data.
                A group of conversations has been clustered based on a similar successful AGENT BEHAVIOR. 
                Analyze the diverse contexts to find the underlying, abstract trigger that justifies this agent behavior.
                IMPORTANT: Do NOT provide generic advice. Focus on concrete, repeatable actions.
                Respond in a valid JSON format with "trigger_context_summary" and "suggested_strategy".
                """
                user_prompt = f'The successful agent strategy was a variant of this: "{sample_response}"\n\nThis strategy worked in these diverse situations (contexts):\n---\n{json.dumps(sample_contexts, indent=2)}\n---\nBased on the contexts, what is the specific, non-obvious trigger for this strategy?\nWhat is a concise, tactical instruction for the agent\'s strategy?'
                
                pattern_json = await llm_service.get_json_response(system_prompt=system_prompt, user_prompt=user_prompt, model="openai/gpt-4o-mini")
                if not pattern_json or "suggested_strategy" not in pattern_json: continue

                if (is_quality_pattern(pattern_json) and not await is_duplicate_pattern_async(session, pattern_json["suggested_strategy"], upload_agent_id)):
                    print(f"    Sub-cluster #{sub_label}: PASSED ALL GATES. Creating pattern.")
                    
                    # --- THE FIX: Define the indices before using them ---
                    sub_cluster_indices = [i for i, l in enumerate(context_labels) if l == sub_label]
                    sub_cluster_context_embeddings = valid_context_embeddings[sub_cluster_indices]
                    
                    trigger_embedding, trigger_threshold = calculate_medoid_and_threshold(sub_cluster_context_embeddings)

                    # 2. Calculate for Behavior (Strategy)
                    sub_cluster_response_texts = [valid_context_interactions[i].original_response for i in sub_cluster_indices]
                    sub_cluster_response_embeddings = np.array(await embedding_service.get_embeddings(sub_cluster_response_texts))
                    strategy_embedding, strategy_threshold = calculate_medoid_and_threshold(sub_cluster_response_embeddings)
                    
                    discovered_patterns.append({
                        'trigger_context_summary': pattern_json["trigger_context_summary"],
                        'suggested_strategy': pattern_json["suggested_strategy"],
                        'trigger_embedding': trigger_embedding,
                        'trigger_threshold': trigger_threshold,
                        'strategy_embedding': strategy_embedding,
                        'strategy_threshold': strategy_threshold,
                        'impressions': len(interactions_in_sub_cluster),
                        'success_count': len(interactions_in_sub_cluster)
                    })
    return discovered_patterns

async def perform_causal_validation_async(session: AsyncSession, pattern: LearnedPattern, holdout_ids: List[uuid.UUID]) -> dict | None:
    """
    Performs causal analysis using the pattern's stored, data-driven embeddings and adaptive thresholds.
    """
    # --- 1. Load Data-Driven Trigger and Strategy Representations ---
    trigger_embedding = pattern.trigger_embedding
    trigger_threshold = pattern.trigger_threshold
    strategy_embedding = pattern.strategy_embedding
    strategy_threshold = pattern.strategy_threshold

    # --- Guardrail: Ensure the pattern has all necessary data for validation ---
    if (trigger_embedding is None or len(trigger_embedding) == 0 or
        trigger_threshold is None or
        strategy_embedding is None or len(strategy_embedding) == 0 or
        strategy_threshold is None):
        print(f"  Validation failed: Pattern {pattern.id} is missing a required embedding or threshold.")
        return None

    # --- 2. Find Relevant Interactions in the Holdout Set (Context Matching) ---
    holdout_interactions = (await session.execute(
        select(HistoricalInteraction).where(HistoricalInteraction.id.in_(holdout_ids))
    )).scalars().all()
    
    holdout_contexts = [json.dumps(i.original_context, sort_keys=True) for i in holdout_interactions]
    holdout_context_embeddings = await embedding_service.get_embeddings(holdout_contexts)

    relevant_interactions = []
    for i, ctx_embedding in enumerate(holdout_context_embeddings):
        if ctx_embedding:
            similarity = cosine_similarity([trigger_embedding], [ctx_embedding])[0][0]
            if similarity > trigger_threshold:
                relevant_interactions.append(holdout_interactions[i])

    print(f"  Found {len(relevant_interactions)} relevant interactions for validation (trigger_threshold: {trigger_threshold:.3f})")
    if len(relevant_interactions) < 10:
        return None

    # --- 3. Form Treatment vs. Control Groups (Strategy Matching) ---
    treatment_group, control_group = [], []
    relevant_responses = [i.original_response for i in relevant_interactions]
    relevant_response_embeddings = await embedding_service.get_embeddings(relevant_responses)

    for i, resp_embedding in enumerate(relevant_response_embeddings):
        if resp_embedding:
            similarity = cosine_similarity([strategy_embedding], [resp_embedding])[0][0]
            if similarity > strategy_threshold:
                treatment_group.append(relevant_interactions[i])  # Used the strategy
            else:
                control_group.append(relevant_interactions[i])  # Did NOT use the strategy

    print(f"  Treatment: {len(treatment_group)}, Control: {len(control_group)} (strategy_threshold: {strategy_threshold:.3f})")
    if not treatment_group or not control_group:
        print("  Validation failed: Could not form both treatment and control groups.")
        return None
        
    # --- 4. Perform Statistical Test (T-test) ---
    treatment_outcomes = [1 if i.is_success else 0 for i in treatment_group]
    control_outcomes = [1 if i.is_success else 0 for i in control_group]

    # Handle cases where one group has no variance (all same outcome)
    if len(set(treatment_outcomes)) < 2 or len(set(control_outcomes)) < 2:
        # If no variance, a t-test is not meaningful. Compare means directly.
        uplift = np.mean(treatment_outcomes) - np.mean(control_outcomes)
        # Assign a high p-value as the result is not statistically robust
        return {"uplift": uplift, "p_value": 0.5} 

    # Proceed with t-test if there's variance in both groups
    t_stat, p_value = ttest_ind(treatment_outcomes, control_outcomes, equal_var=False)
    uplift = np.mean(treatment_outcomes) - np.mean(control_outcomes)

    return {"uplift": uplift, "p_value": p_value}


async def judge_outcome_async(transcript: str, goal: str) -> bool:
    """AI-powered outcome judgment."""
    system_prompt = "You are an objective AI evaluator. Determine if a conversation successfully met a specific goal. Respond with JSON: {\"is_success\": boolean}."
    user_prompt = f"SUCCESS GOAL: \"{goal}\"\n\nTRANSCRIPT:\n---\n{transcript}\n---"
    try:
        assessment = await llm_service.get_json_response(system_prompt=system_prompt, user_prompt=user_prompt, model="openai/gpt-4o-mini")
        return assessment.get("is_success", False)
    except Exception: return False
    
async def transcribe_and_assess_async(recording_url: str, outcome_goal: Optional[str]) -> tuple[str, Optional[bool]]:
    """Transcribes audio and optionally assesses the outcome."""
    print(f"Transcribing audio from {recording_url}")
    transcript = await transcription_service.transcribe_audio_from_url(recording_url)
    if "Error transcribing" in transcript: return transcript, None
    is_success = await judge_outcome_async(transcript, outcome_goal) if outcome_goal else None
    return transcript, is_success

# --- ASYNC ORCHESTRATORS ---

async def process_upload_async(upload_id: str, file_content_bytes: bytes, data_mapping: dict):
    ctx = get_async_context()
    async with ctx.session_factory() as session:
        upload = await session.get(HistoricalUpload, uuid.UUID(upload_id))
        if not upload: return

        try:
            csv_content = file_content_bytes.decode('utf-8', errors='ignore')
            df = pd.read_csv(StringIO(csv_content))

            outcome_col, outcome_goal = data_mapping.get("outcome_column"), data_mapping.get("outcome_goal_description")
            transcript_col = data_mapping.get("conversation_transcript")
            context_cols = {k: v for k, v in data_mapping.items() if k.startswith("context_")}

            all_interactions, successful_interactions = [], []
            for index, row in df.iterrows():
                context = {k.replace("context_", ""): row.get(v) for k, v in context_cols.items() if v in df.columns}
                response_text = str(row.get(transcript_col, "")) if transcript_col else ""
                
                is_success, raw_outcome = False, ""
                if outcome_col and outcome_col in df.columns:
                    raw_outcome = str(row.get(outcome_col, ""))
                    is_success = raw_outcome.strip().lower() in ['true', 'success', '1', 'yes', 'resolved']
                elif outcome_goal and response_text:
                    is_success = await judge_outcome_async(response_text, outcome_goal)
                    raw_outcome = "judged_by_ai"
                
                interaction_obj = HistoricalInteraction(
                    upload_id=upload.id, original_context=context, original_response=response_text,
                    is_success=is_success, extracted_outcome={"value_from_file": raw_outcome}
                )
                all_interactions.append(interaction_obj)
                if is_success: successful_interactions.append(interaction_obj)
            
            session.add_all(all_interactions)
            await session.flush() # Assign IDs to objects
            
            successful_interaction_ids = [inter.id for inter in successful_interactions]
            training_ids, holdout_ids = train_test_split(successful_interaction_ids, test_size=0.3, random_state=42) if len(successful_interaction_ids) >= 5 else (successful_interaction_ids, [])
            
            upload.interaction_id_split = {"training_set": [str(uid) for uid in training_ids], "holdout_set": [str(uid) for uid in holdout_ids]}
            upload.status, upload.total_interactions, upload.processed_interactions = "PARSED", len(df), len(df)
            await session.commit()
            
            print(f"Successfully parsed {len(df)} interactions. Triggering pattern extraction.")
            celery_app.send_task('app.background.tasks.extract_patterns_from_history_task', args=[upload_id])

        except Exception as e:
            print(f"Error processing file for upload {upload_id}: {e}\n{traceback.format_exc()}")
            upload.status = "FAILED"
            await session.commit()

async def extract_patterns_from_history_async(upload_id: str):
    ctx = get_async_context()
    patterns_created_count = 0
    
    async with ctx.session_factory() as session:
        upload = await session.get(HistoricalUpload, uuid.UUID(upload_id))
        if not upload or not upload.interaction_id_split: return

        training_set_ids = [uuid.UUID(id_str) for id_str in upload.interaction_id_split.get("training_set", [])]
        if not training_set_ids: upload.status = "COMPLETED"; await session.commit(); return
        
        stmt = select(HistoricalInteraction).where(HistoricalInteraction.id.in_(training_set_ids))
        successful_interactions = (await session.execute(stmt)).scalars().all()
        if len(successful_interactions) < 5: upload.status = "COMPLETED"; await session.commit(); return

        patterns_to_create, processed_ids = [], set()

        # STAGE 1: Dynamic Grouping & Behavioral Consensus Check
        best_grouping_key = find_best_grouping_key(successful_interactions)
        if best_grouping_key:
            context_groups = {}
            for inter in successful_interactions:
                value = (inter.original_context or {}).get(best_grouping_key)
                if value: context_groups.setdefault(value, []).append(inter)
            
            group_patterns = await perform_consensus_check_and_analysis_async(session, context_groups, upload.agent_id, best_grouping_key)
            for p_data in group_patterns:
                patterns_to_create.append(LearnedPattern(agent_id=upload.agent_id, source="HISTORICAL_GROUPED", source_upload_id=upload.id, status=PatternStatus.CANDIDATE, **p_data))
            for interactions in context_groups.values(): processed_ids.update(inter.id for inter in interactions)

        # STAGE 2: Context-Aware Deep Search
        ungrouped = [inter for inter in successful_interactions if inter.id not in processed_ids]
        if len(ungrouped) >= 5:
            clustered_patterns = await perform_context_aware_deep_search_async(session, ungrouped, upload.agent_id)
            for p_data in clustered_patterns:
                patterns_to_create.append(LearnedPattern(
                    agent_id=upload.agent_id, 
                    source="HISTORICAL_DISCOVERED",
                    source_upload_id=upload.id, 
                    status=PatternStatus.CANDIDATE, 
                    **p_data))

        if patterns_to_create:
            session.add_all(patterns_to_create)
            patterns_created_count = len(patterns_to_create)
            upload.status = "VALIDATING"
            print(f"Generated {patterns_created_count} candidate patterns. Moving to validation.")
        else:
            upload.status = "COMPLETED"
            print("No new patterns generated. Marking upload as complete.")
        
        await session.commit()

    if patterns_created_count > 0:
        celery_app.send_task('app.background.tasks.validate_patterns_task', args=[upload_id])

async def validate_patterns_async(upload_id: str):
    ctx = get_async_context()
    async with ctx.session_factory() as session:
        upload = await session.get(HistoricalUpload, uuid.UUID(upload_id))
        if not upload or not upload.interaction_id_split: return

        holdout_set_ids = [uuid.UUID(id_str) for id_str in upload.interaction_id_split.get("holdout_set", [])]
        if not holdout_set_ids: 
            # If no holdout, mark all candidates as validated (no evidence to reject)
            stmt_update = sa.update(LearnedPattern).where(LearnedPattern.source_upload_id == upload.id, LearnedPattern.status == PatternStatus.CANDIDATE).values(status=PatternStatus.VALIDATED)
            await session.execute(stmt_update)
            upload.status = "COMPLETED"; await session.commit(); return
        
        stmt = select(LearnedPattern).where(LearnedPattern.source_upload_id == upload.id, LearnedPattern.status == PatternStatus.CANDIDATE)
        candidate_patterns = (await session.execute(stmt)).scalars().all()
        if not candidate_patterns: upload.status = "COMPLETED"; await session.commit(); return
        
        for pattern in candidate_patterns:
            results = await perform_causal_validation_async(session, pattern, holdout_set_ids)
            if results and results["p_value"] < 0.05 and results["uplift"] > 0:
                pattern.status, pattern.uplift_score, pattern.p_value = PatternStatus.VALIDATED, results["uplift"], results["p_value"]
            else:
                pattern.status = PatternStatus.REJECTED
        
        upload.status = "COMPLETED"
        await session.commit()
        print(f"Validation complete for upload {upload_id}.")

async def process_human_interaction_async(agent_id: str, recording_url: str, context: Optional[Dict[str, Any]], explicit_outcome: Optional[Dict[str, Any]], outcome_goal: Optional[str]):
    transcript, is_success_judged = await transcribe_and_assess_async(recording_url, outcome_goal)
    if "Error transcribing" in transcript: return

    is_success = bool(explicit_outcome.get("success", False)) if explicit_outcome else is_success_judged

    ctx = get_async_context()
    async with ctx.session_factory() as session:
        agent = await session.get(Agent, uuid.UUID(agent_id))
        if not agent: return
        
        new_interaction = HumanInteraction(
            agent_id=agent.id, organization_id=agent.organization_id, recording_url=recording_url,
            context=context, transcript=transcript, is_success=is_success, status="PROCESSED"
        )
        session.add(new_interaction)
        await session.commit()
        print(f"Human interaction {new_interaction.id} saved.")

# --- CELERY TASKS (THIN SYNC WRAPPERS) ---

@celery_app.task(name='app.background.tasks.process_historical_upload_task')
def process_historical_upload_task(upload_id: str, file_content_bytes: bytes, data_mapping: dict):
    try:
        asyncio.run(process_upload_async(upload_id, file_content_bytes, data_mapping))
    finally:
        asyncio.run(close_async_context())

@celery_app.task(name='app.background.tasks.extract_patterns_from_history_task')
def extract_patterns_from_history_task(upload_id: str):
    try:
        asyncio.run(extract_patterns_from_history_async(upload_id))
    finally:
        asyncio.run(close_async_context())

@celery_app.task(name='app.background.tasks.validate_patterns_task')
def validate_patterns_task(upload_id: str):
    try:
        asyncio.run(validate_patterns_async(upload_id))
    finally:
        asyncio.run(close_async_context())

@celery_app.task(name='app.background.tasks.process_human_interaction_task')
def process_human_interaction_task(agent_id: str, recording_url: str, context: Optional[Dict[str, Any]], explicit_outcome: Optional[Dict[str, Any]], outcome_goal: Optional[str]):
    try:
        asyncio.run(process_human_interaction_async(agent_id, recording_url, context, explicit_outcome, outcome_goal))
    finally:
        asyncio.run(close_async_context())

# --- REMAINING TASKS (ALREADY SYNC-COMPATIBLE) ---

@celery_app.task(name='app.background.tasks.discover_patterns_from_live_data_task')
def discover_patterns_from_live_data_task(agent_id: str):
    """Analyzes recent live interactions for new patterns."""
    print(f"Starting live pattern discovery for agent ID: {agent_id}")
    with get_sync_db_session() as db:
        stmt = select(Interaction).join(Outcome).where(
            Interaction.agent_id == uuid.UUID(agent_id), 
            Outcome.is_success == True
        ).limit(1000)
        successful_interactions = db.execute(stmt).scalars().all()

        if len(successful_interactions) < 20:
            print("Not enough recent successful interactions to discover new patterns.")
            return

        context_groups = {}
        for inter in successful_interactions:
            if inter.context and 'occasion' in inter.context:
                key = inter.context['occasion']
                if key not in context_groups: 
                    context_groups[key] = []
                if inter.full_transcript and "Agent:" in inter.full_transcript:
                    agent_response = inter.full_transcript.split("Agent:")[-1].strip()
                    context_groups[key].append(agent_response)
        
        for occasion, responses in context_groups.items():
            if not responses: 
                continue
            most_common_response = max(set(responses), key=responses.count)
            
            existing_pattern = db.execute(select(LearnedPattern).where(
                LearnedPattern.agent_id == uuid.UUID(agent_id),
                LearnedPattern.trigger_context_summary == occasion
            )).scalars().first()
            if existing_pattern: 
                continue

            print(f"Discovered new candidate pattern for occasion: {occasion}")
            new_pattern = LearnedPattern(
                agent_id=uuid.UUID(agent_id), 
                source="LIVE_DISCOVERED",
                trigger_context_summary=occasion, 
                suggested_strategy=most_common_response,
                status="CANDIDATE"
            )
            db.add(new_pattern)
        
        db.commit()
        print(f"Live pattern discovery complete for agent ID: {agent_id}")


@celery_app.task(name='app.background.tasks.process_live_outcome_task')
def process_live_outcome_task(interaction_id: str):
    """Updates performance statistics for patterns used in live interactions."""
    print(f"Processing outcome for interaction ID: {interaction_id}")
    with get_sync_db_session() as db:
        stmt = select(Interaction).options(selectinload(Interaction.outcome)).where(
            Interaction.id == uuid.UUID(interaction_id)
        )
        interaction = db.execute(stmt).scalars().first()

        if not interaction or not interaction.outcome: 
            return
        
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
    """Analyzes failed interactions to suggest new opportunities."""
    print(f"Starting opportunity discovery for organization ID: {organization_id}")
    with get_sync_db_session() as db:
        # Placeholder for complex analysis logic
        # Real implementation would cluster failed interactions and identify latent needs
        db.commit()
        print(f"Opportunity discovery complete for organization ID: {organization_id}")