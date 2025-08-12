# tools/analyze_clusters.py - v2 with Sanity Checks

import os
import json
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import random
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
dotenv_path = Path(__file__).parent.parent / 'backend' / '.env'
load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in backend/.env file.")

SOURCE_CSV_PATH = "./processed_insurance_calls_500.csv"
OUTPUT_HISTOGRAM_PATH = "./cluster_similarity_histograms.png"

# --- SELF-CONTAINED SERVICES ---
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """A self-contained embedding function for this script."""
    if not texts: return []
    BATCH_SIZE = 50
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [t.replace("\n", " ") or " " for t in texts[i:i + BATCH_SIZE]]
        try:
            response = await client.embeddings.create(input=batch, model="text-embedding-3-small")
            all_embeddings.extend([item.embedding for item in response.data])
        except Exception as e:
            print(f"  Error in embedding batch: {e}")
            all_embeddings.extend([[] for _ in batch])
    return all_embeddings

# --- MAIN ANALYSIS LOGIC ---

async def analyze_data(cluster_assignments: dict, source_df_path: str):
    """Performs sanity checks on embedding data and cluster cohesion."""
    print(f"Loading data from {source_df_path}...")
    df = pd.read_csv(source_df_path)
    all_cluster_data = {}

    for cluster_id, filenames in cluster_assignments.items():
        if not filenames: continue
        print(f"\nProcessing {cluster_id} with {len(filenames)} members...")
        
        cluster_transcripts = df[df['source_file'].isin(filenames)]['transcript'].tolist()
        if len(cluster_transcripts) < 2: continue

        context_embeddings = np.array([emb for emb in await get_embeddings(cluster_transcripts) if emb])
        if len(context_embeddings) < 2: continue

        # --- SANITY CHECK 1: VECTOR NORMALIZATION ---
        norms = np.linalg.norm(context_embeddings, axis=1)
        print(f"  Vector Norm Analysis (should be ~1.0):")
        print(f"    Min Norm:  {np.min(norms):.4f}")
        print(f"    Max Norm:  {np.max(norms):.4f}")
        print(f"    Mean Norm: {np.mean(norms):.4f}")
        if not np.allclose(norms, 1.0, atol=1e-5):
            print("  WARNING: Vectors are not L2-normalized! This can distort cosine similarity.")
            # Forcing normalization for the rest of the analysis
            context_embeddings = context_embeddings / norms[:, np.newaxis]
            print("  ACTION: Vectors have been manually normalized for this analysis.")

        # --- SANITY CHECK 2: INTRA-CLUSTER SIMILARITY ---
        centroid = np.mean(context_embeddings, axis=0)
        similarities_to_centroid = cosine_similarity(context_embeddings, [centroid])
        medoid_index = np.argmax(similarities_to_centroid)
        medoid = context_embeddings[medoid_index]
        
        similarities_to_medoid = cosine_similarity(context_embeddings, [medoid]).flatten()
        all_cluster_data[cluster_id] = {'similarities': similarities_to_medoid}

        print(f"  Intra-Cluster Similarity to MEDOID:")
        print(f"    Min:      {np.min(similarities_to_medoid):.3f}")
        print(f"    10th Pct:   {np.percentile(similarities_to_medoid, 10):.3f}  <-- Candidate Adaptive Threshold")
        print(f"    Median:   {np.median(similarities_to_medoid):.3f}")
        print(f"    90th Pct:   {np.percentile(similarities_to_medoid, 90):.3f}")
        print(f"    Max:      {np.max(similarities_to_medoid):.3f}")
    
    # Plotting logic remains the same...
    if all_cluster_data:
        num_clusters = len(all_cluster_data)
        fig, axes = plt.subplots(1, num_clusters, figsize=(5 * num_clusters, 4), sharey=True, squeeze=False)
        axes = axes.flatten()
        
        for i, (cluster_id, data) in enumerate(all_cluster_data.items()):
            ax = axes[i]
            ax.hist(data['similarities'], bins=20, range=(0.4, 1.0)) # Adjust range for better viz
            ax.set_title(f'Similarity for {cluster_id}')
            ax.set_xlabel('Cosine Similarity to Medoid')
        axes[0].set_ylabel('Frequency')
            
        plt.tight_layout()
        plt.savefig(OUTPUT_HISTOGRAM_PATH)
        print(f"\nSaved similarity histograms to: {os.path.abspath(OUTPUT_HISTOGRAM_PATH)}")
    
    await client.close()


async def main():
    df = pd.read_csv(SOURCE_CSV_PATH)
    all_files = df['source_file'].tolist()
    random.shuffle(all_files) # Simulate the random train/test split
    training_files = all_files[:298]
    
    # Use the actual cluster sizes from your last successful discovery run
    sizes = [80, 105, 61, 26, 26] 
    
    cluster_assignments = {}
    start_idx = 0
    for i, size in enumerate(sizes):
        end_idx = start_idx + size
        if end_idx > len(training_files):
             print(f"Warning: Not enough files to populate cluster {i} fully.")
             size = len(training_files) - start_idx
             end_idx = start_idx + size
        cluster_assignments[f"cluster_{i}"] = training_files[start_idx:end_idx]
        start_idx = end_idx
        
    await analyze_data(cluster_assignments, SOURCE_CSV_PATH)

if __name__ == "__main__":
    asyncio.run(main())
