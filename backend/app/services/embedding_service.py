# backend/app/services/embedding_service.py

from typing import List
from app.core.async_context import get_async_context

EMBEDDING_MODEL = "text-embedding-3-small"

async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates vector embeddings for a list of texts using OpenAI.
    Handles rate limits by automatically batching large requests.
    Fetches the client from the async context to ensure it's process-safe.
    """
    if not texts or not isinstance(texts, list):
        return []

    # Fetch the async context and its OpenAI client
    async_context = get_async_context()
    client = async_context.openai_client
    
    BATCH_SIZE = 50
    all_embeddings = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        try:
            batch_texts = [text.replace("\n", " ") or " " for text in batch_texts]
            
            print(f"  Embedding batch {i//BATCH_SIZE + 1} of {len(texts)//BATCH_SIZE + 1}...")
            response = await client.embeddings.create(
                input=batch_texts,
                model=EMBEDDING_MODEL
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"  Error processing embedding batch: {e}")
            all_embeddings.extend([[] for _ in batch_texts])
            
    return all_embeddings


async def get_embedding(text: str) -> List[float]:
    """
    Generates a vector embedding for a single text.
    """
    embeddings = await get_embeddings([text])
    return embeddings[0] if embeddings else []