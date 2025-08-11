from openai import AsyncOpenAI
from app.core.config import settings
from typing import List

# We use a separate client for OpenAI's direct API for embeddings
client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
)

EMBEDDING_MODEL = "text-embedding-3-small"

async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates vector embeddings for a list of texts using OpenAI.
    Handles rate limits by automatically batching large requests.
    """
    if not texts or not isinstance(texts, list):
        return []

    # OpenAI's API has a limit on the number of texts per request (e.g., 2048)
    # and total tokens. We'll use a conservative batch size to stay safe.
    BATCH_SIZE = 100 
    all_embeddings = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        try:
            # Replace empty strings with a single space
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
            # Add empty lists for the failed batch to maintain index order
            all_embeddings.extend([[] for _ in batch_texts])
            
    return all_embeddings


async def get_embedding(text: str) -> List[float]:
    """
    Generates a vector embedding for a single text.
    """
    embeddings = await get_embeddings([text])
    return embeddings[0] if embeddings else []