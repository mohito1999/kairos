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
    """
    if not texts or not isinstance(texts, list):
        return []
    
    try:
        # Replace empty strings with a single space, as the API doesn't handle them
        texts = [text.replace("\n", " ") or " " for text in texts]
        
        response = await client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return [[] for _ in texts] # Return empty lists on failure

async def get_embedding(text: str) -> List[float]:
    """
    Generates a vector embedding for a single text.
    """
    embeddings = await get_embeddings([text])
    return embeddings[0] if embeddings else []