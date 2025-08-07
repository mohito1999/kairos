from openai import AsyncOpenAI
from app.core.config import settings
import json

# We use the OpenAI client, but point it to the OpenRouter base URL
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.OPENROUTER_API_key,
)

async def get_json_response(system_prompt: str, user_prompt: str, model: str = "mistralai/mistral-7b-instruct:free"):
    """
    Gets a structured JSON response from an LLM via OpenRouter.
    Uses a fast, free model by default for tasks like context extraction.
    """
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content) if content else {}
    except Exception as e:
        print(f"Error getting JSON response from LLM: {e}")
        # In a real app, we'd have more robust error handling and logging here
        return {}