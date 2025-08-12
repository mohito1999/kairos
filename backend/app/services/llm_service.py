# backend/app/services/llm_service.py (Revised)
import json
from app.core.async_context import get_async_context

async def get_json_response(system_prompt: str, user_prompt: str, model: str = "openai/gpt-4o-mini"):
    """
    Gets a structured JSON response from an LLM via OpenRouter.
    Fetches the client from the async context to ensure it's process-safe.
    """
    async_context = get_async_context()
    client = async_context.openrouter_client # Use the new property

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            extra_headers={ "HTTP-Referer": "http://localhost", "X-Title": "Kairos Engine" }
        )
        content = response.choices[0].message.content
        return json.loads(content) if content else {}
    except Exception as e:
        print(f"Error getting JSON response from LLM: {e}")
        return {}