# tools/process_auto_calls.py

import os
import json
import asyncio
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
import uuid

# --- CONFIGURATION ---
dotenv_path = Path(__file__).parent.parent / 'backend' / '.env'
load_dotenv(dotenv_path=dotenv_path)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in backend/.env file.")

# --- NEW CONFIGURATION FOR AUTOMOTIVE DATASET ---
SOURCE_DIR = "/Users/mohitmotwani/Downloads/automotive_inbound"
OUTPUT_CSV_PATH = "./processed_automotive_calls.csv"
FILES_TO_PROCESS = 500

# --- API CLIENT & HELPER ---
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

async def get_llm_json_response(system_prompt: str, user_prompt: str, model: str):
    """
    A robust, self-contained helper to get a JSON response from the LLM.
    Now includes detailed error logging.
    """
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            extra_headers={ "HTTP-Referer": "http://localhost", "X-Title": "Kairos Tools" }
        )
        content = response.choices[0].message.content
        return json.loads(content) if content else {}
    except openai.APIStatusError as e:
        # --- THE FIX: CATCH AND LOG SPECIFIC API ERRORS ---
        print(f"  LLM API Error: Status {e.status_code}, Response: {e.response.text}")
        return {}
    except json.JSONDecodeError as e:
        print(f"  LLM JSON Decode Error: Failed to parse LLM response. Details: {e}")
        return {}
    except Exception as e:
        # Catch any other unexpected errors
        print(f"  An unexpected LLM call error occurred: {type(e).__name__} - {e}")
        return {}


# --- NEW, TAILORED LLM PROMPTS ---

SUCCESS_JUDGE_SYSTEM_PROMPT = """
You are an AI assistant for Kairos, analyzing inbound sales calls for a car dealership.
Your task is to determine if a call was a "success" or a "failure".
The primary goal of the agent is to satisfy the customer's initial request (e.g., check inventory, get a price) AND secure a concrete next step (e.g., book a test drive, get customer info for a quote, schedule a follow-up call).

A "success" is defined as:
- The agent answers the customer's question.
- A clear next action is agreed upon by the customer.
- The agent captures the customer's contact information.

A "failure" is defined as:
- The customer ends the call without a clear next step.
- The agent cannot answer the customer's question (e.g., "I don't have that information").
- The customer expresses frustration and ends the call.
- The call ends with a vague "I'll call back later" from the customer.

Analyze the transcript and respond with a JSON object with a single key "outcome" and a single word value: "success" or "failure".
"""

CONTEXT_EXTRACTION_SYSTEM_PROMPT = """
You are an expert data extraction AI. Analyze the provided car dealership sales call transcript.
Extract the following entities into a valid JSON object with the specified keys and data types:
- "inquiry_type": string (One of: "inventory_check", "price_quote", "feature_question", "general_inquiry").
- "vehicle_of_interest": string (The specific car model mentioned, e.g., "GLC 300", "Sierra 2500". If none, use "unknown").
- "customer_urgency": string (One of: "high", "medium", "low". High if they need the car 'today' or 'this week'. Low if they are 'just looking'. Medium otherwise).
- "trade_in_mentioned": boolean (True if the customer mentions trading in their current vehicle, otherwise False).

Only return the JSON object. Do not include any explanations.
"""

# --- CORE LOGIC ---

async def process_file(file_path: Path):
    """Processes a single JSON file for transcript, outcome, and new automotive context."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        raw_transcript = data.get("text")
        if not raw_transcript: 
            print(f"  Skipping {file_path.name}: No 'text' field found.")
            return None

        # 1. Judge the outcome
        outcome_response = await get_llm_json_response(
            SUCCESS_JUDGE_SYSTEM_PROMPT, raw_transcript, "openai/gpt-4o-mini"
        )
        # --- ROBUSTNESS FIX ---
        if not outcome_response:
            print(f"  Skipping {file_path.name}: Failed to get outcome from LLM.")
            return None
        
        outcome = "failure"
        if isinstance(outcome_response.get("outcome"), str):
            outcome = outcome_response["outcome"].lower()

        # 2. Extract Context
        context_json = await get_llm_json_response(
            CONTEXT_EXTRACTION_SYSTEM_PROMPT, raw_transcript, "openai/gpt-4o-mini"
        )
        # --- ROBUSTNESS FIX ---
        if not context_json:
            print(f"  Skipping {file_path.name}: Failed to get context from LLM.")
            return None

        return {
            "transcript": raw_transcript,
            "call_outcome": outcome,
            "inquiry_type": context_json.get("inquiry_type", "general_inquiry"),
            "vehicle_of_interest": context_json.get("vehicle_of_interest", "unknown"),
            "customer_urgency": context_json.get("customer_urgency", "medium"),
            "trade_in_mentioned": context_json.get("trade_in_mentioned", False),
            "source_file": file_path.name
        }
    except Exception as e:
        print(f"  Error processing {file_path.name}: {e}")
        return None


async def main():
    source_path = Path(SOURCE_DIR)
    json_files = sorted(list(source_path.glob("*.json")))[:FILES_TO_PROCESS]
    if not json_files:
        print(f"No JSON files found in {SOURCE_DIR}"); return

    print(f"Found {len(json_files)} files to process.")
    
    tasks = [process_file(fp) for fp in json_files]
    results = await tqdm.gather(*tasks, desc="Processing Automotive Transcripts")
    
    successful_results = [res for res in results if res is not None]
    if not successful_results:
        print("No files were successfully processed."); return
        
    df = pd.DataFrame(successful_results)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ Successfully processed {len(successful_results)} files.")
    print(f"   Output saved to: {os.path.abspath(OUTPUT_CSV_PATH)}")
    print("\n--- Outcome Distribution ---")
    print(df['call_outcome'].value_counts())
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())