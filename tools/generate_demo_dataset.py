# tools/generate_demo_dataset.py

import os
import uuid
import json
import asyncio
import pandas as pd
import random
from pathlib import Path
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

# --- CONFIGURATION ---
dotenv_path = Path(__file__).parent.parent / 'backend' / '.env'
load_dotenv(dotenv_path=dotenv_path)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in backend/.env file.")

OUTPUT_CSV_PATH = "./demo_insurance_calls.csv"
TOTAL_INTERACTIONS = 300

# --- SCENARIO & STRATEGY DEFINITION ---
# We are defining the "ground truth" patterns we want our engine to discover.

SCENARIOS = {
    "happy_path": {
        "context": {"initial_objection": False, "card_retrieval_hesitation": False, "data_correction_needed": False},
        "agent_strategy": "Be warm, efficient, and guide the customer smoothly through the standard script to a transfer.",
        "success_rate": 0.90 # 90% of these calls will be successful
    },
    "hesitant_customer": {
        "context": {"initial_objection": False, "card_retrieval_hesitation": True, "data_correction_needed": False},
        "agent_strategy": "Acknowledge their hesitation, build trust by emphasizing their control and the no-cost nature of the benefits, then gently guide them to retrieve their card before transferring.",
        "success_rate": 0.75 # This is a good strategy, so it's successful 75% of the time.
    },
    "objection_customer": {
        "context": {"initial_objection": True, "card_retrieval_hesitation": False, "data_correction_needed": True},
        "agent_strategy": "Immediately address the objection with an empathetic statement, pivot back to the core benefit (e.g., 'money back'), correct their data collaboratively, and then proceed to the transfer.",
        "success_rate": 0.70 # Another good strategy.
    },
    "bad_strategy_for_hesitation": {
        "context": {"initial_objection": False, "card_retrieval_hesitation": True, "data_correction_needed": False},
        "agent_strategy": "Ignore their hesitation, be very direct, and push aggressively for the transfer without building trust.",
        "success_rate": 0.15 # This is a bad strategy that will mostly fail.
    }
}

# --- API CLIENT & CORE LOGIC ---
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

async def generate_interaction(scenario_name: str):
    """Generates a single, nuanced conversation based on a defined scenario."""
    scenario = SCENARIOS[scenario_name]
    
    system_prompt = f"""
    You are a professional scriptwriter creating training data for an AI.
    Your task is to generate a realistic, multi-turn phone call transcript between an 'Agent' and a 'Customer' for a Medicare insurance outbound call.
    
    AGENT'S OBJECTIVE: Verify the customer and transfer them to a licensed specialist.
    AGENT'S STRATEGY: You MUST adopt the following strategy: "{scenario['agent_strategy']}"
    CUSTOMER'S CONTEXT: The customer's situation is as follows: {json.dumps(scenario['context'])}
    
    Generate a transcript of 6-10 turns. The final turn should clearly indicate if the call is being transferred or if the customer is ending the call.
    Return ONLY the raw text of the transcript.
    """
    
    try:
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.8,
            max_tokens=1024,
        )
        transcript = response.choices[0].message.content.strip()
        
        # Determine outcome based on the defined success rate
        is_success = random.random() < scenario['success_rate']
        
        return {
            "transcript": transcript,
            "call_outcome": "success" if is_success else "failure",
            "initial_objection": scenario['context'].get('initial_objection'),
            "data_correction_needed": scenario['context'].get('data_correction_needed'),
            "card_retrieval_hesitation": scenario['context'].get('card_retrieval_hesitation'),
            "verified_own_decisions": True, # Assume this for simplicity
            "source_file": f"generated_{scenario_name}_{uuid.uuid4()}.json"
        }
    except Exception as e:
        print(f"  Error generating interaction for {scenario_name}: {e}")
        return None

async def main():
    print(f"Generating {TOTAL_INTERACTIONS} interactions for a high-quality demo dataset...")
    
    # Create a weighted list of scenarios to generate
    scenario_list = []
    scenario_list.extend(["happy_path"] * int(TOTAL_INTERACTIONS * 0.4)) # 40% are easy wins
    scenario_list.extend(["hesitant_customer"] * int(TOTAL_INTERACTIONS * 0.25)) # 25% use winning strategy A
    scenario_list.extend(["objection_customer"] * int(TOTAL_INTERACTIONS * 0.25)) # 25% use winning strategy B
    scenario_list.extend(["bad_strategy_for_hesitation"] * int(TOTAL_INTERACTIONS * 0.1)) # 10% use a losing strategy
    random.shuffle(scenario_list)
    
    tasks = [generate_interaction(name) for name in scenario_list]
    results = await tqdm.gather(*tasks, desc="Generating Transcripts")
    
    successful_results = [res for res in results if res is not None]

    if not successful_results:
        print("No interactions were successfully generated."); return
        
    df = pd.DataFrame(successful_results)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ Successfully generated {len(successful_results)} interactions.")
    print(f"   Output saved to: {os.path.abspath(OUTPUT_CSV_PATH)}")
    print("\n--- Outcome Distribution ---")
    print(df['call_outcome'].value_counts())

if __name__ == "__main__":
    asyncio.run(main())