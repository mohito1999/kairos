# tools/generate_deep_search_data.py
import os
import openai
import pandas as pd
from tqdm import tqdm
import random
import asyncio
import signal

# --- CONFIGURATION ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")

TOTAL_INTERACTIONS = 100
# We'll create a "hidden" pattern in about 30% of the successful calls
HIDDEN_PATTERN_RATIO = 0.3 

# --- SCENARIO DEFINITION ---
SCENARIO = {
    "agent_persona_standard": (
        "You are 'Alex,' a friendly booking agent for 'Wanderlust Travel'. Your goal is to qualify the customer for a follow-up. "
        "Conclude by confirming a specialist will call them."
    ),
    # THIS IS THE HIDDEN WINNING STRATEGY
    "agent_persona_hesitant_handler": (
        "You are 'Alex,' a friendly booking agent. A customer seems hesitant and mentions needing to talk to a partner. "
        "Your goal is to overcome this objection and secure a follow-up. Acknowledge their hesitation, offer to send a shareable summary via WhatsApp, "
        "and then confirm a specialist will call them."
    ),
    # THIS IS THE HIDDEN CUSTOMER BEHAVIOR
    "customer_persona_hesitant": {
        "description": "A customer who is interested but expresses hesitation, often mentioning needing to 'check with my wife/husband/partner' before committing."
    },
    "customer_persona_standard": {
        "description": "A standard customer who is ready to book a trip for a vacation."
    },
}

client = openai.AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

shutdown_event = asyncio.Event()

def handle_shutdown_signal():
    print("\nCtrl+C detected. Shutting down gracefully...")
    shutdown_event.set()

async def generate_interaction(is_hidden_pattern: bool):
    """Generates a conversation, injecting a hidden pattern for a subset of calls."""
    
    if is_hidden_pattern:
        agent_persona = SCENARIO["agent_persona_hesitant_handler"]
        customer_persona_desc = SCENARIO["customer_persona_hesitant"]["description"]
        # This will be a successful interaction
        outcome = "success"
    else:
        agent_persona = SCENARIO["agent_persona_standard"]
        customer_persona_desc = SCENARIO["customer_persona_standard"]["description"]
        # Standard calls have a random outcome
        outcome = "success" if random.random() < 0.5 else "failure"

    messages = [
        {"role": "system", "content": agent_persona},
        {"role": "user", "content": f"Simulate a phone call with me. I am a customer with the following profile: {customer_persona_desc}. Please begin the conversation as the agent."}
    ]
    
    headers = { "HTTP-Referer": "http://localhost", "X-Title": "Kairos Deep Search Generator" }
    transcript = ""

    for _ in range(random.randint(4, 6)): # Longer conversations for more nuance
        if shutdown_event.is_set(): return None

        # Agent turn
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini", messages=messages, temperature=0.7, extra_headers=headers
        )
        agent_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": agent_message})
        transcript += f"Agent: {agent_message}\n"

        # User turn
        user_response_prompt = "Now, as the user, provide a realistic response to the agent's last message. Keep it brief."
        messages.append({"role": "user", "content": user_response_prompt})
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini", messages=messages, temperature=0.9, extra_headers=headers
        )
        user_message = response.choices[0].message.content
        messages.pop()
        messages.append({"role": "user", "content": user_message})
        transcript += f"User: {user_message}\n"

    return {
        "call_transcript": transcript.strip(),
        "call_outcome": outcome,
        # CRITICAL: We are NOT providing a structured key for the hidden pattern.
        # It only exists in the transcript text. We'll use a generic customer type.
        "customer_type": "generic_lead",
        "region": random.choice(["USA", "UK", "Canada", "Australia", "India"])
    }

async def main():
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, handle_shutdown_signal)

    print(f"Generating {TOTAL_INTERACTIONS} test interactions for Deep Search validation...")
    
    num_hidden_pattern = int(TOTAL_INTERACTIONS * HIDDEN_PATTERN_RATIO)
    tasks = [generate_interaction(is_hidden_pattern=True) for _ in range(num_hidden_pattern)] + \
            [generate_interaction(is_hidden_pattern=False) for _ in range(TOTAL_INTERACTIONS - num_hidden_pattern)]
    random.shuffle(tasks)

    interactions = []
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        if shutdown_event.is_set():
            for task in tasks:
                if not task.done(): task.cancel()
            break
        try:
            result = await future
            if result: interactions.append(result)
        except Exception as e:
            print(f"\nAn error occurred in a task: {e}")

    df = pd.DataFrame(interactions)
    if df.empty:
        print("No interactions generated. Exiting.")
        return

    df['full_name'] = [f"User {i+1}" for i in range(len(df))]
    df['phone_number'] = [f"+1555000{str(i+1).zfill(4)}" for i in range(len(df))]
    df = df[['full_name', 'phone_number', 'call_transcript', 'call_outcome', 'customer_type', 'region']]
    
    output_path = "generated_deep_search_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Successfully generated {len(df)} interactions and saved to {os.path.abspath(output_path)}")
    print("\n--- Outcome Distribution ---")
    print(df['call_outcome'].value_counts())

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: handle_shutdown_signal())
    asyncio.run(main())