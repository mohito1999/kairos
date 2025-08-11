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

SUCCESS_RATIO = 0.6  # 60% success, 40% failure
TOTAL_INTERACTIONS = 100

SCENARIO = {
    "agent_persona_success": (
        "You are 'Alex,' a friendly but efficient booking agent for the 'Wanderlust Travel' agency. "
        "Your goal is to understand the customer's needs and qualify them for a follow-up call with a travel specialist. "
        "You MUST end the conversation with a clear confirmation such as 'A specialist will call you back shortly.'"
    ),
    "agent_persona_failure": (
        "You are 'Alex,' a friendly but efficient booking agent for the 'Wanderlust Travel' agency. "
        "You try to qualify the customer, but in this call, you fail to book or confirm a follow-up. "
        "The user ends the call without agreeing to a follow-up."
    ),
    "customer_personas": [
        {"type": "family", "description": "A parent planning a vacation for two adults and two young children. They are moderately budget-conscious but prioritize kid-friendly activities."},
        {"type": "honeymoon", "description": "A newly married couple looking for a romantic, luxurious getaway. They are not very price-sensitive."},
        {"type": "group", "description": "A group of 4-6 college friends planning a fun, activity-packed trip. They are very budget-conscious."},
        {"type": "solo_adventurer", "description": "A solo traveler looking for an adventurous, off-the-beaten-path experience. They are flexible on budget but want unique activities."},
        {"type": "last_minute_business", "description": "A business professional who needs to book a simple, efficient trip for a conference next week. They care only about speed and convenience."}
    ]
}

client = openai.AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

shutdown_event = asyncio.Event()

def handle_shutdown_signal():
    print("\nCtrl+C detected. Shutting down gracefully...")
    shutdown_event.set()

def build_task_list():
    """Pre-allocate a list of (persona, outcome_type) pairs based on ratio."""
    num_success = int(TOTAL_INTERACTIONS * SUCCESS_RATIO)
    num_failure = TOTAL_INTERACTIONS - num_success

    tasks_config = [("success", random.choice(SCENARIO["customer_personas"])) for _ in range(num_success)] + \
                   [("failure", random.choice(SCENARIO["customer_personas"])) for _ in range(num_failure)]

    random.shuffle(tasks_config)
    return tasks_config

async def generate_interaction(outcome_type: str, persona: dict):
    """Generate a conversation with predetermined outcome."""
    if outcome_type == "success":
        agent_persona = SCENARIO["agent_persona_success"]
    else:
        agent_persona = SCENARIO["agent_persona_failure"]

    messages = [
        {"role": "system", "content": agent_persona},
        {"role": "user", "content": f"Simulate a phone call with me. I am a customer with the following profile: {persona['description']}. Please begin the conversation as the agent."}
    ]

    headers = {
        "HTTP-Referer": "http://localhost",
        "X-Title": "Kairos Test Data Generator",
    }

    transcript = ""

    for _ in range(random.randint(3, 6)):
        if shutdown_event.is_set():
            return None

        # Agent turn
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            extra_headers=headers
        )
        agent_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": agent_message})
        transcript += f"Agent: {agent_message}\n"

        # User turn
        user_response_prompt = "Now, as the user, provide a realistic response to the agent's last message. Keep it brief."
        messages.append({"role": "user", "content": user_response_prompt})
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            temperature=0.9,
            extra_headers=headers
        )
        user_message = response.choices[0].message.content
        messages.pop()
        messages.append({"role": "user", "content": user_message})
        transcript += f"User: {user_message}\n"

    return {
        "call_transcript": transcript.strip(),
        "call_outcome": outcome_type,
        "customer_type": persona["type"],
        "region": random.choice(["USA", "UK", "Canada", "Australia", "India"])
    }

async def main():
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, handle_shutdown_signal)

    print(f"Generating {TOTAL_INTERACTIONS} test interactions with a {int(SUCCESS_RATIO*100)}/{int((1-SUCCESS_RATIO)*100)} success/failure ratio...")
    tasks_config = build_task_list()

    tasks = [generate_interaction(outcome, persona) for outcome, persona in tasks_config]
    interactions = []

    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        if shutdown_event.is_set():
            for task in tasks:
                if not task.done():
                    task.cancel()
            break
        try:
            result = await future
            if result:
                interactions.append(result)
        except Exception as e:
            print(f"\nAn error occurred in a task: {e}")

    df = pd.DataFrame(interactions)
    if df.empty:
        print("No interactions generated. Exiting.")
        return

    df['full_name'] = [f"User {i+1}" for i in range(len(df))]
    df['phone_number'] = [f"+1555000{str(i+1).zfill(4)}" for i in range(len(df))]
    df = df[['full_name', 'phone_number', 'call_transcript', 'call_outcome', 'customer_type', 'region']]

    output_path = "generated_test_data.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ Successfully generated {len(df)} interactions and saved to {os.path.abspath(output_path)}")
    print("\n--- Outcome Distribution ---")
    print(df['call_outcome'].value_counts())

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: handle_shutdown_signal())
    asyncio.run(main())
