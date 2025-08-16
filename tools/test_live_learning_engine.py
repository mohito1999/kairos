import requests
import time
import os
import json
import uuid
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# --- CONFIGURATION ---
BASE_URL = "http://localhost:8000"
TEST_USER_EMAIL = "kairos.dev123@gmail.com"
TEST_USER_PASSWORD = "6sK5uUespO5Mqx"

# --- DYNAMIC PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

dotenv_path_frontend = os.path.join(PROJECT_ROOT, 'frontend', '.env.local')
load_dotenv(dotenv_path=dotenv_path_frontend)

dotenv_path_backend = os.path.join(PROJECT_ROOT, 'backend', '.env')
load_dotenv(dotenv_path=dotenv_path_backend)

# Use the RAW database URL for direct connection from local machine
DATABASE_URL = os.getenv("RAW_DATABASE_URL")

def get_auth_token():
    """Authenticates with Supabase to get a JWT."""
    print("Authenticating with Supabase...")
    url = f"{os.getenv('NEXT_PUBLIC_SUPABASE_URL')}/auth/v1/token?grant_type=password"
    headers = {"apikey": os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')}
    payload = {"email": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        token = response.json().get("access_token")
        print("✅ Authentication successful.")
        return token
    except requests.exceptions.RequestException as e:
        print(f"❌ Authentication failed: {e}")
        exit(1)

def create_test_agent(token):
    """Creates a new agent."""
    print("Creating a new test agent for live learning...")
    url = f"{BASE_URL}/api/v1/agents/"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "name": f"Live Learning Agent {int(time.time())}",
        "objective": "To test the live learning pipeline.",
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        agent = response.json()
        print(f"✅ Agent '{agent['name']}' created with ID: {agent['id']}")
        return agent
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to create agent: {e.response.text}")
        exit(1)

def seed_live_data(agent_id):
    """Directly inserts fake 'live' interaction data into the database."""
    print("Seeding the database with sample live interaction data...")
    if not DATABASE_URL:
        print("❌ RAW_DATABASE_URL not found in backend/.env file.")
        exit(1)
    
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        session.execute(text("DELETE FROM outcomes WHERE interaction_id IN (SELECT id FROM interactions WHERE agent_id = :agent_id)"), {"agent_id": agent_id})
        session.execute(text("DELETE FROM interactions WHERE agent_id = :agent_id"), {"agent_id": agent_id})

        interactions_data = [
            {"transcript": "User: That's too expensive. Agent: I understand, but it's a good value.", "is_success": False, "context": {"topic": "price"}},
            {"transcript": "User: Your price is too high. Agent: It seems high, but let me explain the value.", "is_success": False, "context": {"topic": "price"}},
            {"transcript": "User: I can't afford that. Agent: Let's see if we can find a promotion for you.", "is_success": True, "context": {"topic": "price"}},
            {"transcript": "User: Way too much money. Agent: I hear your concern. We do have other tiers.", "is_success": False, "context": {"topic": "price"}},
            {"transcript": "User: The cost is a problem. Agent: I understand. Many clients feel that way at first, but find the ROI is worth it.", "is_success": True, "context": {"topic": "price"}},
            {"transcript": "User: It costs too much. Agent: Okay.", "is_success": False, "context": {"topic": "price"}},
            {"transcript": "User: Is there a discount? Agent: Let me check for any available discounts for you.", "is_success": True, "context": {"topic": "price"}},
        ]

        for data in interactions_data:
            interaction_id = uuid.uuid4()
            session.execute(text("INSERT INTO interactions (id, agent_id, session_id, full_transcript, context, created_at, updated_at) VALUES (:id, :agent_id, :session_id, :transcript, :context, NOW(), NOW())"),
                {"id": interaction_id, "agent_id": agent_id, "session_id": f"session_{uuid.uuid4()}", "transcript": data["transcript"], "context": json.dumps(data["context"])})
            session.execute(text("INSERT INTO outcomes (id, interaction_id, source, is_success, created_at, updated_at) VALUES (:id, :interaction_id, 'EXPLICIT', :is_success, NOW(), NOW())"),
                {"id": uuid.uuid4(), "interaction_id": interaction_id, "is_success": data["is_success"]})
        
        session.commit()
        print(f"✅ Seeded {len(interactions_data)} interactions into the database.")
    except Exception as e:
        session.rollback()
        print(f"❌ Database seeding failed: {e}")
        exit(1)
    finally:
        session.close()

def trigger_live_learning_task_api(token, agent_id):
    """Triggers the task via our new debug API endpoint."""
    print(f"Triggering live learning task for agent {agent_id} via API...")
    url = f"{BASE_URL}/api/v1/analytics/trigger-live-learning/{agent_id}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.post(url, headers=headers)
        response.raise_for_status()
        print("✅ Task triggered successfully. Monitor worker logs for execution.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to trigger task: {e.response.text}")
        exit(1)

def main():
    print("--- Kairos Live Learning Engine E2E Test ---")
    
    token = get_auth_token()
    agent = create_test_agent(token)
    
    seed_live_data(agent['id'])
    trigger_live_learning_task_api(token, agent['id'])
    
    print("\n--- TEST SCRIPT COMPLETE ---")
    print("The live learning pipeline has been started. Check your Docker logs for detailed output.")

if __name__ == "__main__":
    main()