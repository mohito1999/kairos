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

# --- DYNAMIC PATHS to load .env files ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

dotenv_path_frontend = os.path.join(PROJECT_ROOT, 'frontend', '.env.local')
load_dotenv(dotenv_path=dotenv_path_frontend)
dotenv_path_backend = os.path.join(PROJECT_ROOT, 'backend', '.env')
load_dotenv(dotenv_path=dotenv_path_backend)

# Use the RAW database URL for direct connection from local machine
DATABASE_URL = os.getenv("RAW_DATABASE_URL")

# --- MOCKED DATA ---
OPPORTUNITY_DATA = [
    # The original 5
    {"transcript": "User: Hi, do you guys have a driving range? Agent: I'm sorry, we only book full golf course packages.", "is_success": False},
    {"transcript": "User: I don't need a full 18 holes, I just want to practice my swing. Do you have a driving range? Agent: Apologies, we don't offer that service.", "is_success": False},
    {"transcript": "User: I'm looking for a place to just hit a bucket of balls. Agent: We don't have a driving range available for booking.", "is_success": False},
    {"transcript": "User: My friends and I just want to go to a driving range, not a full course. Agent: I can't help with that, sorry.", "is_success": False},
    {"transcript": "User: Is there a driving range I can book through you? Agent: Unfortunately, that's not a service we provide.", "is_success": False},
    # Add 15 more to reach 20
    {"transcript": "User: Can I book just the driving range? Agent: No, that's not an option.", "is_success": False},
    {"transcript": "User: We just want to practice at the range. Agent: Our packages are for full rounds only.", "is_success": False},
    {"transcript": "User: How much for just the driving range? Agent: That service is not available.", "is_success": False},
    {"transcript": "User: Looking for a driving range booking. Agent: We can't book that.", "is_success": False},
    {"transcript": "User: Is it possible to only use the driving range? Agent: I'm afraid not.", "is_success": False},
    {"transcript": "User: We don't want to play a full game, just the driving range. Agent: I cannot help with that.", "is_success": False},
    {"transcript": "User: Driving range only, please. Agent: That is not something we offer.", "is_success": False},
    {"transcript": "User: My son wants to learn at a driving range. Agent: We don't have packages for that.", "is_success": False},
    {"transcript": "User: I need a booking for a driving range session. Agent: We do not handle those bookings.", "is_success": False},
    {"transcript": "User: Can you find a local driving range for me? Agent: That is outside the scope of our services.", "is_success": False},
    {"transcript": "User: Just the range, not the course. Agent: Sorry, that's not possible.", "is_success": False},
    {"transcript": "User: We are beginners and just need a driving range. Agent: We only offer full course experiences.", "is_success": False},
    {"transcript": "User: I was told you might have driving range access. Agent: I'm sorry, that information is incorrect.", "is_success": False},
    {"transcript": "User: A bucket of balls at the driving range is all I need. Agent: We can't accommodate that request.", "is_success": False},
    {"transcript": "User: Forget the course, what about the driving range? Agent: We do not have an option for that.", "is_success": False},
]


def get_auth_token():
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

def setup_test_data(token):
    print("Creating a new test agent and seeding failure data...")
    agent_url = f"{BASE_URL}/api/v1/agents/"
    headers = {"Authorization": f"Bearer {token}"}
    agent_payload = {"name": f"Opportunity Test Agent {int(time.time())}", "objective": "To find new opportunities"}
    agent_res = requests.post(agent_url, headers=headers, json=agent_payload)
    agent_res.raise_for_status()
    agent = agent_res.json()
    agent_id = agent['id']
    org_id = agent['organization_id']
    print(f"✅ Agent created with ID: {agent_id}")

    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        session.execute(text("DELETE FROM suggested_opportunities WHERE organization_id = :org_id"), {"org_id": org_id})
        session.execute(text("DELETE FROM outcomes WHERE interaction_id IN (SELECT id FROM interactions WHERE agent_id = :agent_id)"), {"agent_id": agent_id})
        session.execute(text("DELETE FROM interactions WHERE agent_id = :agent_id"), {"agent_id": agent_id})
        
        for data in OPPORTUNITY_DATA:
            interaction_id = uuid.uuid4()
            session.execute(text("INSERT INTO interactions (id, agent_id, session_id, full_transcript, created_at, updated_at) VALUES (:id, :agent_id, :session_id, :transcript, NOW(), NOW())"),
                {"id": interaction_id, "agent_id": agent_id, "session_id": f"session_{uuid.uuid4()}", "transcript": data["transcript"]})
            session.execute(text("INSERT INTO outcomes (id, interaction_id, source, is_success, created_at, updated_at) VALUES (:id, :interaction_id, 'EXPLICIT', :is_success, NOW(), NOW())"),
                {"id": uuid.uuid4(), "interaction_id": interaction_id, "is_success": data["is_success"]})
        session.commit()
        print(f"✅ Seeded {len(OPPORTUNITY_DATA)} failed interactions into the database.")
        return org_id
    finally:
        session.close()

def trigger_opportunity_task_api(token, org_id):
    """Triggers the task via our debug API endpoint."""
    print(f"Triggering opportunity generation task for organization {org_id} via API...")
    # NOTE: This endpoint doesn't exist yet. We will add it.
    url = f"{BASE_URL}/api/v1/analytics/trigger-opportunity-generation/{org_id}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.post(url, headers=headers)
        response.raise_for_status()
        print("✅ Task triggered successfully. Monitor worker logs for execution.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to trigger task: {e.response.text}")
        exit(1)

def main():
    print("--- Kairos Opportunity Engine E2E Test ---")
    token = get_auth_token()
    org_id = setup_test_data(token)
    trigger_opportunity_task_api(token, org_id)
    print("\n--- TEST SCRIPT COMPLETE ---")

if __name__ == "__main__":
    main()