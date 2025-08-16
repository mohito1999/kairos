#tools/test_live_learning_unique.py
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

dotenv_path_frontend = os.path.join(PROJECT_ROOT, 'frontend', '.env.local')
load_dotenv(dotenv_path=dotenv_path_frontend)
dotenv_path_backend = os.path.join(PROJECT_ROOT, 'backend', '.env')
load_dotenv(dotenv_path=dotenv_path_backend)

DATABASE_URL = os.getenv("RAW_DATABASE_URL")

# --- NEW, UNIQUE MOCKED DATA ---
# This data simulates a "hesitation" battleground, which is different from the insurance data's focus.
UNIQUE_LIVE_DATA = [
    # Cluster 1: "I need to think about it" pattern - FAILURES
    {"transcript": "User: I need to think about it. Agent: Okay, let me know.", "is_success": False, "context": {"topic": "hesitation"}},
    {"transcript": "User: I need to think about it more. Agent: Sure, no problem.", "is_success": False, "context": {"topic": "hesitation"}},  
    {"transcript": "User: Let me think about it. Agent: Alright, take your time.", "is_success": False, "context": {"topic": "hesitation"}},
    {"transcript": "User: I'll have to think. Agent: I understand.", "is_success": False, "context": {"topic": "hesitation"}},
    {"transcript": "User: Let me think this over. Agent: Of course.", "is_success": False, "context": {"topic": "hesitation"}},
    {"transcript": "User: I need some time to think. Agent: No worries.", "is_success": False, "context": {"topic": "hesitation"}},
    
    # Cluster 1: "I need to think about it" pattern - SUCCESSES (same user statements, better agent responses)
    {"transcript": "User: I need to think about it. Agent: I understand, while you think, can I follow up next week with any questions you might have?", "is_success": True, "context": {"topic": "hesitation"}},
    {"transcript": "User: I need to think about it more. Agent: Absolutely, that's smart. Can I send you a summary to review while you decide?", "is_success": True, "context": {"topic": "hesitation"}},
    {"transcript": "User: Let me think about it. Agent: Of course, take your time. Should I call you back on Friday to see how you're feeling?", "is_success": True, "context": {"topic": "hesitation"}},
    {"transcript": "User: I'll have to think. Agent: That's wise. While you do, can I answer any specific questions that might help?", "is_success": True, "context": {"topic": "hesitation"}},
    {"transcript": "User: Let me think this over. Agent: Absolutely. Would it help if I sent over some additional information to consider?", "is_success": True, "context": {"topic": "hesitation"}},
    {"transcript": "User: I need some time to think. Agent: Perfect, thinking it through is important. Can I check in with you next Tuesday?", "is_success": True, "context": {"topic": "hesitation"}},
    
    # Cluster 2: "Talk to spouse/partner" pattern - FAILURES  
    {"transcript": "User: Let me talk to my wife first. Agent: No problem.", "is_success": False, "context": {"topic": "consultation"}},
    {"transcript": "User: I need to discuss with my husband. Agent: Sure thing.", "is_success": False, "context": {"topic": "consultation"}},
    {"transcript": "User: Let me run this by my partner. Agent: Okay.", "is_success": False, "context": {"topic": "consultation"}},
    {"transcript": "User: I should talk to my spouse. Agent: Of course.", "is_success": False, "context": {"topic": "consultation"}},
    {"transcript": "User: Need to check with my wife. Agent: No worries.", "is_success": False, "context": {"topic": "consultation"}},
    
    # Cluster 2: "Talk to spouse/partner" pattern - SUCCESSES
    {"transcript": "User: Let me talk to my wife first. Agent: That's great that you discuss important decisions together. Can I send you a summary to share with her?", "is_success": True, "context": {"topic": "consultation"}},
    {"transcript": "User: I need to discuss with my husband. Agent: Smart approach. Would a brief overview document help with your discussion?", "is_success": True, "context": {"topic": "consultation"}},
    {"transcript": "User: Let me run this by my partner. Agent: Absolutely, that shows good judgment. Should I prepare a simple breakdown you can show them?", "is_success": True, "context": {"topic": "consultation"}},
    {"transcript": "User: I should talk to my spouse. Agent: Of course, important decisions should be discussed. Can I follow up after you've had that conversation?", "is_success": True, "context": {"topic": "consultation"}},
    {"transcript": "User: Need to check with my wife. Agent: That makes complete sense. Would it help if I put together the key points for your discussion?", "is_success": True, "context": {"topic": "consultation"}},
    
    # Additional data for more robust clustering
    {"transcript": "User: Maybe later. Agent: Alright, no worries.", "is_success": False, "context": {"topic": "hesitation"}},
    {"transcript": "User: I'm undecided. Agent: Okay, I understand.", "is_success": False, "context": {"topic": "hesitation"}},
    {"transcript": "User: Maybe later. Agent: Sure thing. How about I check back with you next month at a better time?", "is_success": True, "context": {"topic": "hesitation"}},
    {"transcript": "User: I'm undecided. Agent: That's totally normal. Would seeing a quick demo help you get a better feel for it?", "is_success": True, "context": {"topic": "hesitation"}},
]


def get_auth_token():
    print("Authenticating...")
    url = f"{os.getenv('NEXT_PUBLIC_SUPABASE_URL')}/auth/v1/token?grant_type=password"
    headers = {"apikey": os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')}
    payload = {"email": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['access_token']

def setup_test_data(token):
    print("Creating agent and seeding unique live data...")
    agent_url = f"{BASE_URL}/api/v1/agents/"
    headers = {"Authorization": f"Bearer {token}"}
    agent_payload = {"name": f"Unique Live Test Agent {int(time.time())}", "objective": "Test unique pattern discovery"}
    agent_res = requests.post(agent_url, headers=headers, json=agent_payload)
    agent_res.raise_for_status()
    agent = agent_res.json()
    agent_id = agent['id']
    print(f"✅ Agent created with ID: {agent_id}")

    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        # Clear old data for this specific agent
        session.execute(text("DELETE FROM outcomes WHERE interaction_id IN (SELECT id FROM interactions WHERE agent_id = :agent_id)"), {"agent_id": agent_id})
        session.execute(text("DELETE FROM interactions WHERE agent_id = :agent_id"), {"agent_id": agent_id})
        
        for data in UNIQUE_LIVE_DATA:
            interaction_id = uuid.uuid4()
            session.execute(text("INSERT INTO interactions (id, agent_id, session_id, full_transcript, context, created_at, updated_at) VALUES (:id, :agent_id, :session_id, :transcript, :context, NOW(), NOW())"),
                {"id": interaction_id, "agent_id": agent_id, "session_id": f"session_{uuid.uuid4()}", "transcript": data["transcript"], "context": json.dumps(data["context"])})
            session.execute(text("INSERT INTO outcomes (id, interaction_id, source, is_success, created_at, updated_at) VALUES (:id, :interaction_id, 'EXPLICIT', :is_success, NOW(), NOW())"),
                {"id": uuid.uuid4(), "interaction_id": interaction_id, "is_success": data["is_success"]})
        session.commit()
        print(f"✅ Seeded {len(UNIQUE_LIVE_DATA)} unique interactions.")
        return agent_id
    finally:
        session.close()

def trigger_live_learning_task_api(token, agent_id):
    print(f"Triggering live learning task for agent {agent_id}...")
    url = f"{BASE_URL}/api/v1/analytics/trigger-live-learning/{agent_id}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(url, headers=headers)
    response.raise_for_status()
    print("✅ Task triggered.")

def main():
    print("--- Kairos UNIQUE Live Learning Engine E2E Test ---")
    token = get_auth_token()
    agent_id = setup_test_data(token)
    trigger_live_learning_task_api(token, agent_id)
    print("\n--- TEST SCRIPT COMPLETE ---")

if __name__ == "__main__":
    main()