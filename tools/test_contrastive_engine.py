import requests
import time
import os
import json

# --- CONFIGURATION ---
BASE_URL = "http://localhost:8000"
# Use the credentials for the pre-seeded test user in your Supabase project
# You can find this in your Supabase dashboard under Authentication -> Users
TEST_USER_EMAIL = "kairos.dev123@gmail.com"
TEST_USER_PASSWORD = "6sK5uUespO5Mqx" # Replace if you changed the default
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'processed_insurance_calls.csv')

# --- DATA MAPPING FOR THE INSURANCE DATASET ---
DATA_MAPPING = {
    "conversation_transcript": "transcript",
    "outcome_column": "call_outcome",
    "context_initial_objection": "initial_objection",
    "context_data_correction_needed": "data_correction_needed",
    "context_card_retrieval_hesitation": "card_retrieval_hesitation"
}

def get_auth_token():
    """Authenticates with Supabase to get a JWT."""
    print("Authenticating with Supabase to get user token...")
    url = f"{os.getenv('NEXT_PUBLIC_SUPABASE_URL')}/auth/v1/token?grant_type=password"
    headers = {"apikey": os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')}
    payload = {"email": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        token = response.json().get("access_token")
        if not token:
            raise Exception("Access token not found in response.")
        print("✅ Authentication successful.")
        return token
    except requests.exceptions.RequestException as e:
        print(f"❌ Authentication failed: {e}")
        print("Please ensure the test user exists in Supabase with the correct credentials.")
        exit(1)

def create_test_agent(token):
    """Creates a new agent to associate the upload with."""
    print("Creating a new test agent...")
    url = f"{BASE_URL}/api/v1/agents/"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "name": f"Insurance Call Analyzer {int(time.time())}",
        "objective": "To determine the outcome of Medicare insurance calls and find winning strategies.",
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

def upload_data(token, agent_id):
    """Uploads the CSV and triggers the pipeline."""
    print(f"Uploading '{os.path.basename(CSV_FILE_PATH)}' for processing...")
    url = f"{BASE_URL}/api/v1/historical-data/upload"
    headers = {"Authorization": f"Bearer {token}"}
    
    files = {'file': (os.path.basename(CSV_FILE_PATH), open(CSV_FILE_PATH, 'rb'), 'text/csv')}
    data = {
        'agent_id': agent_id,
        'data_mapping': json.dumps(DATA_MAPPING)
    }

    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()
        upload_job = response.json()
        print(f"✅ File upload accepted. Upload ID: {upload_job['upload_id']}")
        print("The backend is now processing the data asynchronously via Celery.")
        print("You can monitor the logs of the 'kairos_worker' and 'kairos_api' containers for progress.")
    except requests.exceptions.RequestException as e:
        print(f"❌ File upload failed: {e.response.text}")
        exit(1)

def main():
    """Main execution function."""
    # We need the Supabase variables from the frontend .env file
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', '.env.local')
    load_dotenv(dotenv_path=dotenv_path)

    if not os.path.exists(CSV_FILE_PATH):
        print(f"❌ Error: CSV file not found at {CSV_FILE_PATH}")
        return

    # Ensure Docker services are running
    print("--- Kairos Contrastive Engine E2E Test ---")
    print("Ensuring Docker services are running...")
    if os.system("docker-compose -f ../backend/docker-compose.yml ps | grep 'kairos_api' | grep 'Up'") != 0:
        print("❌ Docker services not running. Please run 'docker-compose up -d' in the 'backend' directory first.")
        return
        
    token = get_auth_token()
    agent = create_test_agent(token)
    upload_data(token, agent['id'])
    
    print("\n--- TEST SCRIPT COMPLETE ---")
    print("The processing pipeline has been started. Check your Docker logs for detailed output from the Celery workers.")

if __name__ == "__main__":
    main()