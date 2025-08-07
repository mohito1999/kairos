import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

from app.models.organization import Organization
from app.models.user import User
from app.main import app
from app.core.dependencies import get_current_user_with_provisioining

@pytest.mark.asyncio
async def test_create_agent(test_client: AsyncClient, db_session: AsyncSession):
    """
    Test creating an agent with proper session management.
    """
    # --- SETUP: Create prerequisite data using the SAME session ---
    test_org = Organization(name="Test Org Inc.")
    db_session.add(test_org)
    await db_session.commit()
    await db_session.refresh(test_org)
    
    # Supabase user IDs are UUIDs, so our model should handle them.
    # We will ensure our model is robust enough for this.
    test_user_supabase_id = uuid.uuid4()
    
    test_user = User(
        supabase_auth_id=test_user_supabase_id,
        organization_id=test_org.id,
        email="test@testorg.com",
        role="owner"
    )
    db_session.add(test_user)
    await db_session.commit()
    await db_session.refresh(test_user)

    # --- MOCK: Override auth dependency ---
    async def override_get_user():
        return test_user

    app.dependency_overrides[get_current_user_with_provisioining] = override_get_user

    # --- EXECUTE: Make the API call ---
    agent_data = {
        "name": "Test Sales Agent",
        "objective": "To book as many demos as possible."
    }
    
    response = await test_client.post("/api/v1/agents/", json=agent_data)

    # --- ASSERT: Check the results ---
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == agent_data["name"]
    assert data["organization_id"] == str(test_org.id)