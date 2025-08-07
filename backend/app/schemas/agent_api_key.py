import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional

# This schema is never received from a client, only created internally.
class AgentApiKeyCreate(BaseModel):
    agent_id: uuid.UUID
    organization_id: uuid.UUID
    hashed_key: str
    key_prefix: str

# This is the public-facing version of the key, which includes the full, unhashed key
# It is ONLY shown to the user immediately upon creation.
class AgentApiKeyPublic(BaseModel):
    id: uuid.UUID
    agent_id: uuid.UUID
    key_prefix: str
    full_key: str # The one-time visible key
    created_at: datetime

# This is the schema for listing keys, which does NOT include the full key.
class AgentApiKey(BaseModel):
    id: uuid.UUID
    agent_id: uuid.UUID
    key_prefix: str
    is_active: bool
    last_used_at: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True