import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional

# Base properties
class AgentBase(BaseModel):
    name: str
    objective: str
    success_goal_description: Optional[str] = None

# Properties to receive on creation
class AgentCreate(AgentBase):
    pass

# Properties to receive on update
class AgentUpdate(BaseModel):
    name: Optional[str] = None
    objective: Optional[str] = None
    success_goal_description: Optional[str] = None

# Properties stored in DB
class AgentInDB(AgentBase):
    id: uuid.UUID
    organization_id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Properties to return to client
class Agent(AgentInDB):
    pass