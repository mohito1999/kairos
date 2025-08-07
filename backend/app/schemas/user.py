import uuid
from datetime import datetime
from pydantic import BaseModel, EmailStr, ConfigDict

# Base properties
class UserBase(BaseModel):
    email: EmailStr
    full_name: str | None = None

# Properties to receive on creation
class UserCreate(UserBase):
    supabase_auth_id: uuid.UUID
    organization_id: uuid.UUID
    role: str = "member"

# Properties to receive on update
class UserUpdate(BaseModel):
    full_name: str | None = None
    role: str | None = None

# Properties stored in DB
class UserInDB(UserBase):
    id: uuid.UUID
    supabase_auth_id: uuid.UUID
    organization_id: uuid.UUID
    role: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

# Properties to return to client
class User(UserInDB):
    pass