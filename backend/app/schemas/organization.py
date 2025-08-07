import uuid
from datetime import datetime
from pydantic import BaseModel

# Base properties
class OrganizationBase(BaseModel):
    name: str

# Properties to receive on creation
class OrganizationCreate(OrganizationBase):
    pass

# Properties to receive on update
class OrganizationUpdate(OrganizationBase):
    pass

# Properties stored in DB
class OrganizationInDB(OrganizationBase):
    id: uuid.UUID
    stripe_customer_id: str | None = None
    subscription_tier: str
    subscription_status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Properties to return to client
class Organization(OrganizationInDB):
    pass