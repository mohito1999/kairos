import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.schemas.user import UserCreate
from app.schemas.organization import OrganizationCreate
from .organization_service import create_organization

from app.models.user import User

async def get_user_by_supabase_id(db: AsyncSession, supabase_id: uuid.UUID) -> User | None:
    """
    Fetches a user from our database using their Supabase Auth ID.
    """
    result = await db.execute(select(User).filter(User.supabase_auth_id == supabase_id))
    return result.scalars().first()

async def create_user_with_organization(db: AsyncSession, user_in: UserCreate) -> User:
    """
    Creates a new user and their initial organization.
    The organization_id is handled internally, not passed in.
    """
    # Create an organization for the new user.
    org_name = f"{user_in.email.split('@')[0]}'s Organization"
    new_organization = await create_organization(db, OrganizationCreate(name=org_name))
    
    # Create the user and link them to the new organization
    new_user = User(
        email=user_in.email,
        supabase_auth_id=user_in.supabase_auth_id,
        organization_id=new_organization.id, # The ID comes from the new org
        full_name=user_in.full_name,
        role=user_in.role
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user
