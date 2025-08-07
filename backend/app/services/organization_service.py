from sqlalchemy.ext.asyncio import AsyncSession
from app.models.organization import Organization
from app.schemas.organization import OrganizationCreate

async def create_organization(db: AsyncSession, organization_in: OrganizationCreate) -> Organization:
    """
    Creates a new organization in the database.
    """
    new_organization = Organization(name=organization_in.name)
    db.add(new_organization)
    await db.commit()
    await db.refresh(new_organization)
    return new_organization