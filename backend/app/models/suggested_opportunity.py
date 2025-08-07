import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, func, JSON
from sqlalchemy.dialects.postgresql import UUID
from app.models.base import Base

class SuggestedOpportunity(Base):
    __tablename__ = 'suggested_opportunities'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'), nullable=False)
    
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    suggested_action = Column(Text, nullable=False)
    impact_forecast = Column(JSON, nullable=True)
    
    source = Column(String, nullable=False) # e.g., LATENT_NEED, LOGICAL_EXTRAPOLATION
    status = Column(String, nullable=False, default='NEW', index=True) # e.g., NEW, REVIEWED, IMPLEMENTED
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)