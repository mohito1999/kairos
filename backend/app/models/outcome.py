import uuid
from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, func, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.models.base import Base

class Outcome(Base):
    __tablename__ = 'outcomes'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    interaction_id = Column(UUID(as_uuid=True), ForeignKey('interactions.id'), unique=True, nullable=False)
    
    source = Column(String, nullable=False) # e.g., EXPLICIT, IMPLICIT, AI_ASSISTED
    metrics = Column(JSON, nullable=True)
    is_success = Column(Boolean, nullable=False, index=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    interaction = relationship("Interaction", back_populates="outcome")