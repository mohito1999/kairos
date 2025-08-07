import uuid
from sqlalchemy import Column, Text, Boolean, DateTime, ForeignKey, func, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.models.base import Base

class HistoricalInteraction(Base):
    __tablename__ = 'historical_interactions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    upload_id = Column(UUID(as_uuid=True), ForeignKey('historical_uploads.id'), nullable=False)
    
    original_context = Column(JSON, nullable=True)
    original_response = Column(Text, nullable=True)
    extracted_outcome = Column(JSON, nullable=True)
    is_success = Column(Boolean, nullable=False)

    contributed_to_pattern_id = Column(UUID(as_uuid=True), ForeignKey('learned_patterns.id'), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    upload = relationship("HistoricalUpload", back_populates="interactions")