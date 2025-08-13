import uuid
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, func, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.models.base import Base

class HistoricalUpload(Base):
    __tablename__ = 'historical_uploads'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'), nullable=False)

    filename = Column(String, nullable=False)
    total_interactions = Column(Integer, nullable=True)
    processed_interactions = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False) # e.g., UPLOADING, PROCESSING, COMPLETED, FAILED

    interaction_id_split = Column(JSON, nullable=True)

    upload_timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    processing_completed_timestamp = Column(DateTime(timezone=True), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    interactions = relationship("HistoricalInteraction", back_populates="upload", cascade="all, delete-orphan")