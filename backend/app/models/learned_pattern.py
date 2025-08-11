import uuid
import enum
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, func, JSON, Float, Enum as SQLAlchemyEnum
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from app.models.base import Base

class PatternStatus(enum.Enum):
    CANDIDATE = "CANDIDATE"; VALIDATED = "VALIDATED"; REJECTED = "REJECTED"
    APPROVED = "APPROVED"; ACTIVE = "ACTIVE"; PAUSED = "PAUSED"; ARCHIVED = "ARCHIVED"

class LearnedPattern(Base):
    __tablename__ = 'learned_patterns'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'), nullable=False)
    source = Column(String, nullable=False)
    # --- NEW: Explicit link to the source upload ---
    source_upload_id = Column(UUID(as_uuid=True), ForeignKey('historical_uploads.id'), nullable=True, index=True)
    
    trigger_context_summary = Column(Text, nullable=True)
    trigger_embedding = Column(Vector(1536), nullable=True)
    suggested_strategy = Column(Text, nullable=False)
    
    impressions = Column(Integer, nullable=False, default=0)
    success_count = Column(Integer, nullable=False, default=0)
    
    status = Column(SQLAlchemyEnum(PatternStatus, name="pattern_status_enum", create_type=False), nullable=False, default=PatternStatus.CANDIDATE, index=True)
    uplift_score = Column(Float, nullable=True)
    p_value = Column(Float, nullable=True)
    confidence_interval = Column(JSON, nullable=True)
    evidence_payload = Column(JSON, nullable=True)

    version = Column(Integer, nullable=False, default=1)
    parent_pattern_id = Column(UUID(as_uuid=True), ForeignKey('learned_patterns.id'), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)