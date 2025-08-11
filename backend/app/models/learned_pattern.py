# backend/app/models/learned_pattern.py

import uuid
import enum
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, func, JSON, Float, Enum as SQLAlchemyEnum
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from app.models.base import Base

# Define the new, more granular status lifecycle for a pattern
class PatternStatus(enum.Enum):
    CANDIDATE = "CANDIDATE"       # Freshly discovered by the engine, needs validation.
    VALIDATED = "VALIDATED"       # Passed statistical checks against a holdout set.
    REJECTED = "REJECTED"         # Failed statistical checks.
    APPROVED = "APPROVED"         # Manually approved by a human for live testing.
    ACTIVE = "ACTIVE"             # Currently running in the live A/B test engine.
    PAUSED = "PAUSED"             # Manually paused from live A/B testing.
    ARCHIVED = "ARCHIVED"         # Retired, no longer in use but kept for records.

class LearnedPattern(Base):
    __tablename__ = 'learned_patterns'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'), nullable=False)
    source = Column(String, nullable=False) # e.g., HISTORICAL_DISCOVERED, HISTORICAL_GROUPED
    
    trigger_context_summary = Column(Text, nullable=True)
    trigger_embedding = Column(Vector(1536), nullable=True)
    suggested_strategy = Column(Text, nullable=False)
    
    # --- Performance Metrics ---
    impressions = Column(Integer, nullable=False, default=0)
    success_count = Column(Integer, nullable=False, default=0)
    
    # --- NEW: Lifecycle and Validation Fields ---
    status = Column(SQLAlchemyEnum(PatternStatus, name="pattern_status_enum"), nullable=False, default=PatternStatus.CANDIDATE, index=True)
    uplift_score = Column(Float, nullable=True)
    p_value = Column(Float, nullable=True)
    confidence_interval = Column(JSON, nullable=True) # e.g., {"lower": 0.05, "upper": 0.15}
    evidence_payload = Column(JSON, nullable=True) # To store representative transcripts, etc.

    # --- Original Fields ---
    version = Column(Integer, nullable=False, default=1)
    parent_pattern_id = Column(UUID(as_uuid=True), ForeignKey('learned_patterns.id'), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)