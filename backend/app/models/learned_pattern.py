import uuid
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, func, JSON, Float
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from app.models.base import Base

class LearnedPattern(Base):
    __tablename__ = 'learned_patterns'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'), nullable=False)
    source = Column(String, nullable=False) # e.g., HISTORICAL_CONTRASTIVE, LIVE_DISCOVERED, MANUAL
    
    # --- NEW: Fields for Contrastive Analysis ---
    battleground_context = Column(JSON, nullable=True) # The specific situation, e.g., {"objection_type": "price"}
    positive_examples = Column(JSON, nullable=True) # Snippets of successful transcripts
    negative_examples = Column(JSON, nullable=True) # Snippets of failed transcripts
    uplift_score = Column(Float, nullable=True) # The measured performance lift from validation
    p_value = Column(Float, nullable=True) # The statistical significance of the uplift
    
    # --- Existing fields ---
    trigger_context_summary = Column(Text, nullable=True)
    trigger_embedding = Column(Vector(1536), nullable=True)
    suggested_strategy = Column(Text, nullable=False)
    
    impressions = Column(Integer, nullable=False, default=0)
    success_count = Column(Integer, nullable=False, default=0)
    
    status = Column(String, nullable=False, default='CANDIDATE', index=True) # CANDIDATE, VALIDATED, ACTIVE, PAUSED
    version = Column(Integer, nullable=False, default=1)
    
    parent_pattern_id = Column(UUID(as_uuid=True), ForeignKey('learned_patterns.id'), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)