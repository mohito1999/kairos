import uuid
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, func, JSON, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.models.base import Base

class LearnedPattern(Base):
    __tablename__ = 'learned_patterns'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'), nullable=False)
    source = Column(String, nullable=False)
    source_upload_id = Column(UUID(as_uuid=True), ForeignKey('historical_uploads.id'), nullable=True)

    # --- Fields for Contrastive Analysis ---
    battleground_context = Column(JSON, nullable=True)
    positive_examples = Column(JSON, nullable=True)
    negative_examples = Column(JSON, nullable=True)
    uplift_score = Column(Float, nullable=True)
    p_value = Column(Float, nullable=True)
    
    # --- Core Strategy & Trigger Fields ---
    trigger_context_summary = Column(Text, nullable=True)
    trigger_embedding = Column(Vector(1536), nullable=True)
    trigger_threshold = Column(Float, nullable=True)
    
    suggested_strategy = Column(Text, nullable=False)
    strategy_embedding = Column(Vector(1536), nullable=True)
    strategy_threshold = Column(Float, nullable=True)
    
    # --- Performance Tracking ---
    impressions = Column(Integer, nullable=False, default=0)
    success_count = Column(Integer, nullable=False, default=0)
    
    # --- Metadata ---
    status = Column(String, nullable=False, default='CANDIDATE', index=True)
    version = Column(Integer, nullable=False, default=1)
    parent_pattern_id = Column(UUID(as_uuid=True), ForeignKey('learned_patterns.id'), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)