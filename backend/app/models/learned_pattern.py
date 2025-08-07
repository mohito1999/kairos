import uuid
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from app.models.base import Base

class LearnedPattern(Base):
    __tablename__ = 'learned_patterns'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'), nullable=False)
    source = Column(String, nullable=False) # e.g., LIVE_DISCOVERED, HISTORICAL, MANUAL
    
    trigger_context_summary = Column(Text, nullable=True)
    trigger_embedding = Column(Vector(1536), nullable=True)
    suggested_strategy = Column(Text, nullable=False)
    
    impressions = Column(Integer, nullable=False, default=0)
    success_count = Column(Integer, nullable=False, default=0)
    
    status = Column(String, nullable=False, default='CANDIDATE', index=True) # e.g., CANDIDATE, ACTIVE, PAUSED
    version = Column(Integer, nullable=False, default=1)
    
    parent_pattern_id = Column(UUID(as_uuid=True), ForeignKey('learned_patterns.id'), nullable=True)
    historical_evidence_count = Column(Integer, nullable=False, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)