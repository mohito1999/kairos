import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, func, JSON, Numeric
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.models.base import Base

class Interaction(Base):
    __tablename__ = 'interactions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'), nullable=False)
    session_id = Column(String, nullable=False, index=True)
    
    context = Column(JSON, nullable=True)
    context_embedding = Column(Vector(1536), nullable=True)
    full_transcript = Column(Text, nullable=True)
    
    applied_pattern_id = Column(UUID(as_uuid=True), ForeignKey('learned_patterns.id'), nullable=True)
    cost_usd = Column(Numeric(10, 6), nullable=False, default=0.0)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    outcome = relationship("Outcome", back_populates="interaction", uselist=False, cascade="all, delete-orphan")