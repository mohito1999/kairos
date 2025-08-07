import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from app.models.base import Base

class Experiment(Base):
    __tablename__ = 'experiments'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'), nullable=False)
    name = Column(String, nullable=False)
    
    control_pattern_id = Column(UUID(as_uuid=True), ForeignKey('learned_patterns.id'), nullable=False)
    challenger_pattern_id = Column(UUID(as_uuid=True), ForeignKey('learned_patterns.id'), nullable=False)
    
    status = Column(String, nullable=False, default='RUNNING') # e.g., RUNNING, CONCLUDED_...
    start_time = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)