import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.models.base import Base

class Agent(Base):
    __tablename__ = 'agents'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'), nullable=False)
    
    objective = Column(String, nullable=False)
    success_goal_description = Column(Text, nullable=True)
    success_goal_embedding = Column(Vector(1536), nullable=True) # OpenAI text-embedding-3-small
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    organization = relationship("Organization", back_populates="agents")
    api_keys = relationship("AgentApiKey", back_populates="agent", cascade="all, delete-orphan")