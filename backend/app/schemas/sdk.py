import uuid
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

# --- /interactions/start ---
class InteractionStartRequest(BaseModel):
    session_id: str
    context: Optional[Dict[str, Any]] = None
    full_transcript: Optional[str] = None

class InteractionStartResponse(BaseModel):
    interaction_id: uuid.UUID
    strategy_to_inject: str
    pattern_id: Optional[uuid.UUID] = None

# --- /interactions/outcome ---
class InteractionOutcomeRequest(BaseModel):
    interaction_id: uuid.UUID
    metrics: Dict[str, Any]

class InteractionOutcomeResponse(BaseModel):
    status: str = "outcome_recorded"

# --- /context/extract ---
class ContextExtractRequest(BaseModel):
    transcript: str
    schema_definition: Optional[Dict[str, str]] = None

class ContextExtractResponse(BaseModel):
    extracted_context: Dict[str, Any]

# --- /interactions/assess ---
class OutcomeAssessment(BaseModel):
    is_success: bool
    confidence_score: float
    reason: str
    failure_type: Optional[str] = None

class InteractionAssessRequest(BaseModel):
    session_id: str
    transcript: str
    goal: str

class InteractionAssessResponse(OutcomeAssessment):
    pass