from sqlalchemy.orm import declarative_base

Base = declarative_base()

# Import all models here so that Alembic and SQLAlchemy know about them.
from app.models.organization import Organization
from app.models.user import User
from app.models.agent import Agent
from app.models.agent_api_key import AgentApiKey
from app.models.interaction import Interaction
from app.models.outcome import Outcome
from app.models.learned_pattern import LearnedPattern
from app.models.experiment import Experiment
from app.models.historical_upload import HistoricalUpload
from app.models.historical_interaction import HistoricalInteraction
from app.models.suggested_opportunity import SuggestedOpportunity