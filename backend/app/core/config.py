from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Database
    RAW_DATABASE_URL: str
    
    # Supabase
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    
    # AI Services
    OPENROUTER_API_KEY: str
    OPENAI_API_KEY: str
    
    # JWT
    JWT_SECRET_KEY: str
    
    @property
    def DATABASE_URL(self) -> str:
        # SQLAlchemy 2.0 requires the asyncpg driver for async operations
        if self.RAW_DATABASE_URL.startswith("postgresql://"):
            return self.RAW_DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
        return self.RAW_DATABASE_URL

    @property
    def SYNC_DATABASE_URL(self) -> str:
        # Alembic needs a synchronous driver
        if self.RAW_DATABASE_URL.startswith("postgresql://"):
            return self.RAW_DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
        return self.RAW_DATABASE_URL

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()