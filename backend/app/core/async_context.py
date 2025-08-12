# backend/app/core/async_context.py (Revised)
from contextvars import ContextVar
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from openai import AsyncOpenAI
from app.core.config import settings

_async_state_var: ContextVar = ContextVar("async_state", default=None)

class AsyncContext:
    """A container for lazily initialized async resources."""
    def __init__(self):
        self._engine = None
        self._session_factory = None
        self._openai_client = None
        self._openrouter_client = None # Add new client holder

    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
        return self._engine

    @property
    def session_factory(self):
        if self._session_factory is None:
            self._session_factory = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)
        return self._session_factory

    @property
    def openai_client(self):
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        return self._openai_client

    # --- NEW PROPERTY FOR OPENROUTER ---
    @property
    def openrouter_client(self):
        if self._openrouter_client is None:
            self._openrouter_client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.OPENROUTER_API_KEY
            )
        return self._openrouter_client

    async def close(self):
        """Gracefully close all open connections."""
        if self._openai_client:
            await self._openai_client.close()
        if self._openrouter_client: # Add cleanup for new client
            await self._openrouter_client.close()
        if self._engine:
            await self._engine.dispose()
        # Reset all
        self._engine = None; self._session_factory = None
        self._openai_client = None; self._openrouter_client = None

def get_async_context() -> AsyncContext:
    if (ctx := _async_state_var.get()) is None:
        ctx = AsyncContext()
        _async_state_var.set(ctx)
    return ctx

async def close_async_context():
    if (ctx := _async_state_var.get()) is not None:
        await ctx.close()
        _async_state_var.set(None)