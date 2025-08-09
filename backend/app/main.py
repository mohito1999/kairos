from fastapi import FastAPI
from app.api.v1.endpoints import users, agents, api_keys, sdk, historical_data, analytics, human_interactions

app = FastAPI(
    title="Kairos API",
    description="The intelligence layer for AI agents.",
    version="0.1.0"
)

app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["Agents"])
app.include_router(api_keys.router, prefix="/api/v1/api-keys", tags=["API Keys"])
app.include_router(historical_data.router, prefix="/api/v1/historical-data", tags=["Historical Data"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(human_interactions.router, prefix="/api/v1/human-interactions", tags=["Human Interactions"])

# Public, SDK-facing APIs
app.include_router(sdk.router, prefix="/sdk/v1", tags=["SDK"])


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}
