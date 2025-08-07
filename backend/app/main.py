from fastapi import FastAPI
from app.api.v1.endpoints import users

app = FastAPI(
    title="Kairos API",
    description="The intelligence layer for AI agents.",
    version="0.1.0"
)

app.include_router(users.router, prefix="/api/v1/users", tags=["Users"]) # Add this line

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}
