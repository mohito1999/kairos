from fastapi import FastAPI

app = FastAPI(
    title="Kairos API",
    description="The intelligence layer for AI agents.",
    version="0.1.0"
)

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}