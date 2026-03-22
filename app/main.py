from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="AI Study Assistant",
    description="Offline AI system for OCR, text cleaning, and AI-powered study help.",
    version="0.1.0",
)

app.include_router(router)


@app.get("/health")
def health_check():
    """Simple health check to confirm the server is running."""
    return {"status": "ok", "version": "0.1.0"}
