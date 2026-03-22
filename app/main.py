from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from app.api import routes

app = FastAPI(title="AI Study Assistant", version="1.0.0")

# Enable CORS for frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router)

@app.get("/")
async def serve_ui():
    """Serves the visually dynamic frontend application."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(index_path)

@app.get("/health")
async def health_check():
    """Returns the health status of the API."""
    return {"status": "ok", "version": "1.0.0"}
