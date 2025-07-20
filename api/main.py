"""
MetaExtract API - Main application.
"""
import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import settings
from .routes import router

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")

# Create upload directory
os.makedirs(settings.upload_dir, exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "MetaExtract API",
        "version": settings.api_version,
        "docs": "/docs"
    } 