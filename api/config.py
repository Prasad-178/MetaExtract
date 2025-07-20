"""
Configuration for MetaExtract API.
"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Info
    api_title: str = "MetaExtract API"
    api_description: str = "Convert unstructured text to structured JSON"
    api_version: str = "1.0.0"
    
    # Server
    host: str = "0.0.0.0"
    port: int = int(os.environ.get("PORT", 8000))
    
    # File uploads
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    upload_dir: str = "uploads"
    
    # Processing
    confidence_threshold: float = 0.7
    processing_timeout: int = 300  # 5 minutes
    
    # OpenAI
    openai_api_key: str = ""
    
    class Config:
        env_file = ".env"


settings = Settings() 