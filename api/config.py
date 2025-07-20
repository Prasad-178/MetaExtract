"""
Configuration settings for MetaExtract API.
"""
import os
from typing import Dict, Any
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    api_title: str = "MetaExtract API"
    api_description: str = "Advanced system for converting unstructured text to structured JSON using complex schemas"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # File Upload Configuration
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: list = [".txt", ".md", ".csv", ".json", ".pdf", ".docx"]
    upload_dir: str = "uploads"
    
    # Processing Configuration
    default_chunk_size: int = 4000
    default_overlap_size: int = 200
    default_confidence_threshold: float = 0.7
    max_agents: int = 10
    processing_timeout: int = 300  # 5 minutes
    
    # LLM Configuration (for future real LLM integration)
    llm_provider: str = "openai"  # openai, anthropic, etc.
    llm_model: str = "gpt-4"
    llm_api_key: str = ""
    llm_max_tokens: int = 4000
    llm_temperature: float = 0.1
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    
    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    # Security
    cors_origins: list = ["*"]  # In production, specify exact origins
    cors_credentials: bool = True
    cors_methods: list = ["*"]
    cors_headers: list = ["*"]
    
    class Config:
        env_file = ".env"
        env_prefix = "METAEXTRACT_"


# Global settings instance
settings = Settings()


def get_strategy_config() -> Dict[str, Any]:
    """Get default strategy configuration."""
    return {
        "simple_prompt": {
            "max_input_size": 2000,
            "recommended_for": "Simple schemas with few fields"
        },
        "enhanced_prompt": {
            "max_input_size": 4000,
            "recommended_for": "Moderate complexity schemas"
        },
        "hierarchical_chunking": {
            "max_input_size": 50000,
            "recommended_for": "Large documents with moderate schema complexity"
        },
        "multi_agent_parallel": {
            "max_input_size": 100000,
            "recommended_for": "Complex schemas with independent sections"
        },
        "multi_agent_sequential": {
            "max_input_size": 100000,
            "recommended_for": "Complex schemas with dependencies"
        },
        "hybrid": {
            "max_input_size": 200000,
            "recommended_for": "Very complex schemas and large documents"
        }
    }


def get_complexity_thresholds() -> Dict[str, Dict[str, Any]]:
    """Get schema complexity thresholds for strategy selection."""
    return {
        "low": {
            "max_nesting_depth": 3,
            "max_objects": 10,
            "max_enum_size": 5,
            "strategies": ["simple_prompt", "enhanced_prompt"]
        },
        "medium": {
            "max_nesting_depth": 5,
            "max_objects": 50,
            "max_enum_size": 20,
            "strategies": ["enhanced_prompt", "hierarchical_chunking"]
        },
        "high": {
            "max_nesting_depth": 7,
            "max_objects": 150,
            "max_enum_size": 50,
            "strategies": ["hierarchical_chunking", "multi_agent_parallel", "multi_agent_sequential"]
        },
        "very_high": {
            "max_nesting_depth": float('inf'),
            "max_objects": float('inf'),
            "max_enum_size": float('inf'),
            "strategies": ["hybrid"]
        }
    } 