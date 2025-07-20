"""
Request and response models for MetaExtract API.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class ExtractionRequest(BaseModel):
    """Request for text extraction."""
    text: str
    schema: Dict[str, Any]
    strategy: Optional[str] = "auto"


class ExtractionResult(BaseModel):
    """Result of extraction."""
    data: Dict[str, Any]
    confidence: float
    field_confidences: Dict[str, float]
    low_confidence_fields: List[str]
    processing_time: float
    strategy_used: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str 