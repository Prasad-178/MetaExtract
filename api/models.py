"""
Request and response models for MetaExtract API.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ExtractionRequest(BaseModel):
    """Request for text extraction."""
    text: str
    json_schema: Dict[str, Any] = Field(..., alias="schema")
    strategy: Optional[str] = "auto"
    
    class Config:
        populate_by_name = True


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