"""
Pydantic models for MetaExtract API requests and responses.
"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class ExtractionStrategy(str, Enum):
    """Available extraction strategies."""
    SIMPLE_PROMPT = "simple_prompt"
    ENHANCED_PROMPT = "enhanced_prompt"
    HIERARCHICAL_CHUNKING = "hierarchical_chunking"
    MULTI_AGENT_PARALLEL = "multi_agent_parallel"
    MULTI_AGENT_SEQUENTIAL = "multi_agent_sequential"
    HYBRID = "hybrid"
    AUTO = "auto"  # Let system decide


class InputType(str, Enum):
    """Supported input types."""
    TEXT = "text"
    FILE = "file"
    URL = "url"


class ExtractionRequest(BaseModel):
    """Request model for text extraction."""
    input_text: Optional[str] = Field(None, description="Raw text to extract from")
    input_url: Optional[str] = Field(None, description="URL to extract content from")
    schema: Dict[str, Any] = Field(..., description="JSON schema to extract against")
    strategy: ExtractionStrategy = Field(ExtractionStrategy.AUTO, description="Extraction strategy to use")
    
    # Optional parameters
    chunk_size: Optional[int] = Field(4000, description="Maximum chunk size for document splitting")
    overlap_size: Optional[int] = Field(200, description="Overlap size between chunks")
    confidence_threshold: Optional[float] = Field(0.7, description="Minimum confidence threshold")
    max_agents: Optional[int] = Field(5, description="Maximum number of agents for multi-agent strategies")
    
    class Config:
        json_schema_extra = {
            "example": {
                "input_text": "John Doe is a software engineer with 5 years of experience...",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "profession": {"type": "string"},
                        "experience_years": {"type": "integer"}
                    }
                },
                "strategy": "auto"
            }
        }


class FileExtractionRequest(BaseModel):
    """Request model for file-based extraction."""
    schema: Dict[str, Any] = Field(..., description="JSON schema to extract against")
    strategy: ExtractionStrategy = Field(ExtractionStrategy.AUTO, description="Extraction strategy to use")
    
    # Optional parameters
    chunk_size: Optional[int] = Field(4000, description="Maximum chunk size for document splitting")
    overlap_size: Optional[int] = Field(200, description="Overlap size between chunks")
    confidence_threshold: Optional[float] = Field(0.7, description="Minimum confidence threshold")
    max_agents: Optional[int] = Field(5, description="Maximum number of agents for multi-agent strategies")


class FieldConfidence(BaseModel):
    """Confidence information for a specific field."""
    field_path: str = Field(..., description="JSONPath to the field")
    value: Any = Field(..., description="Extracted value")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors if any")


class ExtractionResult(BaseModel):
    """Response model for extraction results."""
    success: bool = Field(..., description="Whether extraction was successful")
    extracted_data: Optional[Dict[str, Any]] = Field(None, description="Extracted structured data")
    strategy_used: str = Field(..., description="Strategy that was actually used")
    
    # Metadata
    schema_complexity: Dict[str, Any] = Field(..., description="Schema complexity analysis")
    processing_time: float = Field(..., description="Total processing time in seconds")
    chunk_count: int = Field(0, description="Number of chunks processed")
    agent_count: int = Field(1, description="Number of agents used")
    
    # Confidence and validation
    overall_confidence: float = Field(..., description="Overall confidence score")
    field_confidences: List[FieldConfidence] = Field(default_factory=list, description="Per-field confidence scores")
    validation_errors: List[str] = Field(default_factory=list, description="Schema validation errors")
    low_confidence_fields: List[str] = Field(default_factory=list, description="Fields flagged for human review")
    
    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if extraction failed")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")


class SchemaAnalysisRequest(BaseModel):
    """Request model for schema analysis."""
    schema: Dict[str, Any] = Field(..., description="JSON schema to analyze")


class SchemaAnalysisResult(BaseModel):
    """Response model for schema analysis."""
    complexity_metrics: Dict[str, Any] = Field(..., description="Detailed complexity metrics")
    recommended_strategy: str = Field(..., description="Recommended extraction strategy")
    estimated_processing_time: float = Field(..., description="Estimated processing time in seconds")
    resource_requirements: Dict[str, Any] = Field(..., description="Estimated resource requirements")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Component health status")


class ExtractionStatus(BaseModel):
    """Status of an async extraction job."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status: pending, processing, completed, failed")
    progress: float = Field(..., description="Progress percentage (0.0-1.0)")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")
    result: Optional[ExtractionResult] = Field(None, description="Result if completed")
    created_at: str = Field(..., description="Job creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp") 