"""
Pydantic models for agentic extraction approach
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class AgentRole(Enum):
    """Different agent roles in the extraction process"""
    SCHEMA_ANALYZER = "schema_analyzer"
    EXTRACTION_SPECIALIST = "extraction_specialist"
    QUALITY_ASSURANCE = "quality_assurance"


class AgenticStrategy(Enum):
    """Agentic extraction strategies"""
    ENHANCED_AGENTIC = "enhanced_agentic"


class AgenticExtractionRequest(BaseModel):
    """Request model for agentic extraction"""
    text: str = Field(..., description="Input text to extract from")
    json_schema: Dict[str, Any] = Field(..., description="JSON schema to extract to")
    strategy: Optional[str] = Field("enhanced_agentic", description="Agentic strategy to use")
    enable_rag: bool = Field(False, description="Enable RAG for domain knowledge")
    max_agents: int = Field(3, description="Maximum number of agents to use")
    confidence_threshold: float = Field(0.7, description="Minimum confidence threshold")
    enable_cross_validation: bool = Field(True, description="Enable agent cross-validation")


class AgentResult(BaseModel):
    """Result from a single agent"""
    agent_id: str
    agent_role: AgentRole
    extracted_data: Dict[str, Any]
    confidence: float
    reasoning: str
    processing_time: float
    tokens_used: int
    field_confidences: Dict[str, float]
    validation_errors: List[str] = Field(default_factory=list)


class AgenticExtractionResult(BaseModel):
    """Result from agentic extraction"""
    success: bool
    final_data: Optional[Dict[str, Any]]
    overall_confidence: float
    agent_results: List[AgentResult]
    consensus_fields: Dict[str, Any] = Field(default_factory=dict)
    conflicting_fields: Dict[str, List[Any]] = Field(default_factory=dict)
    low_confidence_fields: List[str] = Field(default_factory=list)
    strategy_used: str
    total_processing_time: float
    total_tokens_used: int
    agents_used: int
    validation_errors: List[str] = Field(default_factory=list)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict) 