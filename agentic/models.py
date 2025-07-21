"""
Pydantic models for agentic extraction approach
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class AgentRole(Enum):
    """Different agent roles in the extraction process"""
    SCHEMA_ANALYZER = "schema_analyzer"
    TEXT_PROCESSOR = "text_processor" 
    EXTRACTION_SPECIALIST = "extraction_specialist"
    QUALITY_ASSURANCE = "quality_assurance"
    AGGREGATOR = "aggregator"
    KNOWLEDGE_RETRIEVER = "knowledge_retriever"


class AgenticStrategy(Enum):
    """Agentic extraction strategies"""
    PARALLEL = "parallel"           # Run multiple agents in parallel
    HIERARCHICAL = "hierarchical"   # Sequential agent workflow
    COLLABORATIVE = "collaborative" # Agents collaborate and review each other's work
    ADAPTIVE = "adaptive"           # Dynamically choose strategy based on complexity


class AgenticExtractionRequest(BaseModel):
    """Request model for agentic extraction"""
    text: str = Field(..., description="Input text to extract from")
    json_schema: Dict[str, Any] = Field(..., description="JSON schema to extract to")
    strategy: Optional[str] = Field("adaptive", description="Agentic strategy to use")
    enable_rag: bool = Field(False, description="Enable RAG for domain knowledge")
    max_agents: int = Field(5, description="Maximum number of agents to use")
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
    validation_errors: List[str] = []


class AgenticExtractionResult(BaseModel):
    """Result from agentic extraction"""
    success: bool
    final_data: Optional[Dict[str, Any]]
    overall_confidence: float
    agent_results: List[AgentResult]
    consensus_fields: Dict[str, Any] = Field(default_factory=dict)
    conflicting_fields: Dict[str, List[Any]] = Field(default_factory=dict)
    strategy_used: str
    total_processing_time: float
    total_tokens_used: int
    agents_used: int
    validation_errors: List[str] = []
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)


class SchemaSection(BaseModel):
    """A section of a complex schema for agent specialization"""
    section_id: str
    section_name: str
    schema_part: Dict[str, Any]
    priority: int = Field(1, description="Priority level (1-5)")
    complexity_score: float
    agent_assignment: Optional[AgentRole] = None


class AgenticTask(BaseModel):
    """Task assigned to an agent"""
    task_id: str
    agent_role: AgentRole
    description: str
    input_text: str
    schema_section: Optional[SchemaSection] = None
    dependencies: List[str] = Field(default_factory=list)
    tools_available: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict) 