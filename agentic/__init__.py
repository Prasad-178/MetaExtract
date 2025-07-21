"""
Agentic AI approach for MetaExtract

This package implements a simplified multi-agent system for document processing and schema extraction,
using 3 specialized agents for enhanced accuracy and cost efficiency.
"""

from .orchestrator import AgenticMetaExtract
from .models import AgenticExtractionRequest, AgenticExtractionResult

__all__ = ['AgenticMetaExtract', 'AgenticExtractionRequest', 'AgenticExtractionResult'] 