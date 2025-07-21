"""
Agentic AI approach for MetaExtract

This package implements a multi-agent system for document processing and schema extraction,
leveraging existing MetaExtract functions as tools for specialized agents.
"""

from .orchestrator import AgenticMetaExtract
from .models import AgenticExtractionRequest, AgenticExtractionResult

__all__ = ['AgenticMetaExtract', 'AgenticExtractionRequest', 'AgenticExtractionResult'] 