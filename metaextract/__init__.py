"""
MetaExtract: Adaptive Schema-Guided Extraction System

An agentic AI system that converts unstructured text into structured JSON 
following complex schemas with minimal constraints.

Key Features:
- Multi-agent orchestration
- Adaptive strategy selection based on schema complexity  
- Hierarchical processing for complex schemas
- Large document handling (50-page docs to 10MB files)
- Confidence scoring and validation
"""

__version__ = "0.1.0"
__author__ = "Metaforms AI Assignment"

from .core.schema_analyzer import SchemaComplexityAnalyzer
from .core.strategy_selector import AdaptiveStrategySelector
from .core.orchestrator import MultiAgentOrchestrator
# from .api.main import create_app

__all__ = [
    "SchemaComplexityAnalyzer",
    "AdaptiveStrategySelector", 
    "MultiAgentOrchestrator",
    # "create_app"
] 