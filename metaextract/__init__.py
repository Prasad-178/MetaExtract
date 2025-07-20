"""
MetaExtract: Simplified Schema-Guided Extraction System

A focused AI system that converts unstructured text into structured JSON 
following complex schemas with minimal constraints.

Key Features:
- Intelligent schema complexity analysis
- Adaptive strategy selection (simple/chunked/hierarchical)
- Real OpenAI GPT-4 integration
- Large document handling (50-page docs to 10MB files)
- Confidence scoring and validation
"""

__version__ = "0.2.0"
__author__ = "Metaforms AI Assignment - Simplified"

from .simplified_extractor import SimplifiedMetaExtract, SchemaComplexity, ExtractionStrategy, ExtractionResult

__all__ = [
    "SimplifiedMetaExtract",
    "SchemaComplexity", 
    "ExtractionStrategy",
    "ExtractionResult"
] 