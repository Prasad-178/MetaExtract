"""
Agent tools that wrap existing MetaExtract functionality
"""

import json
import sys
import os
from typing import Dict, Any, List, Optional, Tuple
from crewai.tools import BaseTool
from pydantic import Field

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from metaextract.simplified_extractor import SimplifiedMetaExtract, SchemaComplexity, ExtractionStrategy


class SchemaAnalyzerTool(BaseTool):
    """Tool for analyzing schema complexity"""
    name: str = "schema_analyzer"
    description: str = "Analyze JSON schema complexity and recommend processing strategies"
    
    def _run(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze schema complexity using existing MetaExtract logic"""
        try:
            # Create a temporary extractor instance to use analysis methods
            temp_extractor = SimplifiedMetaExtract("dummy-key")
            complexity = temp_extractor._analyze_schema_complexity(schema)
            
            return {
                "complexity_score": complexity.complexity_score,
                "nesting_depth": complexity.nesting_depth,
                "total_objects": complexity.total_objects,
                "total_properties": complexity.total_properties,
                "has_complex_types": complexity.has_complex_types,
                "estimated_tokens": complexity.estimated_tokens,
                "is_complex": complexity.is_complex,
                "recommended_strategy": "hierarchical" if complexity.is_complex else "simple"
            }
        except Exception as e:
            return {"error": str(e), "complexity_score": 0}


class TextChunkerTool(BaseTool):
    """Tool for chunking large text documents"""
    name: str = "text_chunker"
    description: str = "Split large text into manageable chunks with overlap"
    
    def _run(self, text: str, max_chunk_size: int = 8000, overlap: int = 200) -> List[str]:
        """Chunk text using existing MetaExtract logic"""
        try:
            temp_extractor = SimplifiedMetaExtract("dummy-key")
            temp_extractor.chunk_overlap = overlap
            chunks = temp_extractor._chunk_text(text, max_chunk_size)
            return chunks
        except Exception as e:
            return [text]  # Return original text if chunking fails


class SchemaSectionerTool(BaseTool):
    """Tool for breaking complex schemas into sections"""
    name: str = "schema_sectioner"
    description: str = "Break complex schemas into manageable sections for parallel processing"
    
    def _run(self, schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Break schema into sections using existing MetaExtract logic"""
        try:
            temp_extractor = SimplifiedMetaExtract("dummy-key")
            sections = temp_extractor._break_schema_into_sections(schema)
            return sections
        except Exception as e:
            return {"main": schema}


class DataValidatorTool(BaseTool):
    """Tool for validating extracted data against schema"""
    name: str = "data_validator"
    description: str = "Validate extracted data against JSON schema and calculate confidence scores"
    
    def _run(self, data: Dict[str, Any], schema: Dict[str, Any], input_text: str = "") -> Dict[str, Any]:
        """Validate data using existing MetaExtract logic"""
        try:
            temp_extractor = SimplifiedMetaExtract("dummy-key")
            
            # Validate JSON schema
            validation_errors = temp_extractor._validate_json_schema(data, schema)
            
            # Calculate field confidences
            field_confidences = temp_extractor._calculate_field_confidences(data, schema, input_text)
            
            # Calculate overall confidence
            overall_confidence = sum(field_confidences.values()) / len(field_confidences) if field_confidences else 0.0
            
            # Identify low confidence fields
            low_confidence_fields = [
                field for field, confidence in field_confidences.items() 
                if confidence < 0.6
            ]
            
            return {
                "is_valid": len(validation_errors) == 0,
                "validation_errors": validation_errors,
                "overall_confidence": overall_confidence,
                "field_confidences": field_confidences,
                "low_confidence_fields": low_confidence_fields
            }
        except Exception as e:
            return {
                "is_valid": False,
                "validation_errors": [str(e)],
                "overall_confidence": 0.0,
                "field_confidences": {},
                "low_confidence_fields": []
            }


class DataMergerTool(BaseTool):
    """Tool for merging extraction results from multiple sources"""
    name: str = "data_merger"
    description: str = "Merge multiple extraction results into a single coherent result"
    
    def _run(self, results: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple results using existing MetaExtract logic"""
        try:
            temp_extractor = SimplifiedMetaExtract("dummy-key")
            merged_data = temp_extractor._merge_chunk_results(results, schema)
            return merged_data
        except Exception as e:
            # Fallback merge logic
            merged = {}
            for result in results:
                if isinstance(result, dict):
                    for key, value in result.items():
                        if key not in merged:
                            merged[key] = value
                        elif isinstance(value, list) and isinstance(merged[key], list):
                            merged[key].extend(value)
                        elif value and not merged[key]:
                            merged[key] = value
            return merged


class KnowledgeRetrievalTool(BaseTool):
    """Tool for retrieving domain-specific knowledge to improve extraction"""
    name: str = "knowledge_retrieval"
    description: str = "Retrieve relevant domain knowledge to improve extraction accuracy"
    
    def _run(self, query: str, schema_context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve domain knowledge (simplified implementation)"""
        # This is a placeholder for RAG implementation
        # In a full implementation, this would query a vector database
        # with domain-specific knowledge about extraction patterns
        
        knowledge_base = {
            "person": {
                "common_fields": ["name", "email", "phone", "address"],
                "patterns": ["full name format", "email validation", "phone formats"]
            },
            "organization": {
                "common_fields": ["name", "address", "website", "industry"],
                "patterns": ["company naming conventions", "industry classifications"]
            },
            "document": {
                "common_fields": ["title", "author", "date", "content"],
                "patterns": ["citation formats", "date patterns", "document structure"]
            }
        }
        
        # Simple keyword matching for demo
        relevant_knowledge = {}
        query_lower = query.lower()
        
        for domain, info in knowledge_base.items():
            if domain in query_lower:
                relevant_knowledge[domain] = info
        
        return {
            "relevant_knowledge": relevant_knowledge,
            "suggestions": [
                "Consider using common field patterns for better extraction",
                "Validate extracted data against domain-specific rules",
                "Cross-reference with typical document structures"
            ]
        }


class ConfidenceScorerTool(BaseTool):
    """Tool for calculating detailed confidence scores"""
    name: str = "confidence_scorer"
    description: str = "Calculate comprehensive confidence scores for extracted data"
    
    def _run(self, 
             extracted_data: Dict[str, Any], 
             source_text: str,
             schema: Dict[str, Any],
             agent_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate advanced confidence metrics"""
        
        # Basic confidence using existing logic
        temp_extractor = SimplifiedMetaExtract("dummy-key")
        field_confidences = temp_extractor._calculate_field_confidences(extracted_data, schema, source_text)
        
        # Enhanced confidence calculation considering agent consensus
        if agent_results and len(agent_results) > 1:
            consensus_bonus = self._calculate_consensus_bonus(extracted_data, agent_results)
            for field in field_confidences:
                field_confidences[field] = min(1.0, field_confidences[field] + consensus_bonus)
        
        overall_confidence = sum(field_confidences.values()) / len(field_confidences) if field_confidences else 0.0
        
        return {
            "overall_confidence": overall_confidence,
            "field_confidences": field_confidences,
            "confidence_breakdown": {
                "text_match_score": self._calculate_text_match_score(extracted_data, source_text),
                "schema_compliance_score": self._calculate_schema_compliance(extracted_data, schema),
                "consensus_score": self._calculate_consensus_bonus(extracted_data, agent_results) if agent_results else 0.0
            }
        }
    
    def _calculate_text_match_score(self, data: Dict[str, Any], text: str) -> float:
        """Calculate how well extracted data matches source text"""
        if not data:
            return 0.0
        
        matches = 0
        total = 0
        text_lower = text.lower()
        
        def check_matches(obj, path=""):
            nonlocal matches, total
            if isinstance(obj, dict):
                for key, value in obj.items():
                    check_matches(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for item in obj:
                    check_matches(item, path)
            elif isinstance(obj, str) and obj:
                total += 1
                if obj.lower() in text_lower:
                    matches += 1
        
        check_matches(data)
        return matches / total if total > 0 else 0.0
    
    def _calculate_schema_compliance(self, data: Dict[str, Any], schema: Dict[str, Any]) -> float:
        """Calculate schema compliance score"""
        temp_extractor = SimplifiedMetaExtract("dummy-key")
        errors = temp_extractor._validate_json_schema(data, schema)
        return max(0.0, 1.0 - len(errors) * 0.1)
    
    def _calculate_consensus_bonus(self, data: Dict[str, Any], agent_results: List[Dict[str, Any]]) -> float:
        """Calculate bonus for agent consensus"""
        if not agent_results or len(agent_results) < 2:
            return 0.0
        
        # Simple consensus calculation - in reality this would be more sophisticated
        return min(0.2, len(agent_results) * 0.05)  # Max 20% bonus 