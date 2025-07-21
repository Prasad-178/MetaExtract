"""
Simplified orchestrator for agentic extraction approach
"""

import time
import json
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import asdict

from .models import (
    AgenticExtractionRequest, AgenticExtractionResult, AgentResult, 
    AgentRole
)
from .crews import SimplifiedCrewFactory
from metaextract.simplified_extractor import SimplifiedMetaExtract

logger = logging.getLogger(__name__)


class AgenticMetaExtract:
    """
    Simplified agentic extraction engine using 3 core agents
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize agentic extractor"""
        self.api_key = openai_api_key
        self.crew_factory = SimplifiedCrewFactory(openai_api_key)
        
        # For fallback to traditional approach if needed
        self.traditional_extractor = SimplifiedMetaExtract(openai_api_key)
        
        # Performance tracking
        self.extraction_history = []
    
    async def extract(self, request: AgenticExtractionRequest) -> AgenticExtractionResult:
        """
        Enhanced agentic extraction method with traditional approach components
        """
        start_time = time.time()
        
        try:
            # Analyze schema complexity using traditional approach methods
            complexity = self.traditional_extractor._analyze_schema_complexity(request.json_schema)
            logger.info(f"Schema complexity: {complexity.complexity_score:.1f} (depth: {complexity.nesting_depth}, objects: {complexity.total_objects})")
            
            # Create enhanced crew with 3 agents
            crew = self.crew_factory.create_fast_extraction_crew(request.text, request.json_schema)
            
            # Execute crew
            logger.info(f"Starting enhanced agentic extraction with {len(crew.agents)} agents")
            crew_result = await self._execute_crew_async(crew)
            
            # Parse and extract JSON from crew result
            extracted_data, base_confidence = self._parse_crew_result_to_json(crew_result)
            
            # Enhanced validation and scoring using traditional approach methods
            validation_errors = []
            field_confidences = {}
            low_confidence_fields = []
            
            if extracted_data:
                # Schema validation using traditional approach
                validation_errors = self.traditional_extractor._validate_json_schema(
                    extracted_data, request.json_schema
                )
                
                # Field confidence calculation using traditional approach
                field_confidences = self.traditional_extractor._calculate_field_confidences(
                    extracted_data, request.json_schema, request.text
                )
                
                # Identify low confidence fields
                low_confidence_fields = [
                    field for field, confidence in field_confidences.items() 
                    if confidence < request.confidence_threshold
                ]
                
                # Calculate enhanced overall confidence
                enhanced_confidence = self._calculate_enhanced_confidence(
                    base_confidence, validation_errors, field_confidences, complexity
                )
            else:
                enhanced_confidence = 0.0
            
            processing_time = time.time() - start_time
            
            # Create detailed agent results for tracking
            agent_results = [
                AgentResult(
                    agent_id="schema_analyzer",
                    agent_role=AgentRole.SCHEMA_ANALYZER,
                    extracted_data={},
                    confidence=0.9,  # Schema analysis is typically reliable
                    reasoning=f"Analyzed schema complexity: {complexity.complexity_score:.1f}, depth: {complexity.nesting_depth}",
                    processing_time=processing_time * 0.2,  # Estimated proportion
                    tokens_used=self._estimate_tokens_used(json.dumps(request.json_schema), {}),
                    field_confidences={}
                ),
                AgentResult(
                    agent_id="data_extractor",
                    agent_role=AgentRole.EXTRACTION_SPECIALIST,
                    extracted_data=extracted_data or {},
                    confidence=base_confidence,
                    reasoning="Extracted structured data from input text using schema guidance",
                    processing_time=processing_time * 0.6,  # Main extraction work
                    tokens_used=self._estimate_tokens_used(request.text, request.json_schema),
                    field_confidences=field_confidences
                ),
                AgentResult(
                    agent_id="qa_validator",
                    agent_role=AgentRole.QUALITY_ASSURANCE,
                    extracted_data=extracted_data or {},
                    confidence=enhanced_confidence,
                    reasoning=f"Validated data: {len(validation_errors)} errors, {len(low_confidence_fields)} low-confidence fields",
                    processing_time=processing_time * 0.2,  # Validation work
                    tokens_used=500,  # Estimated for validation
                    field_confidences=field_confidences
                )
            ]
            
            result = AgenticExtractionResult(
                success=extracted_data is not None,
                final_data=extracted_data,
                overall_confidence=enhanced_confidence,
                agent_results=agent_results,
                strategy_used="enhanced_agentic",
                total_processing_time=processing_time,
                total_tokens_used=sum(ar.tokens_used for ar in agent_results),
                agents_used=3,  # Schema analyzer, extractor, QA
                validation_errors=validation_errors,
                consensus_fields=self._identify_consensus_fields(field_confidences),
                conflicting_fields={},  # No conflicts in sequential approach
                low_confidence_fields=low_confidence_fields,
                performance_metrics={
                    "schema_complexity": complexity.complexity_score,
                    "validation_score": 1.0 - (len(validation_errors) * 0.1),
                    "field_coverage": len(field_confidences) / max(1, complexity.total_properties),
                    "processing_efficiency": min(1.0, 30.0 / processing_time)  # Target 30s or less
                }
            )
            
            # Track performance
            self._track_performance(request, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced agentic extraction failed: {e}")
            
            # Fallback to traditional approach
            logger.info("Falling back to traditional extraction")
            traditional_result = await self.traditional_extractor.extract(
                request.text, request.json_schema
            )
            
            return AgenticExtractionResult(
                success=traditional_result.success,
                final_data=traditional_result.extracted_data,
                overall_confidence=traditional_result.confidence_score,
                agent_results=[],
                strategy_used="fallback_traditional",
                total_processing_time=time.time() - start_time,
                total_tokens_used=traditional_result.token_usage,
                agents_used=0,
                validation_errors=traditional_result.validation_errors,
                performance_metrics={"fallback_used": True}
            )
    
    async def _execute_crew_async(self, crew) -> str:
        """Execute crew asynchronously"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, crew.kickoff)
        return result
    
    def _parse_crew_result_to_json(self, crew_result) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Improved parsing of crew results to extract clean JSON
        """
        try:
            # Get the result text
            if hasattr(crew_result, 'raw'):
                result_text = str(crew_result.raw)
            elif hasattr(crew_result, 'result'):
                result_text = str(crew_result.result)
            else:
                result_text = str(crew_result)
            
            # Clean the text
            result_text = result_text.strip()
            
            # Try to find and extract JSON
            extracted_json = self._extract_json_from_text(result_text)
            
            if extracted_json:
                return extracted_json, 0.85  # High confidence for successful extraction
            else:
                logger.warning("Could not extract valid JSON from crew result")
                return None, 0.0
                
        except Exception as e:
            logger.error(f"Error parsing crew result: {e}")
            return None, 0.0
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON object from text using multiple strategies
        """
        # Strategy 1: Find content between first { and last } (for complete JSON)
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_candidate = text[start_idx:end_idx + 1]
                
                # Clean up the JSON string
                json_candidate = self._clean_json_string(json_candidate)
                
                # Try to parse the complete JSON
                parsed = json.loads(json_candidate)
                if isinstance(parsed, dict) and parsed:
                    return parsed
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Look for large JSON patterns (multi-line)
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON
            r'\{.*?\}',  # Simple JSON
        ]
        
        # Find the longest JSON match
        longest_match = None
        longest_length = 0
        
        for pattern in json_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    potential_json = match.group().strip()
                    if len(potential_json) > longest_length:
                        # Try to parse to validate
                        cleaned = self._clean_json_string(potential_json)
                        parsed = json.loads(cleaned)
                        if isinstance(parsed, dict) and parsed:
                            longest_match = parsed
                            longest_length = len(potential_json)
                except json.JSONDecodeError:
                    continue
        
        if longest_match:
            return longest_match
        
        # Strategy 3: Try to parse the entire text as JSON
        try:
            cleaned_text = self._clean_json_string(text)
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean JSON string to fix common formatting issues
        """
        # Remove markdown code blocks if present
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*$', '', json_str)
        
        # Remove leading/trailing whitespace and newlines
        json_str = json_str.strip()
        
        # Fix common escaping issues
        json_str = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', json_str)
        
        # Remove any text before the first { or after the last }
        start = json_str.find('{')
        end = json_str.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            json_str = json_str[start:end + 1]
        
        return json_str
    
    def _calculate_enhanced_confidence(self, 
                                    base_confidence: float, 
                                    validation_errors: List[str], 
                                    field_confidences: Dict[str, float],
                                    complexity) -> float:
        """
        Calculate enhanced confidence score combining multiple factors
        """
        # Start with base confidence from agent
        confidence = base_confidence
        
        # Adjust based on validation errors
        if validation_errors:
            validation_penalty = min(0.3, len(validation_errors) * 0.1)
            confidence = max(0.3, confidence - validation_penalty)
        
        # Boost confidence based on field confidence scores
        if field_confidences:
            avg_field_confidence = sum(field_confidences.values()) / len(field_confidences)
            # Weighted average: 70% base confidence, 30% field confidence
            confidence = (confidence * 0.7) + (avg_field_confidence * 0.3)
        
        # Small boost for completing complex schemas successfully
        if not validation_errors and complexity.is_complex:
            confidence = min(1.0, confidence + 0.05)
        
        return max(0.0, min(1.0, confidence))
    
    def _identify_consensus_fields(self, field_confidences: Dict[str, float]) -> Dict[str, Any]:
        """
        Identify fields with high confidence as consensus fields
        """
        consensus = {}
        for field, confidence in field_confidences.items():
            if confidence >= 0.8:  # High confidence threshold
                consensus[field] = confidence
        return consensus
    
    def _estimate_tokens_used(self, text: str, schema: Dict[str, Any]) -> int:
        """
        Estimate tokens used in the extraction process
        """
        # Rough estimation: text + schema + model overhead
        text_tokens = len(text) // 4  # Rough approximation
        schema_tokens = len(json.dumps(schema)) // 4
        overhead_tokens = 800  # For prompts and responses (reduced from original)
        
        return text_tokens + schema_tokens + overhead_tokens
    
    def _track_performance(self, request: AgenticExtractionRequest, result: AgenticExtractionResult):
        """Track performance metrics for continuous improvement"""
        
        performance_record = {
            "timestamp": time.time(),
            "strategy_used": result.strategy_used,
            "agents_used": result.agents_used,
            "processing_time": result.total_processing_time,
            "confidence": result.overall_confidence,
            "success": result.success,
            "schema_complexity": len(str(request.json_schema)),
            "text_length": len(request.text),
            "validation_errors": len(result.validation_errors)
        }
        
        self.extraction_history.append(performance_record)
        
        # Keep only last 50 records for efficiency
        if len(self.extraction_history) > 50:
            self.extraction_history = self.extraction_history[-50:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        if not self.extraction_history:
            return {"message": "No extraction history available"}
        
        recent_extractions = self.extraction_history[-10:]  # Last 10 extractions
        
        avg_confidence = sum(r["confidence"] for r in recent_extractions) / len(recent_extractions)
        avg_processing_time = sum(r["processing_time"] for r in recent_extractions) / len(recent_extractions)
        success_rate = sum(1 for r in recent_extractions if r["success"]) / len(recent_extractions)
        
        return {
            "total_extractions": len(self.extraction_history),
            "recent_performance": {
                "avg_confidence": round(avg_confidence, 3),
                "avg_processing_time": round(avg_processing_time, 2),
                "success_rate": round(success_rate, 3),
                "strategy_used": "simplified_fast"
            },
            "optimization_status": "Optimized for speed and cost efficiency"
        } 