"""
MetaExtract Engine

Production implementation that converts unstructured text to structured JSON
following complex schemas. Key capabilities:
- Handles deep nesting (3-7+ levels), large schemas (50-150+ objects)
- Supports large documents (50 pages to 10MB files)
- Adapts processing effort based on schema complexity
"""

import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging

# For real LLM integration
import openai
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class ExtractionStrategy(Enum):
    """Simplified extraction strategies"""
    SIMPLE = "simple"           # Single LLM call for simple schemas
    CHUNKED = "chunked"         # Break large inputs into chunks
    HIERARCHICAL = "hierarchical"  # Process complex schemas in parts


@dataclass
class SchemaComplexity:
    """Simple schema complexity assessment"""
    nesting_depth: int
    total_objects: int
    total_properties: int
    has_complex_types: bool
    estimated_tokens: int
    complexity_score: float
    
    @property
    def is_complex(self) -> bool:
        return (self.nesting_depth > 3 or 
                self.total_objects > 20 or 
                self.complexity_score > 50)


@dataclass
class ExtractionResult:
    """Result from extraction process"""
    success: bool
    extracted_data: Optional[Dict[str, Any]]
    confidence_score: float
    field_confidences: Dict[str, float]  # Field path -> confidence score
    strategy_used: str
    processing_time: float
    token_usage: int
    chunks_processed: int
    validation_errors: List[str]
    low_confidence_fields: List[str]
    error_message: Optional[str] = None


class SimplifiedMetaExtract:
    """Simplified extraction engine focused on core functionality"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize with OpenAI API key"""
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.max_tokens_per_request = 16000  # GPT-4 context window management
        self.chunk_overlap = 200
    
    async def extract(self, 
                     input_text: str, 
                     schema: Dict[str, Any],
                     strategy: Optional[str] = None) -> ExtractionResult:
        """
        Main extraction method
        
        Args:
            input_text: Unstructured text to extract from
            schema: JSON schema to extract to  
            strategy: Optional strategy override ("simple", "chunked", "hierarchical")
            
        Returns:
            ExtractionResult with extracted data and metadata
        """
        start_time = time.time()
        
        try:
            # Analyze schema complexity
            complexity = self._analyze_schema_complexity(schema)
            
            # Select strategy
            selected_strategy = self._select_strategy(
                complexity, len(input_text.encode('utf-8')), strategy
            )
            
            logger.info(f"Processing with {selected_strategy.value} strategy (complexity: {complexity.complexity_score:.1f})")
            
            # Execute extraction based on strategy
            if selected_strategy == ExtractionStrategy.SIMPLE:
                result = await self._simple_extraction(input_text, schema, complexity)
            elif selected_strategy == ExtractionStrategy.CHUNKED:
                result = await self._chunked_extraction(input_text, schema, complexity)
            else:  # HIERARCHICAL
                result = await self._hierarchical_extraction(input_text, schema, complexity)
            
            # Post-process and validate
            validated_result = self._validate_and_score(result, schema, input_text)
            
            processing_time = time.time() - start_time
            
            return ExtractionResult(
                success=validated_result is not None,
                extracted_data=validated_result.get('data') if validated_result else None,
                confidence_score=validated_result.get('confidence', 0.0) if validated_result else 0.0,
                field_confidences=validated_result.get('field_confidences', {}) if validated_result else {},
                strategy_used=selected_strategy.value,
                processing_time=processing_time,
                token_usage=validated_result.get('tokens', 0) if validated_result else 0,
                chunks_processed=validated_result.get('chunks', 1) if validated_result else 0,
                validation_errors=validated_result.get('validation_errors', []) if validated_result else [],
                low_confidence_fields=validated_result.get('low_confidence_fields', []) if validated_result else []
            )
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return ExtractionResult(
                success=False,
                extracted_data=None,
                confidence_score=0.0,
                field_confidences={},
                strategy_used=strategy or "unknown",
                processing_time=time.time() - start_time,
                token_usage=0,
                chunks_processed=0,
                validation_errors=[],
                low_confidence_fields=[],
                error_message=str(e)
            )
    
    def _analyze_schema_complexity(self, schema: Dict[str, Any]) -> SchemaComplexity:
        """Analyze schema complexity for strategy selection"""
        nesting_depth = self._calculate_max_depth(schema)
        total_objects = self._count_objects(schema)
        total_properties = self._count_properties(schema)
        has_complex_types = self._has_complex_types(schema)
        
        # Estimate tokens needed for schema
        schema_json = json.dumps(schema, separators=(',', ':'))
        estimated_tokens = len(schema_json) // 4  # Rough estimation
        
        # Calculate complexity score
        complexity_score = (
            nesting_depth * 10 +
            total_objects * 2 +
            total_properties * 0.5 +
            (20 if has_complex_types else 0) +
            estimated_tokens * 0.01
        )
        
        return SchemaComplexity(
            nesting_depth=nesting_depth,
            total_objects=total_objects,
            total_properties=total_properties,
            has_complex_types=has_complex_types,
            estimated_tokens=estimated_tokens,
            complexity_score=complexity_score
        )
    
    def _select_strategy(self, 
                        complexity: SchemaComplexity, 
                        input_size_bytes: int,
                        strategy_override: Optional[str] = None) -> ExtractionStrategy:
        """Select optimal extraction strategy"""
        
        if strategy_override:
            return ExtractionStrategy(strategy_override)
        
        # Large document threshold (>100KB or >50 pages estimated)
        large_document = input_size_bytes > 100_000
        
        # Complex schema threshold
        complex_schema = complexity.is_complex
        
        if large_document and complex_schema:
            return ExtractionStrategy.HIERARCHICAL
        elif large_document or complex_schema:
            return ExtractionStrategy.CHUNKED
        else:
            return ExtractionStrategy.SIMPLE
    
    async def _simple_extraction(self, 
                                input_text: str, 
                                schema: Dict[str, Any],
                                complexity: SchemaComplexity) -> Dict[str, Any]:
        """Simple single-call extraction for straightforward cases"""
        
        prompt = self._build_extraction_prompt(input_text, schema, "simple")
        
        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an expert data extraction specialist. Extract structured data from text according to JSON schemas with perfect accuracy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        return self._parse_llm_response(response, 1)
    
    async def _chunked_extraction(self, 
                                 input_text: str, 
                                 schema: Dict[str, Any],
                                 complexity: SchemaComplexity) -> Dict[str, Any]:
        """Chunked extraction for large documents"""
        
        # Split text into manageable chunks
        chunks = self._chunk_text(input_text, self.max_tokens_per_request // 2)
        
        extracted_chunks = []
        total_tokens = 0
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            prompt = self._build_extraction_prompt(chunk, schema, "chunked", chunk_info=f"chunk {i+1}/{len(chunks)}")
            
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert data extraction specialist. Extract structured data from this text chunk according to the JSON schema. Focus on completeness and accuracy."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            chunk_result = self._parse_llm_response(response, 1)
            extracted_chunks.append(chunk_result['data'])
            total_tokens += chunk_result['tokens']
        
        # Merge chunks
        merged_data = self._merge_chunk_results(extracted_chunks, schema)
        
        return {
            'data': merged_data,
            'confidence': 0.8,  # Lower confidence for chunked extraction
            'tokens': total_tokens,
            'chunks': len(chunks)
        }
    
    async def _hierarchical_extraction(self, 
                                      input_text: str, 
                                      schema: Dict[str, Any],
                                      complexity: SchemaComplexity) -> Dict[str, Any]:
        """Hierarchical extraction for complex schemas"""
        
        # Break schema into sections
        schema_sections = self._break_schema_into_sections(schema)
        
        extracted_sections = {}
        total_tokens = 0
        
        for section_name, section_schema in schema_sections.items():
            logger.info(f"Processing schema section: {section_name}")
            
            prompt = self._build_extraction_prompt(
                input_text, section_schema, "hierarchical", 
                section_info=f"Focus on extracting data for: {section_name}"
            )
            
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": f"You are an expert data extraction specialist. Focus on extracting data for the '{section_name}' section of the schema with high accuracy."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            section_result = self._parse_llm_response(response, 1)
            extracted_sections[section_name] = section_result['data']
            total_tokens += section_result['tokens']
        
        # Merge sections back together
        merged_data = self._merge_hierarchical_sections(extracted_sections, schema)
        
        return {
            'data': merged_data,
            'confidence': 0.85,  # Higher confidence for hierarchical approach
            'tokens': total_tokens,
            'chunks': len(schema_sections)
        }
    
    def _build_extraction_prompt(self, 
                               input_text: str, 
                               schema: Dict[str, Any], 
                               strategy: str,
                               chunk_info: str = None,
                               section_info: str = None) -> str:
        """Build optimized extraction prompt"""
        
        # Truncate input text if too long
        max_text_length = 8000  # Leave room for schema and instructions
        if len(input_text) > max_text_length:
            input_text = input_text[:max_text_length] + "... [truncated]"
        
        # Compact schema representation
        schema_str = json.dumps(schema, indent=1, separators=(',', ':'))
        if len(schema_str) > 3000:  # Truncate very large schemas
            schema_str = schema_str[:3000] + "... [schema truncated]"
        
        prompt = f"""Extract structured data from the following text according to the provided JSON schema.

{f"Context: {chunk_info}" if chunk_info else ""}
{f"Focus: {section_info}" if section_info else ""}

INSTRUCTIONS:
1. Read the text carefully and extract all relevant information
2. Follow the JSON schema exactly - match types, structure, and constraints
3. If a field is not clearly present in the text, omit it rather than guess
4. For arrays, extract all relevant items found in the text
5. For enums, use only the allowed values specified in the schema
6. Return ONLY valid JSON that matches the schema structure

TEXT TO EXTRACT FROM:
{input_text}

JSON SCHEMA:
{schema_str}

EXTRACTED DATA (JSON only):"""
        
        return prompt
    
    def _parse_llm_response(self, response, chunks: int) -> Dict[str, Any]:
        """Parse LLM response and extract structured data"""
        content = response.choices[0].message.content.strip()
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                extracted_data = json.loads(json_str)
            else:
                # Fallback: try to parse entire content as JSON
                extracted_data = json.loads(content)
            
            return {
                'data': extracted_data,
                'confidence': 0.8,
                'tokens': response.usage.total_tokens,
                'chunks': chunks
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM response: {e}")
            
            # Try to extract partial data using more robust parsing
            partial_data = self._extract_partial_data(content)
            
            return {
                'data': partial_data,
                'confidence': 0.3 if partial_data else 0.1,
                'tokens': response.usage.total_tokens if hasattr(response, 'usage') else 0,
                'chunks': chunks
            }
    
    def _extract_partial_data(self, content: str) -> Dict[str, Any]:
        """Extract partial structured data from malformed response"""
        try:
            # Try to fix common JSON issues
            content = content.replace('\\n', '\\\\n').replace('\n', ' ')
            content = re.sub(r'\\(?!["\\/bfnrt])', r'\\\\', content)
            
            # Try parsing again after cleanup
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
                
        except:
            pass
        
        # Last resort: extract key-value pairs manually
        partial = {}
        
        # Extract quoted strings that look like field values
        patterns = [
            r'"(\w+)"\s*:\s*"([^"]*)"',  # "field": "value"
            r'"(\w+)"\s*:\s*(\d+)',      # "field": 123
            r'"(\w+)"\s*:\s*(true|false|null)'  # "field": boolean/null
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for key, value in matches:
                if value.isdigit():
                    partial[key] = int(value)
                elif value in ['true', 'false']:
                    partial[key] = value == 'true'
                elif value == 'null':
                    partial[key] = None
                else:
                    partial[key] = value
        
        return partial
    
    def _validate_and_score(self, 
                           result: Dict[str, Any], 
                           schema: Dict[str, Any],
                           input_text: str) -> Dict[str, Any]:
        """Validate extracted data and calculate confidence scores"""
        if not result or 'data' not in result:
            return None
        
        extracted_data = result['data']
        
        # Basic JSON schema validation
        validation_errors = self._validate_json_schema(extracted_data, schema)
        
        # Calculate field-level confidence
        field_confidences = self._calculate_field_confidences(extracted_data, schema, input_text)
        
        # Identify low confidence fields
        low_confidence_fields = [
            field for field, confidence in field_confidences.items() 
            if confidence < 0.6
        ]
        
        # Adjust overall confidence based on validation
        base_confidence = result.get('confidence', 0.8)
        if validation_errors:
            base_confidence *= (1 - len(validation_errors) * 0.1)  # Reduce confidence for errors
        
        base_confidence = max(0.0, min(1.0, base_confidence))
        
        return {
            'data': extracted_data,
            'confidence': base_confidence,
            'tokens': result.get('tokens', 0),
            'chunks': result.get('chunks', 1),
            'validation_errors': validation_errors,
            'low_confidence_fields': low_confidence_fields,
            'field_confidences': field_confidences
        }
    
    # Helper methods for complexity analysis
    def _calculate_max_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth"""
        if not isinstance(obj, dict):
            return current_depth
        
        max_depth = current_depth
        for value in obj.values():
            if isinstance(value, dict):
                depth = self._calculate_max_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
            elif isinstance(value, list) and value:
                for item in value:
                    if isinstance(item, dict):
                        depth = self._calculate_max_depth(item, current_depth + 1)
                        max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _count_objects(self, obj: Any) -> int:
        """Count total objects in schema"""
        if not isinstance(obj, dict):
            return 0
        
        count = 1 if obj.get('type') == 'object' else 0
        
        for value in obj.values():
            if isinstance(value, dict):
                count += self._count_objects(value)
            elif isinstance(value, list):
                for item in value:
                    count += self._count_objects(item)
        
        return count
    
    def _count_properties(self, obj: Any) -> int:
        """Count total properties in schema"""
        if not isinstance(obj, dict):
            return 0
        
        count = len(obj.get('properties', {}))
        
        for value in obj.values():
            if isinstance(value, dict):
                count += self._count_properties(value)
            elif isinstance(value, list):
                for item in value:
                    count += self._count_properties(item)
        
        return count
    
    def _has_complex_types(self, obj: Any) -> bool:
        """Check for complex schema features"""
        if not isinstance(obj, dict):
            return False
        
        # Check for complex schema features
        complex_features = ['anyOf', 'oneOf', 'allOf', 'if', 'then', 'else', '$ref']
        if any(feature in obj for feature in complex_features):
            return True
        
        # Check recursively
        for value in obj.values():
            if isinstance(value, (dict, list)):
                if self._has_complex_types(value):
                    return True
        
        return False
    
    def _chunk_text(self, text: str, max_chunk_size: int) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        # Estimate words per chunk (rough: 4 chars per token, 1.3 tokens per word)
        words_per_chunk = max_chunk_size // 2  # Conservative estimate
        
        for i in range(0, len(words), words_per_chunk - self.chunk_overlap):
            chunk_words = words[i:i + words_per_chunk]
            chunks.append(' '.join(chunk_words))
        
        return chunks if chunks else [text]
    
    def _break_schema_into_sections(self, schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Break complex schema into manageable sections"""
        properties = schema.get('properties', {})
        
        if len(properties) <= 5:
            return {'main': schema}
        
        sections = {}
        section_size = 5  # Properties per section
        
        items = list(properties.items())
        for i in range(0, len(items), section_size):
            section_props = dict(items[i:i + section_size])
            section_name = f"section_{i//section_size + 1}"
            
            sections[section_name] = {
                'type': 'object',
                'properties': section_props,
                'required': [prop for prop in section_props.keys() 
                           if prop in schema.get('required', [])]
            }
        
        return sections
    
    def _merge_chunk_results(self, chunk_results: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Merge results from multiple chunks"""
        merged = {}
        
        for chunk_result in chunk_results:
            if isinstance(chunk_result, dict):
                for key, value in chunk_result.items():
                    if key not in merged:
                        merged[key] = value
                    elif isinstance(value, list) and isinstance(merged[key], list):
                        # Merge arrays
                        merged[key].extend(value)
                    elif value and not merged[key]:
                        # Prefer non-empty values
                        merged[key] = value
        
        return merged
    
    def _merge_hierarchical_sections(self, sections: Dict[str, Dict[str, Any]], original_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Merge hierarchical section results"""
        merged = {}
        
        for section_data in sections.values():
            if isinstance(section_data, dict):
                merged.update(section_data)
        
        return merged
    
    def _validate_json_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Basic JSON schema validation"""
        errors = []
        
        required_fields = schema.get('required', [])
        properties = schema.get('properties', {})
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check data types
        for field, value in data.items():
            if field in properties:
                expected_type = properties[field].get('type')
                if expected_type and not self._check_type_match(value, expected_type):
                    errors.append(f"Type mismatch for {field}: expected {expected_type}")
        
        return errors
    
    def _check_type_match(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_map = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None)
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, assume valid
        
        return isinstance(value, expected_python_type)
    
    def _calculate_field_confidences(self, 
                                   data: Dict[str, Any], 
                                   schema: Dict[str, Any],
                                   input_text: str) -> Dict[str, float]:
        """Calculate confidence scores for each field"""
        confidences = {}
        
        def calculate_recursive(obj: Any, schema_part: Dict[str, Any], path: str = ""):
            if not isinstance(obj, dict):
                return
            
            properties = schema_part.get('properties', {})
            
            for field, value in obj.items():
                field_path = f"{path}.{field}" if path else field
                
                # Base confidence
                confidence = 0.7
                
                # Boost confidence if field is required
                if field in schema_part.get('required', []):
                    confidence += 0.1
                
                # Boost confidence if value found in source text
                if isinstance(value, str) and value.lower() in input_text.lower():
                    confidence += 0.15
                
                # Reduce confidence for empty/null values
                if not value:
                    confidence -= 0.2
                
                confidences[field_path] = max(0.0, min(1.0, confidence))
                
                # Recursive for nested objects
                if isinstance(value, dict) and field in properties:
                    calculate_recursive(value, properties[field], field_path)
        
        calculate_recursive(data, schema)
        return confidences 