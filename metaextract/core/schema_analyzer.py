"""
Schema Complexity Analyzer

Analyzes JSON schemas to determine their complexity and guide strategy selection.
Key metrics: nesting levels, object count, enum complexity, field types, etc.
"""

import json
import re
from typing import Dict, List, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum


class ComplexityLevel(Enum):
    """Schema complexity levels"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class SchemaMetrics:
    """Comprehensive schema complexity metrics"""
    # Core metrics
    max_nesting_depth: int
    total_objects: int
    total_properties: int
    total_arrays: int
    
    # Complexity indicators
    enum_count: int
    max_enum_size: int
    conditional_schemas: int  # anyOf, oneOf, if/then/else
    reference_count: int  # $ref usage
    
    # Type complexity
    unique_types: Set[str]
    pattern_count: int
    format_count: int
    
    # Size indicators
    schema_size_bytes: int
    estimated_tokens: int
    
    # Overall assessment
    complexity_level: ComplexityLevel
    complexity_score: float
    
    # Recommendations
    requires_chunking: bool
    requires_multi_agent: bool
    estimated_model_calls: int


class SchemaComplexityAnalyzer:
    """Analyzes JSON schemas to determine their complexity and processing requirements"""
    
    def __init__(self):
        self.type_weights = {
            'object': 2,
            'array': 1.5,
            'string': 0.5,
            'number': 0.3,
            'integer': 0.3,
            'boolean': 0.2,
            'null': 0.1
        }
        
        self.complexity_thresholds = {
            'simple': 10,
            'medium': 25,
            'complex': 50,
            'very_complex': float('inf')
        }
    
    def analyze_schema(self, schema: Dict[str, Any]) -> SchemaMetrics:
        """
        Analyze a JSON schema and return comprehensive complexity metrics
        
        Args:
            schema: JSON schema as dictionary
            
        Returns:
            SchemaMetrics with all complexity analysis
        """
        # Convert schema to JSON string for size calculation
        schema_json = json.dumps(schema, separators=(',', ':'))
        schema_size = len(schema_json.encode('utf-8'))
        
        # Initialize metrics collection
        metrics = {
            'max_nesting_depth': 0,
            'total_objects': 0,
            'total_properties': 0,
            'total_arrays': 0,
            'enum_count': 0,
            'max_enum_size': 0,
            'conditional_schemas': 0,
            'reference_count': 0,
            'unique_types': set(),
            'pattern_count': 0,
            'format_count': 0
        }
        
        # Analyze schema recursively
        self._analyze_recursive(schema, metrics, depth=0)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(metrics, schema_size)
        
        # Determine complexity level
        complexity_level = self._determine_complexity_level(complexity_score)
        
        # Generate processing recommendations
        recommendations = self._generate_recommendations(metrics, complexity_score, schema_size)
        
        return SchemaMetrics(
            max_nesting_depth=metrics['max_nesting_depth'],
            total_objects=metrics['total_objects'],
            total_properties=metrics['total_properties'],
            total_arrays=metrics['total_arrays'],
            enum_count=metrics['enum_count'],
            max_enum_size=metrics['max_enum_size'],
            conditional_schemas=metrics['conditional_schemas'],
            reference_count=metrics['reference_count'],
            unique_types=metrics['unique_types'],
            pattern_count=metrics['pattern_count'],
            format_count=metrics['format_count'],
            schema_size_bytes=schema_size,
            estimated_tokens=schema_size // 4,  # Rough token estimation
            complexity_level=complexity_level,
            complexity_score=complexity_score,
            requires_chunking=recommendations['requires_chunking'],
            requires_multi_agent=recommendations['requires_multi_agent'],
            estimated_model_calls=recommendations['estimated_model_calls']
        )
    
    def _analyze_recursive(self, node: Any, metrics: Dict, depth: int) -> None:
        """Recursively analyze schema node and update metrics"""
        # Update max nesting depth
        metrics['max_nesting_depth'] = max(metrics['max_nesting_depth'], depth)
        
        if not isinstance(node, dict):
            return
        
        # Count references
        if '$ref' in node:
            metrics['reference_count'] += 1
        
        # Count conditional schemas
        for conditional in ['anyOf', 'oneOf', 'allOf', 'if', 'then', 'else']:
            if conditional in node:
                metrics['conditional_schemas'] += 1
        
        # Count patterns and formats
        if 'pattern' in node:
            metrics['pattern_count'] += 1
        if 'format' in node:
            metrics['format_count'] += 1
        
        # Analyze type
        node_type = node.get('type')
        if node_type:
            metrics['unique_types'].add(node_type)
            
            if node_type == 'object':
                metrics['total_objects'] += 1
                # Count properties
                properties = node.get('properties', {})
                metrics['total_properties'] += len(properties)
                
                # Recursively analyze properties
                for prop_name, prop_schema in properties.items():
                    self._analyze_recursive(prop_schema, metrics, depth + 1)
                
                # Analyze additional properties
                additional = node.get('additionalProperties')
                if isinstance(additional, dict):
                    self._analyze_recursive(additional, metrics, depth + 1)
                
            elif node_type == 'array':
                metrics['total_arrays'] += 1
                # Analyze array items
                items = node.get('items')
                if items:
                    if isinstance(items, dict):
                        self._analyze_recursive(items, metrics, depth + 1)
                    elif isinstance(items, list):
                        for item in items:
                            self._analyze_recursive(item, metrics, depth + 1)
        
        # Handle enum
        if 'enum' in node:
            enum_values = node['enum']
            metrics['enum_count'] += 1
            metrics['max_enum_size'] = max(metrics['max_enum_size'], len(enum_values))
        
        # Handle definitions/components
        for definitions_key in ['definitions', '$defs', 'components']:
            if definitions_key in node:
                definitions = node[definitions_key]
                if isinstance(definitions, dict):
                    for def_name, def_schema in definitions.items():
                        self._analyze_recursive(def_schema, metrics, depth + 1)
        
        # Handle conditional schemas recursively
        for conditional_key in ['anyOf', 'oneOf', 'allOf']:
            if conditional_key in node:
                conditional_schemas = node[conditional_key]
                if isinstance(conditional_schemas, list):
                    for conditional_schema in conditional_schemas:
                        self._analyze_recursive(conditional_schema, metrics, depth + 1)
        
        # Handle if/then/else
        for conditional_key in ['if', 'then', 'else']:
            if conditional_key in node:
                self._analyze_recursive(node[conditional_key], metrics, depth + 1)
    
    def _calculate_complexity_score(self, metrics: Dict, schema_size: int) -> float:
        """Calculate overall complexity score"""
        score = 0.0
        
        # Nesting penalty (exponential)
        score += metrics['max_nesting_depth'] ** 1.5 * 2
        
        # Object complexity
        score += metrics['total_objects'] * 1.5
        score += metrics['total_properties'] * 0.5
        score += metrics['total_arrays'] * 1.0
        
        # Enum complexity
        score += metrics['enum_count'] * 1.0
        score += metrics['max_enum_size'] * 0.1
        
        # Conditional complexity (high weight)
        score += metrics['conditional_schemas'] * 3.0
        
        # Reference complexity
        score += metrics['reference_count'] * 0.5
        
        # Pattern/format complexity
        score += metrics['pattern_count'] * 1.0
        score += metrics['format_count'] * 0.5
        
        # Type diversity
        score += len(metrics['unique_types']) * 0.3
        
        # Size penalty
        score += (schema_size / 1000) * 0.5
        
        return score
    
    def _determine_complexity_level(self, score: float) -> ComplexityLevel:
        """Determine complexity level based on score"""
        if score <= self.complexity_thresholds['simple']:
            return ComplexityLevel.SIMPLE
        elif score <= self.complexity_thresholds['medium']:
            return ComplexityLevel.MEDIUM
        elif score <= self.complexity_thresholds['complex']:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.VERY_COMPLEX
    
    def _generate_recommendations(self, metrics: Dict, score: float, schema_size: int) -> Dict[str, Any]:
        """Generate processing recommendations based on analysis"""
        recommendations = {}
        
        # Chunking recommendation
        recommendations['requires_chunking'] = (
            metrics['max_nesting_depth'] > 4 or
            metrics['total_objects'] > 30 or
            schema_size > 50000  # 50KB
        )
        
        # Multi-agent recommendation
        recommendations['requires_multi_agent'] = (
            score > self.complexity_thresholds['medium'] or
            metrics['conditional_schemas'] > 5 or
            metrics['total_objects'] > 20
        )
        
        # Estimate model calls needed
        base_calls = 1
        if recommendations['requires_chunking']:
            base_calls += metrics['total_objects'] // 10
        if recommendations['requires_multi_agent']:
            base_calls += 2
        if metrics['conditional_schemas'] > 0:
            base_calls += metrics['conditional_schemas']
        
        recommendations['estimated_model_calls'] = min(base_calls, 20)  # Cap at 20
        
        return recommendations
    
    def get_complexity_summary(self, metrics: SchemaMetrics) -> str:
        """Generate human-readable complexity summary"""
        summary = f"""
Schema Complexity Analysis:
==========================
Complexity Level: {metrics.complexity_level.value.upper()}
Complexity Score: {metrics.complexity_score:.2f}

Structure:
- Max Nesting Depth: {metrics.max_nesting_depth}
- Total Objects: {metrics.total_objects}
- Total Properties: {metrics.total_properties}
- Total Arrays: {metrics.total_arrays}

Advanced Features:
- Enums: {metrics.enum_count} (max size: {metrics.max_enum_size})
- Conditional Schemas: {metrics.conditional_schemas}
- References: {metrics.reference_count}
- Patterns: {metrics.pattern_count}

Processing Recommendations:
- Requires Chunking: {metrics.requires_chunking}
- Requires Multi-Agent: {metrics.requires_multi_agent}
- Estimated Model Calls: {metrics.estimated_model_calls}

Schema Size: {metrics.schema_size_bytes:,} bytes (~{metrics.estimated_tokens:,} tokens)
        """.strip()
        
        return summary 