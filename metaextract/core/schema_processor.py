"""
Hierarchical Schema Processor

Breaks down complex JSON schemas into manageable chunks while preserving 
dependencies and relationships. Enables efficient processing of large, 
nested schemas through intelligent decomposition.
"""

from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import logging

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of schema chunks"""
    ROOT = "root"
    OBJECT = "object" 
    ARRAY = "array"
    DEFINITION = "definition"
    CONDITIONAL = "conditional"
    SIMPLE_FIELDS = "simple_fields"


@dataclass
class SchemaDependency:
    """Represents a dependency between schema chunks"""
    source_chunk_id: str
    target_chunk_id: str
    dependency_type: str  # "reference", "inheritance", "composition"
    field_path: str
    is_required: bool = False


@dataclass 
class SchemaChunk:
    """A chunk of a larger schema"""
    chunk_id: str
    chunk_type: ChunkType
    schema_content: Dict[str, Any]
    original_path: str
    dependencies: List[SchemaDependency] = field(default_factory=list)
    priority: int = 1
    estimated_complexity: float = 1.0
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: Set[str] = field(default_factory=set)


@dataclass
class ProcessingPlan:
    """Plan for processing schema chunks"""
    chunks: Dict[str, SchemaChunk]
    processing_order: List[str]  # Chunk IDs in dependency order
    parallel_groups: List[List[str]]  # Groups that can be processed in parallel
    estimated_total_complexity: float
    requires_merging: bool


class HierarchicalSchemaProcessor:
    """Processes complex schemas by breaking them into manageable hierarchical chunks"""
    
    def __init__(self, max_chunk_complexity: float = 10.0, max_chunk_size: int = 2000):
        self.max_chunk_complexity = max_chunk_complexity
        self.max_chunk_size = max_chunk_size  # in tokens
        self.chunk_counter = 0
        
        # Complexity weights for different schema elements
        self.complexity_weights = {
            'object': 2.0,
            'array': 1.5,
            'string': 0.3,
            'number': 0.2,
            'boolean': 0.1,
            'enum': 1.0,
            'anyOf': 3.0,
            'oneOf': 3.0,
            'allOf': 2.5,
            'if': 2.0,
            'then': 1.5,
            'else': 1.5,
            '$ref': 1.0
        }
    
    def process_schema(self, schema: Dict[str, Any]) -> ProcessingPlan:
        """
        Break down a complex schema into manageable chunks with dependency tracking
        
        Args:
            schema: The JSON schema to process
            
        Returns:
            ProcessingPlan with chunks and execution order
        """
        logger.info("Starting hierarchical schema processing")
        
        # Reset state
        self.chunk_counter = 0
        chunks = {}
        
        # Step 1: Identify and extract definitions
        definitions_chunks = self._extract_definitions(schema)
        chunks.update(definitions_chunks)
        
        # Step 2: Process main schema structure
        main_chunks = self._process_main_schema(schema, existing_chunks=definitions_chunks)
        chunks.update(main_chunks)
        
        # Step 3: Identify dependencies
        self._identify_dependencies(chunks)
        
        # Step 4: Calculate processing order
        processing_order = self._calculate_processing_order(chunks)
        
        # Step 5: Identify parallel processing opportunities
        parallel_groups = self._identify_parallel_groups(chunks, processing_order)
        
        # Step 6: Calculate total complexity
        total_complexity = sum(chunk.estimated_complexity for chunk in chunks.values())
        
        # Step 7: Determine if merging is required
        requires_merging = len(chunks) > 1
        
        plan = ProcessingPlan(
            chunks=chunks,
            processing_order=processing_order,
            parallel_groups=parallel_groups,
            estimated_total_complexity=total_complexity,
            requires_merging=requires_merging
        )
        
        logger.info(f"Schema processed into {len(chunks)} chunks, estimated complexity: {total_complexity:.2f}")
        
        return plan
    
    def _extract_definitions(self, schema: Dict[str, Any]) -> Dict[str, SchemaChunk]:
        """Extract reusable definitions from schema"""
        chunks = {}
        
        for def_key in ['definitions', '$defs', 'components']:
            if def_key in schema:
                definitions = schema[def_key]
                if isinstance(definitions, dict):
                    for def_name, def_schema in definitions.items():
                        chunk_id = self._generate_chunk_id(f"def_{def_name}")
                        complexity = self._calculate_complexity(def_schema)
                        
                        chunk = SchemaChunk(
                            chunk_id=chunk_id,
                            chunk_type=ChunkType.DEFINITION,
                            schema_content=def_schema,
                            original_path=f"{def_key}.{def_name}",
                            priority=1,  # Definitions have high priority
                            estimated_complexity=complexity
                        )
                        
                        chunks[chunk_id] = chunk
        
        return chunks
    
    def _process_main_schema(self, schema: Dict[str, Any], existing_chunks: Dict[str, SchemaChunk]) -> Dict[str, SchemaChunk]:
        """Process the main schema structure"""
        chunks = {}
        
        # Create root chunk
        root_chunk_id = self._generate_chunk_id("root")
        
        # Separate complex and simple properties
        properties = schema.get('properties', {})
        simple_props, complex_props = self._separate_properties(properties)
        
        # Create chunks for different property types
        if simple_props:
            simple_chunk = self._create_simple_fields_chunk(simple_props, root_chunk_id)
            chunks[simple_chunk.chunk_id] = simple_chunk
        
        for prop_name, prop_schema in complex_props.items():
            complex_chunks = self._process_complex_property(prop_name, prop_schema, root_chunk_id)
            chunks.update(complex_chunks)
        
        # Handle conditional schemas
        conditional_chunks = self._process_conditional_schemas(schema, root_chunk_id)
        chunks.update(conditional_chunks)
        
        # Create root chunk that references other chunks
        root_chunk = SchemaChunk(
            chunk_id=root_chunk_id,
            chunk_type=ChunkType.ROOT,
            schema_content=self._create_root_schema_reference(schema, chunks),
            original_path="",
            priority=3,  # Root has lower priority, processed after dependencies
            estimated_complexity=1.0,
            child_chunk_ids=set(chunks.keys())
        )
        
        chunks[root_chunk_id] = root_chunk
        
        return chunks
    
    def _separate_properties(self, properties: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Separate simple and complex properties"""
        simple_props = {}
        complex_props = {}
        
        for prop_name, prop_schema in properties.items():
            if self._is_simple_property(prop_schema):
                simple_props[prop_name] = prop_schema
            else:
                complex_props[prop_name] = prop_schema
        
        return simple_props, complex_props
    
    def _is_simple_property(self, prop_schema: Dict[str, Any]) -> bool:
        """Check if a property is simple (doesn't need its own chunk)"""
        prop_type = prop_schema.get('type')
        
        if prop_type in ['string', 'number', 'integer', 'boolean', 'null']:
            return True
        
        if prop_type == 'array' and self._is_simple_array(prop_schema):
            return True
        
        if 'enum' in prop_schema and len(prop_schema.get('enum', [])) < 20:
            return True
        
        return False
    
    def _is_simple_array(self, array_schema: Dict[str, Any]) -> bool:
        """Check if an array schema is simple"""
        items = array_schema.get('items', {})
        if isinstance(items, dict):
            return self._is_simple_property(items)
        return False
    
    def _create_simple_fields_chunk(self, simple_props: Dict[str, Any], parent_id: str) -> SchemaChunk:
        """Create a chunk for simple fields"""
        chunk_id = self._generate_chunk_id("simple_fields")
        
        schema_content = {
            "type": "object",
            "properties": simple_props
        }
        
        complexity = self._calculate_complexity(schema_content)
        
        return SchemaChunk(
            chunk_id=chunk_id,
            chunk_type=ChunkType.SIMPLE_FIELDS,
            schema_content=schema_content,
            original_path="properties",
            priority=2,
            estimated_complexity=complexity,
            parent_chunk_id=parent_id
        )
    
    def _process_complex_property(self, prop_name: str, prop_schema: Dict[str, Any], parent_id: str) -> Dict[str, SchemaChunk]:
        """Process a complex property into chunks"""
        chunks = {}
        
        prop_type = prop_schema.get('type')
        complexity = self._calculate_complexity(prop_schema)
        
        # If complexity is manageable, create a single chunk
        if complexity <= self.max_chunk_complexity:
            chunk_id = self._generate_chunk_id(f"prop_{prop_name}")
            
            chunk = SchemaChunk(
                chunk_id=chunk_id,
                chunk_type=ChunkType.OBJECT if prop_type == 'object' else ChunkType.ARRAY,
                schema_content=prop_schema,
                original_path=f"properties.{prop_name}",
                priority=2,
                estimated_complexity=complexity,
                parent_chunk_id=parent_id
            )
            
            chunks[chunk_id] = chunk
        
        else:
            # Break down further
            if prop_type == 'object':
                sub_chunks = self._process_object_schema(prop_name, prop_schema, parent_id)
                chunks.update(sub_chunks)
            elif prop_type == 'array':
                sub_chunks = self._process_array_schema(prop_name, prop_schema, parent_id)
                chunks.update(sub_chunks)
        
        return chunks
    
    def _process_object_schema(self, obj_name: str, obj_schema: Dict[str, Any], parent_id: str) -> Dict[str, SchemaChunk]:
        """Process a complex object schema"""
        chunks = {}
        
        obj_properties = obj_schema.get('properties', {})
        simple_props, complex_props = self._separate_properties(obj_properties)
        
        # Create chunk for simple properties if any
        if simple_props:
            simple_chunk = SchemaChunk(
                chunk_id=self._generate_chunk_id(f"{obj_name}_simple"),
                chunk_type=ChunkType.SIMPLE_FIELDS,
                schema_content={
                    "type": "object",
                    "properties": simple_props
                },
                original_path=f"properties.{obj_name}",
                priority=2,
                estimated_complexity=self._calculate_complexity({"properties": simple_props}),
                parent_chunk_id=parent_id
            )
            chunks[simple_chunk.chunk_id] = simple_chunk
        
        # Process complex properties recursively
        for complex_prop_name, complex_prop_schema in complex_props.items():
            sub_chunks = self._process_complex_property(
                f"{obj_name}_{complex_prop_name}",
                complex_prop_schema,
                parent_id
            )
            chunks.update(sub_chunks)
        
        return chunks
    
    def _process_array_schema(self, array_name: str, array_schema: Dict[str, Any], parent_id: str) -> Dict[str, SchemaChunk]:
        """Process a complex array schema"""
        chunks = {}
        
        items_schema = array_schema.get('items', {})
        if isinstance(items_schema, dict) and not self._is_simple_property(items_schema):
            # Complex array items need their own processing
            item_chunks = self._process_complex_property(
                f"{array_name}_items",
                items_schema,
                parent_id
            )
            chunks.update(item_chunks)
        else:
            # Create a single chunk for the array
            chunk_id = self._generate_chunk_id(f"array_{array_name}")
            complexity = self._calculate_complexity(array_schema)
            
            chunk = SchemaChunk(
                chunk_id=chunk_id,
                chunk_type=ChunkType.ARRAY,
                schema_content=array_schema,
                original_path=f"properties.{array_name}",
                priority=2,
                estimated_complexity=complexity,
                parent_chunk_id=parent_id
            )
            
            chunks[chunk_id] = chunk
        
        return chunks
    
    def _process_conditional_schemas(self, schema: Dict[str, Any], parent_id: str) -> Dict[str, SchemaChunk]:
        """Process conditional schemas (anyOf, oneOf, allOf, if/then/else)"""
        chunks = {}
        
        for conditional_key in ['anyOf', 'oneOf', 'allOf']:
            if conditional_key in schema:
                conditional_schemas = schema[conditional_key]
                if isinstance(conditional_schemas, list):
                    for i, conditional_schema in enumerate(conditional_schemas):
                        chunk_id = self._generate_chunk_id(f"{conditional_key}_{i}")
                        complexity = self._calculate_complexity(conditional_schema)
                        
                        chunk = SchemaChunk(
                            chunk_id=chunk_id,
                            chunk_type=ChunkType.CONDITIONAL,
                            schema_content=conditional_schema,
                            original_path=f"{conditional_key}[{i}]",
                            priority=2,
                            estimated_complexity=complexity,
                            parent_chunk_id=parent_id
                        )
                        
                        chunks[chunk_id] = chunk
        
        # Handle if/then/else
        if 'if' in schema:
            for condition_key in ['if', 'then', 'else']:
                if condition_key in schema:
                    chunk_id = self._generate_chunk_id(f"condition_{condition_key}")
                    complexity = self._calculate_complexity(schema[condition_key])
                    
                    chunk = SchemaChunk(
                        chunk_id=chunk_id,
                        chunk_type=ChunkType.CONDITIONAL,
                        schema_content=schema[condition_key],
                        original_path=condition_key,
                        priority=2,
                        estimated_complexity=complexity,
                        parent_chunk_id=parent_id
                    )
                    
                    chunks[chunk_id] = chunk
        
        return chunks
    
    def _create_root_schema_reference(self, original_schema: Dict[str, Any], chunks: Dict[str, SchemaChunk]) -> Dict[str, Any]:
        """Create a root schema that references other chunks"""
        root_schema = {
            "type": "object",
            "description": "Root schema with chunk references",
            "chunk_references": list(chunks.keys())
        }
        
        # Include required fields and other root-level properties
        for key in ['required', 'additionalProperties', 'title', 'description']:
            if key in original_schema:
                root_schema[key] = original_schema[key]
        
        return root_schema
    
    def _identify_dependencies(self, chunks: Dict[str, SchemaChunk]) -> None:
        """Identify dependencies between chunks"""
        for chunk_id, chunk in chunks.items():
            self._find_chunk_dependencies(chunk, chunks)
    
    def _find_chunk_dependencies(self, chunk: SchemaChunk, all_chunks: Dict[str, SchemaChunk]) -> None:
        """Find dependencies for a specific chunk"""
        schema_str = json.dumps(chunk.schema_content)
        
        # Look for $ref dependencies
        for other_chunk_id, other_chunk in all_chunks.items():
            if other_chunk_id == chunk.chunk_id:
                continue
            
            # Check if this chunk references another chunk
            if self._has_reference_to_chunk(chunk.schema_content, other_chunk):
                dependency = SchemaDependency(
                    source_chunk_id=chunk.chunk_id,
                    target_chunk_id=other_chunk_id,
                    dependency_type="reference",
                    field_path=other_chunk.original_path,
                    is_required=True
                )
                chunk.dependencies.append(dependency)
        
        # Add parent-child dependencies
        if chunk.parent_chunk_id:
            dependency = SchemaDependency(
                source_chunk_id=chunk.chunk_id,
                target_chunk_id=chunk.parent_chunk_id,
                dependency_type="composition",
                field_path="parent",
                is_required=True
            )
            chunk.dependencies.append(dependency)
    
    def _has_reference_to_chunk(self, schema: Dict[str, Any], target_chunk: SchemaChunk) -> bool:
        """Check if schema has a reference to target chunk"""
        # Simplified reference detection
        # In production, this would be more sophisticated
        schema_str = json.dumps(schema)
        return f"#/definitions/{target_chunk.original_path}" in schema_str or \
               f"#/$defs/{target_chunk.original_path}" in schema_str
    
    def _calculate_processing_order(self, chunks: Dict[str, SchemaChunk]) -> List[str]:
        """Calculate optimal processing order based on dependencies"""
        # Topological sort based on dependencies
        order = []
        visited = set()
        temp_visited = set()
        
        def visit(chunk_id: str):
            if chunk_id in temp_visited:
                # Circular dependency detected, handle gracefully
                logger.warning(f"Circular dependency detected involving chunk {chunk_id}")
                return
            
            if chunk_id in visited:
                return
            
            temp_visited.add(chunk_id)
            
            chunk = chunks[chunk_id]
            for dependency in chunk.dependencies:
                if dependency.target_chunk_id in chunks:
                    visit(dependency.target_chunk_id)
            
            temp_visited.remove(chunk_id)
            visited.add(chunk_id)
            order.append(chunk_id)
        
        # Sort chunks by priority first
        sorted_chunks = sorted(chunks.items(), key=lambda x: x[1].priority)
        
        for chunk_id, chunk in sorted_chunks:
            if chunk_id not in visited:
                visit(chunk_id)
        
        return order
    
    def _identify_parallel_groups(self, chunks: Dict[str, SchemaChunk], processing_order: List[str]) -> List[List[str]]:
        """Identify chunks that can be processed in parallel"""
        parallel_groups = []
        processed = set()
        
        for chunk_id in processing_order:
            if chunk_id in processed:
                continue
            
            # Find all chunks that can be processed with this one
            current_group = [chunk_id]
            chunk = chunks[chunk_id]
            
            # Check for other chunks that don't depend on each other
            for other_chunk_id in processing_order:
                if (other_chunk_id != chunk_id and 
                    other_chunk_id not in processed and
                    not self._has_dependency_conflict(chunk_id, other_chunk_id, chunks)):
                    current_group.append(other_chunk_id)
            
            parallel_groups.append(current_group)
            processed.update(current_group)
        
        return parallel_groups
    
    def _has_dependency_conflict(self, chunk_id1: str, chunk_id2: str, chunks: Dict[str, SchemaChunk]) -> bool:
        """Check if two chunks have conflicting dependencies"""
        chunk1 = chunks[chunk_id1]
        chunk2 = chunks[chunk_id2]
        
        # Check if chunk1 depends on chunk2 or vice versa
        for dep in chunk1.dependencies:
            if dep.target_chunk_id == chunk_id2:
                return True
        
        for dep in chunk2.dependencies:
            if dep.target_chunk_id == chunk_id1:
                return True
        
        return False
    
    def _calculate_complexity(self, schema: Dict[str, Any]) -> float:
        """Calculate complexity score for a schema"""
        complexity = 0.0
        
        def calculate_recursive(node: Any, depth: int = 0) -> float:
            if depth > 10:  # Prevent infinite recursion
                return 0.0
            
            if not isinstance(node, dict):
                return 0.1
            
            score = 0.0
            
            # Base complexity for object
            node_type = node.get('type', 'unknown')
            score += self.complexity_weights.get(node_type, 0.5)
            
            # Depth penalty
            score += depth * 0.5
            
            # Property count
            properties = node.get('properties', {})
            score += len(properties) * 0.2
            
            # Enum complexity
            if 'enum' in node:
                score += len(node['enum']) * 0.1
            
            # Conditional complexity
            for conditional in ['anyOf', 'oneOf', 'allOf']:
                if conditional in node:
                    score += self.complexity_weights.get(conditional, 2.0)
                    if isinstance(node[conditional], list):
                        for item in node[conditional]:
                            score += calculate_recursive(item, depth + 1)
            
            # Reference complexity
            if '$ref' in node:
                score += self.complexity_weights.get('$ref', 1.0)
            
            # Recursively calculate for properties
            for prop_schema in properties.values():
                score += calculate_recursive(prop_schema, depth + 1)
            
            # Array items complexity
            if 'items' in node:
                score += calculate_recursive(node['items'], depth + 1)
            
            return score
        
        return calculate_recursive(schema)
    
    def _generate_chunk_id(self, base_name: str) -> str:
        """Generate unique chunk ID"""
        self.chunk_counter += 1
        return f"{base_name}_{self.chunk_counter}"
    
    def get_processing_summary(self, plan: ProcessingPlan) -> str:
        """Generate human-readable processing summary"""
        summary = f"""
Hierarchical Processing Plan:
===========================
Total Chunks: {len(plan.chunks)}
Estimated Complexity: {plan.estimated_total_complexity:.2f}
Requires Merging: {plan.requires_merging}

Processing Order:
{' → '.join(plan.processing_order)}

Parallel Groups:
"""
        
        for i, group in enumerate(plan.parallel_groups, 1):
            summary += f"  Group {i}: {', '.join(group)}\n"
        
        summary += "\nChunk Details:\n"
        for chunk_id in plan.processing_order:
            chunk = plan.chunks[chunk_id]
            summary += f"  {chunk_id}: {chunk.chunk_type.value} (complexity: {chunk.estimated_complexity:.2f})\n"
            if chunk.dependencies:
                dep_ids = [dep.target_chunk_id for dep in chunk.dependencies]
                summary += f"    Dependencies: {', '.join(dep_ids)}\n"
        
        return summary.strip() 