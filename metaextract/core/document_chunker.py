"""
Large Document Chunker

Handles massive input documents (50-page docs to 10MB files) by intelligently 
splitting them into manageable chunks while preserving context and relationships.
"""

import re
import json
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of document chunks"""
    PARAGRAPH = "paragraph"
    SECTION = "section"
    TABLE = "table"
    LIST = "list"
    HEADER = "header"
    METADATA = "metadata"
    CONTEXT = "context"


class ChunkPriority(Enum):
    """Priority levels for chunks"""
    CRITICAL = 1    # Contains essential extraction targets
    HIGH = 2        # Likely contains relevant information
    MEDIUM = 3      # May contain relevant information
    LOW = 4         # Unlikely to contain relevant information


@dataclass
class DocumentChunk:
    """A chunk of a larger document"""
    chunk_id: str
    chunk_type: ChunkType
    content: str
    priority: ChunkPriority
    start_position: int
    end_position: int
    
    # Context information
    preceding_context: str = ""
    following_context: str = ""
    section_context: str = ""
    
    # Relationships
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)
    related_chunk_ids: List[str] = field(default_factory=list)
    
    # Metadata
    estimated_tokens: int = 0
    contains_structured_data: bool = False
    confidence_score: float = 1.0
    
    # Schema-specific information
    relevant_schema_paths: List[str] = field(default_factory=list)


@dataclass
class ChunkingPlan:
    """Plan for processing document chunks"""
    chunks: Dict[str, DocumentChunk]
    processing_order: List[str]
    priority_groups: Dict[ChunkPriority, List[str]]
    total_chunks: int
    total_tokens: int
    requires_overlap_handling: bool
    context_preservation_strategy: str


class LargeDocumentChunker:
    """Chunks large documents intelligently while preserving context"""
    
    def __init__(self, 
                 max_chunk_size: int = 3000,  # tokens
                 overlap_size: int = 200,     # tokens
                 context_window: int = 500):   # tokens
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.context_window = context_window
        self.chunk_counter = 0
        
        # Patterns for different content types
        self.patterns = {
            'header': re.compile(r'^(#{1,6}\s+.+|.+\n[=-]{3,})', re.MULTILINE),
            'list_item': re.compile(r'^\s*[-*+•]\s+|^\s*\d+\.\s+', re.MULTILINE),
            'table_row': re.compile(r'\|.*\|', re.MULTILINE),
            'email_header': re.compile(r'^(From|To|Subject|Date|CC|BCC):\s+', re.MULTILINE | re.IGNORECASE),
            'json_block': re.compile(r'\{[\s\S]*?\}', re.MULTILINE),
            'xml_tag': re.compile(r'<[^>]+>'),
            'code_block': re.compile(r'```[\s\S]*?```|`[^`]+`'),
            'url': re.compile(r'https?://[^\s]+'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'date': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b')
        }
        
        # Token estimation (rough approximation)
        self.avg_chars_per_token = 4
    
    def chunk_document(self, 
                      text: str, 
                      schema: Optional[Dict[str, Any]] = None,
                      document_type: str = "text") -> ChunkingPlan:
        """
        Chunk a large document intelligently
        
        Args:
            text: The document text to chunk
            schema: Optional JSON schema to guide chunking
            document_type: Type of document (text, markdown, email, etc.)
            
        Returns:
            ChunkingPlan with chunks and processing strategy
        """
        logger.info(f"Chunking document: {len(text)} characters, type: {document_type}")
        
        # Reset state
        self.chunk_counter = 0
        chunks = {}
        
        # Step 1: Analyze document structure
        document_structure = self._analyze_document_structure(text, document_type)
        
        # Step 2: Identify schema-relevant sections
        schema_guidance = self._analyze_schema_relevance(text, schema) if schema else {}
        
        # Step 3: Create initial chunks based on structure
        initial_chunks = self._create_structural_chunks(text, document_structure, schema_guidance)
        chunks.update(initial_chunks)
        
        # Step 4: Split large chunks if needed
        refined_chunks = self._refine_chunk_sizes(chunks)
        chunks.update(refined_chunks)
        
        # Step 5: Add context to chunks
        self._add_context_information(chunks, text)
        
        # Step 6: Identify relationships between chunks
        self._identify_chunk_relationships(chunks)
        
        # Step 7: Calculate processing order and priorities
        processing_order = self._calculate_processing_order(chunks)
        priority_groups = self._group_by_priority(chunks)
        
        # Step 8: Calculate metrics
        total_tokens = sum(chunk.estimated_tokens for chunk in chunks.values())
        requires_overlap = any(chunk.estimated_tokens > self.max_chunk_size for chunk in chunks.values())
        
        plan = ChunkingPlan(
            chunks=chunks,
            processing_order=processing_order,
            priority_groups=priority_groups,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            requires_overlap_handling=requires_overlap,
            context_preservation_strategy="hierarchical_overlap"
        )
        
        logger.info(f"Document chunked: {len(chunks)} chunks, {total_tokens} total tokens")
        
        return plan
    
    def _analyze_document_structure(self, text: str, document_type: str) -> Dict[str, Any]:
        """Analyze the structure of the document"""
        structure = {
            'type': document_type,
            'total_length': len(text),
            'estimated_tokens': len(text) // self.avg_chars_per_token,
            'sections': [],
            'headers': [],
            'lists': [],
            'tables': [],
            'structured_data': [],
            'metadata': {}
        }
        
        # Find headers
        header_matches = list(self.patterns['header'].finditer(text))
        structure['headers'] = [(m.start(), m.end(), m.group().strip()) for m in header_matches]
        
        # Find lists
        list_matches = list(self.patterns['list_item'].finditer(text))
        structure['lists'] = [(m.start(), m.end()) for m in list_matches]
        
        # Find tables
        table_matches = list(self.patterns['table_row'].finditer(text))
        if table_matches:
            structure['tables'] = self._group_table_rows(table_matches)
        
        # Find structured data (JSON, XML)
        json_matches = list(self.patterns['json_block'].finditer(text))
        structure['structured_data'].extend([(m.start(), m.end(), 'json') for m in json_matches])
        
        # Find email headers (if email document)
        if document_type == 'email':
            email_headers = list(self.patterns['email_header'].finditer(text))
            structure['metadata']['email_headers'] = [(m.start(), m.end(), m.group().strip()) for m in email_headers]
        
        # Identify sections based on headers
        structure['sections'] = self._identify_sections(text, structure['headers'])
        
        return structure
    
    def _analyze_schema_relevance(self, text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze which parts of the document are relevant to the schema"""
        guidance = {
            'high_relevance_patterns': [],
            'field_mappings': {},
            'extraction_hints': []
        }
        
        if not schema:
            return guidance
        
        # Extract field names and descriptions from schema
        field_info = self._extract_schema_fields(schema)
        
        for field_path, field_data in field_info.items():
            field_name = field_path.split('.')[-1]
            field_description = field_data.get('description', '')
            field_type = field_data.get('type', 'string')
            
            # Create patterns to find relevant content
            relevance_patterns = self._create_relevance_patterns(field_name, field_description, field_type)
            guidance['field_mappings'][field_path] = {
                'patterns': relevance_patterns,
                'type': field_type,
                'priority': self._calculate_field_priority(field_data)
            }
        
        return guidance
    
    def _extract_schema_fields(self, schema: Dict[str, Any], path: str = "") -> Dict[str, Dict[str, Any]]:
        """Extract all fields from a JSON schema with their paths"""
        fields = {}
        
        def extract_recursive(node: Dict[str, Any], current_path: str):
            if not isinstance(node, dict):
                return
            
            node_type = node.get('type')
            
            if node_type == 'object':
                properties = node.get('properties', {})
                for prop_name, prop_schema in properties.items():
                    prop_path = f"{current_path}.{prop_name}" if current_path else prop_name
                    fields[prop_path] = prop_schema
                    extract_recursive(prop_schema, prop_path)
            
            elif node_type == 'array':
                items = node.get('items', {})
                if isinstance(items, dict):
                    items_path = f"{current_path}[]"
                    extract_recursive(items, items_path)
        
        extract_recursive(schema, path)
        return fields
    
    def _create_relevance_patterns(self, field_name: str, description: str, field_type: str) -> List[str]:
        """Create patterns to find content relevant to a schema field"""
        patterns = []
        
        # Basic field name patterns
        patterns.append(rf'\b{re.escape(field_name)}\b')
        patterns.append(rf'\b{re.escape(field_name.replace("_", " "))}\b')
        patterns.append(rf'\b{re.escape(field_name.replace("-", " "))}\b')
        
        # Type-specific patterns
        if field_type in ['string', 'text']:
            if 'email' in field_name.lower() or 'email' in description.lower():
                patterns.append(self.patterns['email'].pattern)
            if 'phone' in field_name.lower() or 'phone' in description.lower():
                patterns.append(self.patterns['phone'].pattern)
            if 'url' in field_name.lower() or 'url' in description.lower():
                patterns.append(self.patterns['url'].pattern)
            if 'date' in field_name.lower() or 'date' in description.lower():
                patterns.append(self.patterns['date'].pattern)
        
        # Extract keywords from description
        if description:
            keywords = re.findall(r'\b[a-zA-Z]{3,}\b', description.lower())
            for keyword in keywords[:5]:  # Limit to top 5 keywords
                patterns.append(rf'\b{re.escape(keyword)}\b')
        
        return patterns
    
    def _calculate_field_priority(self, field_data: Dict[str, Any]) -> ChunkPriority:
        """Calculate priority for a schema field"""
        # Required fields get higher priority
        if field_data.get('required', False):
            return ChunkPriority.CRITICAL
        
        # Fields with examples or enums get higher priority
        if 'examples' in field_data or 'enum' in field_data:
            return ChunkPriority.HIGH
        
        # Simple types get medium priority
        if field_data.get('type') in ['string', 'number', 'integer', 'boolean']:
            return ChunkPriority.MEDIUM
        
        return ChunkPriority.LOW
    
    def _create_structural_chunks(self, 
                                 text: str, 
                                 structure: Dict[str, Any], 
                                 schema_guidance: Dict[str, Any]) -> Dict[str, DocumentChunk]:
        """Create initial chunks based on document structure"""
        chunks = {}
        
        # If document is small enough, create a single chunk
        if structure['estimated_tokens'] <= self.max_chunk_size:
            chunk = self._create_single_chunk(text, schema_guidance)
            chunks[chunk.chunk_id] = chunk
            return chunks
        
        # Create chunks based on sections
        if structure['sections']:
            section_chunks = self._create_section_chunks(text, structure['sections'], schema_guidance)
            chunks.update(section_chunks)
        else:
            # Fall back to paragraph-based chunking
            paragraph_chunks = self._create_paragraph_chunks(text, schema_guidance)
            chunks.update(paragraph_chunks)
        
        # Handle special content types
        if structure['tables']:
            table_chunks = self._create_table_chunks(text, structure['tables'])
            chunks.update(table_chunks)
        
        if structure['structured_data']:
            data_chunks = self._create_structured_data_chunks(text, structure['structured_data'])
            chunks.update(data_chunks)
        
        return chunks
    
    def _create_single_chunk(self, text: str, schema_guidance: Dict[str, Any]) -> DocumentChunk:
        """Create a single chunk for small documents"""
        chunk_id = self._generate_chunk_id("full_document")
        
        # Calculate relevance score
        relevance_score = self._calculate_text_relevance(text, schema_guidance)
        priority = ChunkPriority.HIGH if relevance_score > 0.5 else ChunkPriority.MEDIUM
        
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            chunk_type=ChunkType.SECTION,
            content=text,
            priority=priority,
            start_position=0,
            end_position=len(text),
            estimated_tokens=len(text) // self.avg_chars_per_token,
            contains_structured_data=bool(self.patterns['json_block'].search(text)),
            confidence_score=relevance_score
        )
        
        return chunk
    
    def _create_section_chunks(self, 
                              text: str, 
                              sections: List[Tuple[int, int, str]], 
                              schema_guidance: Dict[str, Any]) -> Dict[str, DocumentChunk]:
        """Create chunks based on document sections"""
        chunks = {}
        
        for i, (start, end, title) in enumerate(sections):
            section_text = text[start:end]
            chunk_id = self._generate_chunk_id(f"section_{i}")
            
            relevance_score = self._calculate_text_relevance(section_text, schema_guidance)
            priority = self._determine_chunk_priority(section_text, relevance_score)
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                chunk_type=ChunkType.SECTION,
                content=section_text,
                priority=priority,
                start_position=start,
                end_position=end,
                section_context=title,
                estimated_tokens=len(section_text) // self.avg_chars_per_token,
                contains_structured_data=bool(self.patterns['json_block'].search(section_text)),
                confidence_score=relevance_score
            )
            
            chunks[chunk_id] = chunk
        
        return chunks
    
    def _create_paragraph_chunks(self, text: str, schema_guidance: Dict[str, Any]) -> Dict[str, DocumentChunk]:
        """Create chunks based on paragraphs"""
        chunks = {}
        paragraphs = text.split('\n\n')
        current_chunk_text = ""
        current_start = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            potential_chunk = current_chunk_text + "\n\n" + paragraph if current_chunk_text else paragraph
            potential_tokens = len(potential_chunk) // self.avg_chars_per_token
            
            if potential_tokens > self.max_chunk_size and current_chunk_text:
                # Create chunk with current content
                chunk_id = self._generate_chunk_id("paragraph_chunk")
                relevance_score = self._calculate_text_relevance(current_chunk_text, schema_guidance)
                priority = self._determine_chunk_priority(current_chunk_text, relevance_score)
                
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_type=ChunkType.PARAGRAPH,
                    content=current_chunk_text,
                    priority=priority,
                    start_position=current_start,
                    end_position=current_start + len(current_chunk_text),
                    estimated_tokens=len(current_chunk_text) // self.avg_chars_per_token,
                    contains_structured_data=bool(self.patterns['json_block'].search(current_chunk_text)),
                    confidence_score=relevance_score
                )
                
                chunks[chunk_id] = chunk
                
                # Start new chunk
                current_chunk_text = paragraph
                current_start = current_start + len(current_chunk_text) + 2  # +2 for \n\n
            else:
                # Add to current chunk
                current_chunk_text = potential_chunk
        
        # Handle remaining content
        if current_chunk_text:
            chunk_id = self._generate_chunk_id("paragraph_chunk")
            relevance_score = self._calculate_text_relevance(current_chunk_text, schema_guidance)
            priority = self._determine_chunk_priority(current_chunk_text, relevance_score)
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                chunk_type=ChunkType.PARAGRAPH,
                content=current_chunk_text,
                priority=priority,
                start_position=current_start,
                end_position=current_start + len(current_chunk_text),
                estimated_tokens=len(current_chunk_text) // self.avg_chars_per_token,
                contains_structured_data=bool(self.patterns['json_block'].search(current_chunk_text)),
                confidence_score=relevance_score
            )
            
            chunks[chunk_id] = chunk
        
        return chunks
    
    def _create_table_chunks(self, text: str, tables: List[Tuple[int, int]]) -> Dict[str, DocumentChunk]:
        """Create chunks for tables"""
        chunks = {}
        
        for i, (start, end) in enumerate(tables):
            table_text = text[start:end]
            chunk_id = self._generate_chunk_id(f"table_{i}")
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                chunk_type=ChunkType.TABLE,
                content=table_text,
                priority=ChunkPriority.HIGH,  # Tables often contain structured data
                start_position=start,
                end_position=end,
                estimated_tokens=len(table_text) // self.avg_chars_per_token,
                contains_structured_data=True,
                confidence_score=0.8
            )
            
            chunks[chunk_id] = chunk
        
        return chunks
    
    def _create_structured_data_chunks(self, text: str, structured_data: List[Tuple[int, int, str]]) -> Dict[str, DocumentChunk]:
        """Create chunks for structured data (JSON, XML, etc.)"""
        chunks = {}
        
        for i, (start, end, data_type) in enumerate(structured_data):
            data_text = text[start:end]
            chunk_id = self._generate_chunk_id(f"{data_type}_{i}")
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                chunk_type=ChunkType.METADATA,
                content=data_text,
                priority=ChunkPriority.CRITICAL,  # Structured data is very important
                start_position=start,
                end_position=end,
                estimated_tokens=len(data_text) // self.avg_chars_per_token,
                contains_structured_data=True,
                confidence_score=0.95
            )
            
            chunks[chunk_id] = chunk
        
        return chunks
    
    def _refine_chunk_sizes(self, chunks: Dict[str, DocumentChunk]) -> Dict[str, DocumentChunk]:
        """Refine chunk sizes by splitting large chunks"""
        refined_chunks = {}
        
        for chunk_id, chunk in chunks.items():
            if chunk.estimated_tokens <= self.max_chunk_size:
                # Chunk is fine as is
                continue
            
            # Split large chunk
            sub_chunks = self._split_large_chunk(chunk)
            refined_chunks.update(sub_chunks)
        
        return refined_chunks
    
    def _split_large_chunk(self, chunk: DocumentChunk) -> Dict[str, DocumentChunk]:
        """Split a large chunk into smaller ones"""
        sub_chunks = {}
        content = chunk.content
        
        # Try to split on sentence boundaries first
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        current_text = ""
        sub_chunk_index = 0
        
        for sentence in sentences:
            potential_text = current_text + " " + sentence if current_text else sentence
            potential_tokens = len(potential_text) // self.avg_chars_per_token
            
            if potential_tokens > self.max_chunk_size - self.overlap_size and current_text:
                # Create sub-chunk
                sub_chunk_id = f"{chunk.chunk_id}_sub_{sub_chunk_index}"
                
                sub_chunk = DocumentChunk(
                    chunk_id=sub_chunk_id,
                    chunk_type=chunk.chunk_type,
                    content=current_text,
                    priority=chunk.priority,
                    start_position=chunk.start_position + len(content) - len(current_text + " ".join(sentences[sentences.index(sentence):])),
                    end_position=chunk.start_position + len(content) - len(" ".join(sentences[sentences.index(sentence):])),
                    section_context=chunk.section_context,
                    parent_chunk_id=chunk.chunk_id,
                    estimated_tokens=len(current_text) // self.avg_chars_per_token,
                    contains_structured_data=chunk.contains_structured_data,
                    confidence_score=chunk.confidence_score
                )
                
                sub_chunks[sub_chunk_id] = sub_chunk
                
                # Start new sub-chunk with overlap
                overlap_text = current_text[-self.overlap_size * self.avg_chars_per_token:]
                current_text = overlap_text + " " + sentence
                sub_chunk_index += 1
            else:
                current_text = potential_text
        
        # Handle remaining text
        if current_text:
            sub_chunk_id = f"{chunk.chunk_id}_sub_{sub_chunk_index}"
            
            sub_chunk = DocumentChunk(
                chunk_id=sub_chunk_id,
                chunk_type=chunk.chunk_type,
                content=current_text,
                priority=chunk.priority,
                start_position=chunk.end_position - len(current_text),
                end_position=chunk.end_position,
                section_context=chunk.section_context,
                parent_chunk_id=chunk.chunk_id,
                estimated_tokens=len(current_text) // self.avg_chars_per_token,
                contains_structured_data=chunk.contains_structured_data,
                confidence_score=chunk.confidence_score
            )
            
            sub_chunks[sub_chunk_id] = sub_chunk
        
        return sub_chunks
    
    def _add_context_information(self, chunks: Dict[str, DocumentChunk], full_text: str):
        """Add context information to chunks"""
        sorted_chunks = sorted(chunks.values(), key=lambda c: c.start_position)
        
        for i, chunk in enumerate(sorted_chunks):
            # Add preceding context
            context_start = max(0, chunk.start_position - self.context_window * self.avg_chars_per_token)
            chunk.preceding_context = full_text[context_start:chunk.start_position]
            
            # Add following context
            context_end = min(len(full_text), chunk.end_position + self.context_window * self.avg_chars_per_token)
            chunk.following_context = full_text[chunk.end_position:context_end]
    
    def _identify_chunk_relationships(self, chunks: Dict[str, DocumentChunk]):
        """Identify relationships between chunks"""
        chunk_list = list(chunks.values())
        
        for i, chunk in enumerate(chunk_list):
            # Sequential relationships
            if i > 0:
                prev_chunk = chunk_list[i-1]
                if abs(chunk.start_position - prev_chunk.end_position) < 100:  # Close chunks
                    chunk.related_chunk_ids.append(prev_chunk.chunk_id)
            
            if i < len(chunk_list) - 1:
                next_chunk = chunk_list[i+1]
                if abs(next_chunk.start_position - chunk.end_position) < 100:
                    chunk.related_chunk_ids.append(next_chunk.chunk_id)
            
            # Content similarity relationships
            for other_chunk in chunk_list:
                if other_chunk.chunk_id != chunk.chunk_id:
                    similarity = self._calculate_content_similarity(chunk.content, other_chunk.content)
                    if similarity > 0.7:  # High similarity threshold
                        chunk.related_chunk_ids.append(other_chunk.chunk_id)
    
    def _calculate_processing_order(self, chunks: Dict[str, DocumentChunk]) -> List[str]:
        """Calculate optimal processing order"""
        # Sort by priority first, then by position
        sorted_chunks = sorted(
            chunks.values(),
            key=lambda c: (c.priority.value, c.start_position)
        )
        
        return [chunk.chunk_id for chunk in sorted_chunks]
    
    def _group_by_priority(self, chunks: Dict[str, DocumentChunk]) -> Dict[ChunkPriority, List[str]]:
        """Group chunks by priority level"""
        groups = {priority: [] for priority in ChunkPriority}
        
        for chunk in chunks.values():
            groups[chunk.priority].append(chunk.chunk_id)
        
        return groups
    
    # Helper methods
    def _group_table_rows(self, matches: List) -> List[Tuple[int, int]]:
        """Group consecutive table rows"""
        if not matches:
            return []
        
        tables = []
        current_start = matches[0].start()
        current_end = matches[0].end()
        
        for i in range(1, len(matches)):
            match = matches[i]
            if match.start() - current_end < 50:  # Rows are close together
                current_end = match.end()
            else:
                tables.append((current_start, current_end))
                current_start = match.start()
                current_end = match.end()
        
        tables.append((current_start, current_end))
        return tables
    
    def _identify_sections(self, text: str, headers: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
        """Identify sections based on headers"""
        if not headers:
            return [(0, len(text), "Full Document")]
        
        sections = []
        for i, (start, end, title) in enumerate(headers):
            section_start = start
            if i < len(headers) - 1:
                section_end = headers[i + 1][0]
            else:
                section_end = len(text)
            
            sections.append((section_start, section_end, title))
        
        return sections
    
    def _calculate_text_relevance(self, text: str, schema_guidance: Dict[str, Any]) -> float:
        """Calculate how relevant text is to the schema"""
        if not schema_guidance or not schema_guidance.get('field_mappings'):
            return 0.5  # Default relevance
        
        relevance_score = 0.0
        total_fields = len(schema_guidance['field_mappings'])
        
        for field_path, field_info in schema_guidance['field_mappings'].items():
            patterns = field_info.get('patterns', [])
            field_relevance = 0.0
            
            for pattern in patterns:
                try:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        field_relevance = min(1.0, len(matches) * 0.2)
                        break
                except re.error:
                    continue
            
            relevance_score += field_relevance
        
        return min(1.0, relevance_score / total_fields) if total_fields > 0 else 0.5
    
    def _determine_chunk_priority(self, text: str, relevance_score: float) -> ChunkPriority:
        """Determine chunk priority based on content and relevance"""
        if relevance_score > 0.8:
            return ChunkPriority.CRITICAL
        elif relevance_score > 0.6:
            return ChunkPriority.HIGH
        elif relevance_score > 0.3:
            return ChunkPriority.MEDIUM
        else:
            return ChunkPriority.LOW
    
    def _calculate_content_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text chunks"""
        # Simple word-based similarity
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_chunk_id(self, base_name: str) -> str:
        """Generate unique chunk ID"""
        self.chunk_counter += 1
        return f"{base_name}_{self.chunk_counter}"
    
    def get_chunking_summary(self, plan: ChunkingPlan) -> str:
        """Generate human-readable chunking summary"""
        summary = f"""
Large Document Chunking Summary:
===============================
Total Chunks: {plan.total_chunks}
Total Tokens: {plan.total_tokens:,}
Requires Overlap Handling: {plan.requires_overlap_handling}
Context Preservation: {plan.context_preservation_strategy}

Priority Distribution:
"""
        
        for priority, chunk_ids in plan.priority_groups.items():
            if chunk_ids:
                total_tokens = sum(plan.chunks[cid].estimated_tokens for cid in chunk_ids)
                summary += f"  {priority.name}: {len(chunk_ids)} chunks ({total_tokens:,} tokens)\n"
        
        summary += f"\nProcessing Order (first 10):\n"
        for i, chunk_id in enumerate(plan.processing_order[:10], 1):
            chunk = plan.chunks[chunk_id]
            summary += f"  {i}. {chunk_id} ({chunk.chunk_type.value}, {chunk.priority.name}, {chunk.estimated_tokens} tokens)\n"
        
        if len(plan.processing_order) > 10:
            summary += f"  ... and {len(plan.processing_order) - 10} more chunks\n"
        
        return summary.strip() 