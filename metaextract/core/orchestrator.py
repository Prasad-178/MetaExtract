"""
Multi-Agent Orchestrator

Coordinates multiple LLM agents to extract structured data from unstructured text.
Implements different orchestration patterns: parallel, sequential, and hybrid.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .schema_analyzer import SchemaMetrics
from .strategy_selector import StrategyConfig, ExtractionStrategy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Different agent roles in the extraction process"""
    EXTRACTOR = "extractor"
    VALIDATOR = "validator"
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"
    CONFIDENCE_ASSESSOR = "confidence_assessor"


@dataclass
class ExtractionTask:
    """A task for an agent to process"""
    task_id: str
    agent_role: AgentRole
    schema_section: Dict[str, Any]
    input_text: str
    context: Dict[str, Any]
    dependencies: List[str]  # IDs of tasks this depends on
    priority: int = 1


@dataclass
class ExtractionResult:
    """Result from an agent's processing"""
    task_id: str
    agent_role: AgentRole
    extracted_data: Dict[str, Any]
    confidence_score: float
    processing_time: float
    model_used: str
    token_usage: Dict[str, int]
    validation_notes: List[str]
    low_confidence_fields: List[str]


@dataclass
class OrchestrationReport:
    """Complete report of the orchestration process"""
    final_result: Dict[str, Any]
    overall_confidence: float
    total_processing_time: float
    strategy_used: str
    agents_used: List[str]
    total_model_calls: int
    total_tokens: int
    low_confidence_fields: List[str]
    validation_notes: List[str]


class MultiAgentOrchestrator:
    """Orchestrates multiple agents for complex extraction tasks"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.active_tasks: Dict[str, ExtractionTask] = {}
        self.completed_tasks: Dict[str, ExtractionResult] = {}
        self.agent_prompts = self._initialize_agent_prompts()
        
    async def process_extraction(self, 
                               input_text: str,
                               schema: Dict[str, Any],
                               strategy_config: StrategyConfig,
                               schema_metrics: SchemaMetrics) -> OrchestrationReport:
        """
        Main orchestration method for processing extraction
        
        Args:
            input_text: Unstructured text to extract from
            schema: JSON schema to extract to
            strategy_config: Selected strategy configuration
            schema_metrics: Schema complexity metrics
            
        Returns:
            OrchestrationReport with results and metrics
        """
        start_time = time.time()
        
        logger.info(f"Starting extraction with strategy: {strategy_config.strategy.value}")
        
        # Route to appropriate orchestration pattern
        if strategy_config.strategy == ExtractionStrategy.SIMPLE_PROMPT:
            result = await self._simple_extraction(input_text, schema, strategy_config)
        elif strategy_config.strategy == ExtractionStrategy.ENHANCED_PROMPT:
            result = await self._enhanced_extraction(input_text, schema, strategy_config)
        elif strategy_config.strategy == ExtractionStrategy.HIERARCHICAL_CHUNKING:
            result = await self._hierarchical_extraction(input_text, schema, strategy_config, schema_metrics)
        elif strategy_config.strategy == ExtractionStrategy.MULTI_AGENT_PARALLEL:
            result = await self._parallel_extraction(input_text, schema, strategy_config, schema_metrics)
        elif strategy_config.strategy == ExtractionStrategy.MULTI_AGENT_SEQUENTIAL:
            result = await self._sequential_extraction(input_text, schema, strategy_config, schema_metrics)
        elif strategy_config.strategy == ExtractionStrategy.HYBRID_APPROACH:
            result = await self._hybrid_extraction(input_text, schema, strategy_config, schema_metrics)
        else:
            raise ValueError(f"Unknown strategy: {strategy_config.strategy}")
        
        total_time = time.time() - start_time
        
        # Create orchestration report
        report = OrchestrationReport(
            final_result=result["extracted_data"],
            overall_confidence=result["confidence_score"],
            total_processing_time=total_time,
            strategy_used=strategy_config.strategy.value,
            agents_used=result.get("agents_used", []),
            total_model_calls=result.get("model_calls", 1),
            total_tokens=result.get("total_tokens", 0),
            low_confidence_fields=result.get("low_confidence_fields", []),
            validation_notes=result.get("validation_notes", [])
        )
        
        logger.info(f"Extraction completed in {total_time:.2f}s with confidence {result['confidence_score']:.2f}")
        
        return report
    
    async def _simple_extraction(self, input_text: str, schema: Dict[str, Any], config: StrategyConfig) -> Dict[str, Any]:
        """Simple single-agent extraction"""
        logger.info("Using simple extraction strategy")
        
        prompt = self._build_extraction_prompt(input_text, schema, "simple")
        
        # Simulate LLM call (replace with actual LLM client)
        result = await self._call_llm(prompt, config)
        
        return {
            "extracted_data": result.get("data", {}),
            "confidence_score": result.get("confidence", 0.8),
            "agents_used": ["extractor"],
            "model_calls": 1,
            "total_tokens": result.get("tokens", 1000),
            "low_confidence_fields": result.get("low_confidence", []),
            "validation_notes": []
        }
    
    async def _enhanced_extraction(self, input_text: str, schema: Dict[str, Any], config: StrategyConfig) -> Dict[str, Any]:
        """Enhanced single-agent extraction with advanced prompting"""
        logger.info("Using enhanced extraction strategy")
        
        # Step 1: Initial extraction
        extraction_prompt = self._build_extraction_prompt(input_text, schema, "enhanced")
        extraction_result = await self._call_llm(extraction_prompt, config)
        
        # Step 2: Validation round
        validation_prompt = self._build_validation_prompt(
            extraction_result.get("data", {}), schema, input_text
        )
        validation_result = await self._call_llm(validation_prompt, config)
        
        # Merge results
        final_data = self._merge_extraction_results([
            extraction_result.get("data", {}),
            validation_result.get("corrections", {})
        ])
        
        return {
            "extracted_data": final_data,
            "confidence_score": min(
                extraction_result.get("confidence", 0.8),
                validation_result.get("confidence", 0.8)
            ),
            "agents_used": ["extractor", "validator"],
            "model_calls": 2,
            "total_tokens": extraction_result.get("tokens", 1000) + validation_result.get("tokens", 800),
            "low_confidence_fields": extraction_result.get("low_confidence", []) + validation_result.get("low_confidence", []),
            "validation_notes": validation_result.get("notes", [])
        }
    
    async def _hierarchical_extraction(self, input_text: str, schema: Dict[str, Any], 
                                     config: StrategyConfig, metrics: SchemaMetrics) -> Dict[str, Any]:
        """Hierarchical chunking extraction"""
        logger.info("Using hierarchical extraction strategy")
        
        # Break schema into chunks
        schema_chunks = self._chunk_schema(schema, config.chunk_size)
        
        results = []
        total_tokens = 0
        model_calls = 0
        
        # Process each chunk
        for i, chunk in enumerate(schema_chunks):
            logger.info(f"Processing schema chunk {i+1}/{len(schema_chunks)}")
            
            prompt = self._build_extraction_prompt(input_text, chunk, "hierarchical")
            result = await self._call_llm(prompt, config)
            
            results.append(result.get("data", {}))
            total_tokens += result.get("tokens", 1000)
            model_calls += 1
        
        # Merge hierarchical results
        final_data = self._merge_hierarchical_results(results)
        
        # Final validation
        validation_prompt = self._build_validation_prompt(final_data, schema, input_text)
        validation_result = await self._call_llm(validation_prompt, config)
        model_calls += 1
        total_tokens += validation_result.get("tokens", 800)
        
        return {
            "extracted_data": self._merge_extraction_results([final_data, validation_result.get("corrections", {})]),
            "confidence_score": 0.85,  # Higher confidence due to validation
            "agents_used": ["chunked_extractor", "validator"],
            "model_calls": model_calls,
            "total_tokens": total_tokens,
            "low_confidence_fields": validation_result.get("low_confidence", []),
            "validation_notes": validation_result.get("notes", [])
        }
    
    async def _parallel_extraction(self, input_text: str, schema: Dict[str, Any], 
                                 config: StrategyConfig, metrics: SchemaMetrics) -> Dict[str, Any]:
        """Multi-agent parallel extraction"""
        logger.info("Using parallel multi-agent extraction strategy")
        
        # Create specialized tasks
        tasks = self._create_parallel_tasks(input_text, schema, config)
        
        # Execute tasks in parallel
        task_results = await asyncio.gather(*[
            self._execute_agent_task(task, config) for task in tasks
        ])
        
        # Merge parallel results
        merged_data = self._merge_parallel_results(task_results)
        
        # Cross-validation
        validation_tasks = self._create_validation_tasks(merged_data, schema, input_text)
        validation_results = await asyncio.gather(*[
            self._execute_agent_task(task, config) for task in validation_tasks
        ])
        
        total_calls = len(tasks) + len(validation_tasks)
        total_tokens = sum(r.get("tokens", 1000) for r in task_results + validation_results)
        
        return {
            "extracted_data": merged_data,
            "confidence_score": self._calculate_consensus_confidence(task_results, validation_results),
            "agents_used": [f"specialist_{i}" for i in range(len(tasks))] + ["validator"],
            "model_calls": total_calls,
            "total_tokens": total_tokens,
            "low_confidence_fields": self._identify_low_confidence_fields(task_results, validation_results),
            "validation_notes": [note for vr in validation_results for note in vr.get("notes", [])]
        }
    
    async def _sequential_extraction(self, input_text: str, schema: Dict[str, Any], 
                                   config: StrategyConfig, metrics: SchemaMetrics) -> Dict[str, Any]:
        """Multi-agent sequential extraction"""
        logger.info("Using sequential multi-agent extraction strategy")
        
        # Pipeline stages
        stages = config.custom_params.get("pipeline_stages", ["extraction", "validation", "refinement"])
        
        current_data = {}
        total_tokens = 0
        model_calls = 0
        validation_notes = []
        
        for stage in stages:
            logger.info(f"Executing pipeline stage: {stage}")
            
            if stage == "extraction":
                prompt = self._build_extraction_prompt(input_text, schema, "sequential")
                result = await self._call_llm(prompt, config)
                current_data = result.get("data", {})
                
            elif stage == "validation":
                prompt = self._build_validation_prompt(current_data, schema, input_text)
                result = await self._call_llm(prompt, config)
                corrections = result.get("corrections", {})
                current_data = self._merge_extraction_results([current_data, corrections])
                validation_notes.extend(result.get("notes", []))
                
            elif stage == "refinement":
                prompt = self._build_refinement_prompt(current_data, schema, input_text)
                result = await self._call_llm(prompt, config)
                refinements = result.get("refinements", {})
                current_data = self._merge_extraction_results([current_data, refinements])
            
            total_tokens += result.get("tokens", 1000)
            model_calls += 1
        
        return {
            "extracted_data": current_data,
            "confidence_score": 0.9,  # Highest confidence due to multiple stages
            "agents_used": [f"agent_{stage}" for stage in stages],
            "model_calls": model_calls,
            "total_tokens": total_tokens,
            "low_confidence_fields": result.get("low_confidence", []),
            "validation_notes": validation_notes
        }
    
    async def _hybrid_extraction(self, input_text: str, schema: Dict[str, Any], 
                                config: StrategyConfig, metrics: SchemaMetrics) -> Dict[str, Any]:
        """Hybrid extraction combining multiple strategies"""
        logger.info("Using hybrid extraction strategy")
        
        # Strategy 1: Parallel extraction for high-level structure
        parallel_result = await self._parallel_extraction(input_text, schema, config, metrics)
        
        # Strategy 2: Sequential refinement for complex fields
        complex_fields = self._identify_complex_fields(schema, metrics)
        sequential_config = config
        sequential_config.custom_params["pipeline_stages"] = ["validation", "refinement"]
        
        sequential_result = await self._sequential_extraction(
            input_text, {"properties": complex_fields}, sequential_config, metrics
        )
        
        # Merge hybrid results with voting
        final_data = self._merge_with_voting([
            parallel_result["extracted_data"],
            sequential_result["extracted_data"]
        ])
        
        total_calls = parallel_result["model_calls"] + sequential_result["model_calls"]
        total_tokens = parallel_result["total_tokens"] + sequential_result["total_tokens"]
        
        return {
            "extracted_data": final_data,
            "confidence_score": max(parallel_result["confidence_score"], sequential_result["confidence_score"]),
            "agents_used": parallel_result["agents_used"] + sequential_result["agents_used"],
            "model_calls": total_calls,
            "total_tokens": total_tokens,
            "low_confidence_fields": list(set(
                parallel_result["low_confidence_fields"] + sequential_result["low_confidence_fields"]
            )),
            "validation_notes": parallel_result["validation_notes"] + sequential_result["validation_notes"]
        }
    
    # Helper methods
    def _initialize_agent_prompts(self) -> Dict[str, str]:
        """Initialize prompts for different agent roles"""
        return {
            "simple": """
Extract structured data from the following text according to the provided JSON schema.
Return only valid JSON that strictly follows the schema.

Text: {input_text}

Schema: {schema}

Extracted Data:""",
            
            "enhanced": """
You are an expert data extraction agent. Extract structured data from the text according to the JSON schema.

Instructions:
1. Read the text carefully
2. Identify all relevant information for each schema field
3. Ensure data types match the schema exactly
4. Mark uncertain extractions with low confidence
5. Return valid JSON only

Text: {input_text}

Schema: {schema}

Think step by step, then provide the extracted data as valid JSON:""",
            
            "validation": """
You are a validation agent. Review the extracted data against the original text and schema.

Original Text: {input_text}
Schema: {schema}
Extracted Data: {extracted_data}

Tasks:
1. Verify accuracy of extracted data
2. Check schema compliance
3. Identify missing or incorrect fields
4. Provide corrections if needed

Return corrections as JSON with confidence scores:""",
            
            "refinement": """
You are a refinement agent. Improve the extracted data quality.

Original Text: {input_text}
Schema: {schema}
Current Data: {current_data}

Tasks:
1. Fill missing optional fields if data is available
2. Improve data quality and formatting
3. Ensure consistency across related fields
4. Maintain strict schema compliance

Return refinements as JSON:"""
        }
    
    async def _call_llm(self, prompt: str, config: StrategyConfig) -> Dict[str, Any]:
        """
        Simulate LLM API call
        In production, this would call actual LLM APIs (OpenAI, Anthropic, etc.)
        """
        # Simulate API delay
        await asyncio.sleep(0.1)
        
        # Mock response based on prompt type
        if "extraction" in prompt.lower() or "extract" in prompt.lower():
            return {
                "data": {"mock": "extracted_data", "confidence": 0.85},
                "confidence": 0.85,
                "tokens": 1200,
                "low_confidence": ["field1", "field2"]
            }
        elif "validation" in prompt.lower():
            return {
                "corrections": {"corrected_field": "new_value"},
                "confidence": 0.9,
                "tokens": 800,
                "notes": ["Fixed field1", "Verified field2"],
                "low_confidence": ["field3"]
            }
        elif "refinement" in prompt.lower():
            return {
                "refinements": {"refined_field": "improved_value"},
                "confidence": 0.95,
                "tokens": 600,
                "low_confidence": []
            }
        else:
            return {
                "data": {"general": "response"},
                "confidence": 0.8,
                "tokens": 1000,
                "low_confidence": []
            }
    
    def _build_extraction_prompt(self, input_text: str, schema: Dict[str, Any], strategy: str) -> str:
        """Build extraction prompt for given strategy"""
        template = self.agent_prompts.get(strategy, self.agent_prompts["simple"])
        return template.format(
            input_text=input_text[:2000],  # Truncate for demo
            schema=json.dumps(schema, indent=2)[:1000]  # Truncate for demo
        )
    
    def _build_validation_prompt(self, extracted_data: Dict[str, Any], schema: Dict[str, Any], input_text: str) -> str:
        """Build validation prompt"""
        template = self.agent_prompts["validation"]
        return template.format(
            input_text=input_text[:1000],
            schema=json.dumps(schema, indent=2)[:500],
            extracted_data=json.dumps(extracted_data, indent=2)[:500]
        )
    
    def _build_refinement_prompt(self, current_data: Dict[str, Any], schema: Dict[str, Any], input_text: str) -> str:
        """Build refinement prompt"""
        template = self.agent_prompts["refinement"]
        return template.format(
            input_text=input_text[:1000],
            schema=json.dumps(schema, indent=2)[:500],
            current_data=json.dumps(current_data, indent=2)[:500]
        )
    
    def _chunk_schema(self, schema: Dict[str, Any], chunk_size: int) -> List[Dict[str, Any]]:
        """Break schema into manageable chunks"""
        # Simplified chunking - in production, this would be more sophisticated
        properties = schema.get("properties", {})
        chunk_keys = list(properties.keys())
        
        chunks = []
        for i in range(0, len(chunk_keys), max(1, len(chunk_keys) // 3)):
            chunk_properties = {k: properties[k] for k in chunk_keys[i:i+3]}
            chunks.append({
                "type": "object",
                "properties": chunk_properties
            })
        
        return chunks if chunks else [schema]
    
    def _create_parallel_tasks(self, input_text: str, schema: Dict[str, Any], config: StrategyConfig) -> List[ExtractionTask]:
        """Create parallel tasks for different schema sections"""
        tasks = []
        properties = schema.get("properties", {})
        
        # Group properties by type/complexity
        simple_props = {}
        complex_props = {}
        
        for key, prop_schema in properties.items():
            if prop_schema.get("type") in ["string", "number", "boolean"]:
                simple_props[key] = prop_schema
            else:
                complex_props[key] = prop_schema
        
        if simple_props:
            tasks.append(ExtractionTask(
                task_id="simple_fields",
                agent_role=AgentRole.SPECIALIST,
                schema_section={"type": "object", "properties": simple_props},
                input_text=input_text,
                context={"focus": "simple_fields"},
                dependencies=[]
            ))
        
        if complex_props:
            tasks.append(ExtractionTask(
                task_id="complex_fields",
                agent_role=AgentRole.SPECIALIST,
                schema_section={"type": "object", "properties": complex_props},
                input_text=input_text,
                context={"focus": "complex_fields"},
                dependencies=[]
            ))
        
        return tasks
    
    def _create_validation_tasks(self, extracted_data: Dict[str, Any], schema: Dict[str, Any], input_text: str) -> List[ExtractionTask]:
        """Create validation tasks"""
        return [ExtractionTask(
            task_id="validation",
            agent_role=AgentRole.VALIDATOR,
            schema_section=schema,
            input_text=input_text,
            context={"extracted_data": extracted_data},
            dependencies=[]
        )]
    
    async def _execute_agent_task(self, task: ExtractionTask, config: StrategyConfig) -> Dict[str, Any]:
        """Execute a single agent task"""
        prompt = self._build_extraction_prompt(task.input_text, task.schema_section, "enhanced")
        return await self._call_llm(prompt, config)
    
    def _merge_extraction_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple extraction results"""
        merged = {}
        for result in results:
            if isinstance(result, dict):
                merged.update(result)
        return merged
    
    def _merge_hierarchical_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge hierarchical chunk results"""
        return self._merge_extraction_results(results)
    
    def _merge_parallel_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge parallel agent results"""
        return self._merge_extraction_results([r.get("data", {}) for r in results])
    
    def _merge_with_voting(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge results using voting mechanism"""
        # Simplified voting - take first non-empty value
        merged = {}
        for result in results:
            for key, value in result.items():
                if key not in merged and value:
                    merged[key] = value
        return merged
    
    def _calculate_consensus_confidence(self, task_results: List[Dict[str, Any]], validation_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence based on agent consensus"""
        all_confidences = []
        for result in task_results + validation_results:
            all_confidences.append(result.get("confidence", 0.8))
        return sum(all_confidences) / len(all_confidences) if all_confidences else 0.8
    
    def _identify_low_confidence_fields(self, task_results: List[Dict[str, Any]], validation_results: List[Dict[str, Any]]) -> List[str]:
        """Identify fields with low confidence across agents"""
        low_confidence = []
        for result in task_results + validation_results:
            low_confidence.extend(result.get("low_confidence", []))
        return list(set(low_confidence))
    
    def _identify_complex_fields(self, schema: Dict[str, Any], metrics: SchemaMetrics) -> Dict[str, Any]:
        """Identify complex fields that need special attention"""
        complex_fields = {}
        properties = schema.get("properties", {})
        
        for key, prop_schema in properties.items():
            if (prop_schema.get("type") == "object" or 
                prop_schema.get("type") == "array" or
                "anyOf" in prop_schema or 
                "oneOf" in prop_schema):
                complex_fields[key] = prop_schema
        
        return complex_fields 