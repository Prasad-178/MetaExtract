"""
Adaptive Strategy Selector

Selects the optimal extraction strategy based on schema complexity analysis.
Strategies range from simple prompting to multi-agent orchestration.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
import copy

from .schema_analyzer import SchemaMetrics, ComplexityLevel


class ExtractionStrategy(Enum):
    """Available extraction strategies"""
    SIMPLE_PROMPT = "simple_prompt"
    ENHANCED_PROMPT = "enhanced_prompt"
    HIERARCHICAL_CHUNKING = "hierarchical_chunking"
    MULTI_AGENT_PARALLEL = "multi_agent_parallel"
    MULTI_AGENT_SEQUENTIAL = "multi_agent_sequential"
    HYBRID_APPROACH = "hybrid_approach"


@dataclass
class StrategyConfig:
    """Configuration for a specific extraction strategy"""
    strategy: ExtractionStrategy
    description: str
    
    # Processing parameters
    use_chunking: bool
    chunk_size: int
    max_parallel_agents: int
    validation_rounds: int
    
    # Model configuration
    primary_model: str
    fallback_model: Optional[str]
    temperature: float
    max_tokens: int
    
    # Strategy-specific settings
    custom_params: Dict[str, Any]


class AdaptiveStrategySelector:
    """Selects optimal extraction strategy based on schema complexity"""
    
    def __init__(self):
        self.strategy_configs = self._initialize_strategies()
        
        # Strategy selection rules based on complexity metrics
        self.selection_rules = {
            ComplexityLevel.SIMPLE: [
                ExtractionStrategy.SIMPLE_PROMPT,
                ExtractionStrategy.ENHANCED_PROMPT
            ],
            ComplexityLevel.MEDIUM: [
                ExtractionStrategy.ENHANCED_PROMPT,
                ExtractionStrategy.HIERARCHICAL_CHUNKING
            ],
            ComplexityLevel.COMPLEX: [
                ExtractionStrategy.HIERARCHICAL_CHUNKING,
                ExtractionStrategy.MULTI_AGENT_PARALLEL
            ],
            ComplexityLevel.VERY_COMPLEX: [
                ExtractionStrategy.MULTI_AGENT_SEQUENTIAL,
                ExtractionStrategy.HYBRID_APPROACH
            ]
        }
    
    def select_strategy(self, 
                       schema_metrics: SchemaMetrics, 
                       input_size_bytes: int = 0,
                       user_preferences: Optional[Dict[str, Any]] = None) -> StrategyConfig:
        """
        Select the optimal extraction strategy based on complexity analysis
        
        Args:
            schema_metrics: Results from schema complexity analysis
            input_size_bytes: Size of input document in bytes
            user_preferences: Optional user preferences (speed vs accuracy, etc.)
            
        Returns:
            StrategyConfig with optimal strategy and parameters
        """
        # Get candidate strategies based on complexity level
        candidates = self.selection_rules[schema_metrics.complexity_level]
        
        # Apply additional selection criteria
        selected_strategy = self._apply_selection_criteria(
            candidates, schema_metrics, input_size_bytes, user_preferences
        )
        
        # Get base configuration
        config = copy.deepcopy(self.strategy_configs[selected_strategy])
        
        # Customize configuration based on specific metrics
        config = self._customize_config(config, schema_metrics, input_size_bytes)
        
        return config
    
    def _apply_selection_criteria(self, 
                                 candidates: List[ExtractionStrategy],
                                 schema_metrics: SchemaMetrics,
                                 input_size_bytes: int,
                                 user_preferences: Optional[Dict[str, Any]]) -> ExtractionStrategy:
        """Apply additional criteria to select from candidate strategies"""
        
        # Default to first candidate
        selected = candidates[0]
        
        # Large input document handling
        if input_size_bytes > 1_000_000:  # 1MB+
            if ExtractionStrategy.HIERARCHICAL_CHUNKING in candidates:
                selected = ExtractionStrategy.HIERARCHICAL_CHUNKING
            elif ExtractionStrategy.MULTI_AGENT_PARALLEL in candidates:
                selected = ExtractionStrategy.MULTI_AGENT_PARALLEL
        
        # High conditional complexity prefers sequential processing
        if schema_metrics.conditional_schemas > 10:
            if ExtractionStrategy.MULTI_AGENT_SEQUENTIAL in candidates:
                selected = ExtractionStrategy.MULTI_AGENT_SEQUENTIAL
        
        # Deep nesting prefers hierarchical approach
        if schema_metrics.max_nesting_depth > 5:
            if ExtractionStrategy.HIERARCHICAL_CHUNKING in candidates:
                selected = ExtractionStrategy.HIERARCHICAL_CHUNKING
        
        # Large number of objects prefers parallel processing
        if schema_metrics.total_objects > 30:
            if ExtractionStrategy.MULTI_AGENT_PARALLEL in candidates:
                selected = ExtractionStrategy.MULTI_AGENT_PARALLEL
        
        # User preference overrides
        if user_preferences:
            prefer_speed = user_preferences.get('prefer_speed', False)
            prefer_accuracy = user_preferences.get('prefer_accuracy', False)
            
            if prefer_speed and ExtractionStrategy.SIMPLE_PROMPT in candidates:
                selected = ExtractionStrategy.SIMPLE_PROMPT
            elif prefer_accuracy and ExtractionStrategy.HYBRID_APPROACH in candidates:
                selected = ExtractionStrategy.HYBRID_APPROACH
        
        return selected
    
    def _customize_config(self, 
                         config: StrategyConfig, 
                         schema_metrics: SchemaMetrics,
                         input_size_bytes: int) -> StrategyConfig:
        """Customize strategy configuration based on specific metrics"""
        
        # Adjust chunk size based on schema complexity
        if config.use_chunking:
            base_chunk_size = 4000  # tokens
            if schema_metrics.max_nesting_depth > 4:
                config.chunk_size = base_chunk_size // 2
            elif schema_metrics.total_objects > 50:
                config.chunk_size = base_chunk_size * 2
            else:
                config.chunk_size = base_chunk_size
        
        # Adjust parallel agents based on object count
        if config.max_parallel_agents > 1:
            config.max_parallel_agents = min(
                max(schema_metrics.total_objects // 10, 2),
                8  # Cap at 8 agents
            )
        
        # Adjust validation rounds based on complexity
        if schema_metrics.conditional_schemas > 5:
            config.validation_rounds += 1
        if schema_metrics.max_enum_size > 100:
            config.validation_rounds += 1
        
        # Adjust model parameters based on complexity
        if schema_metrics.complexity_score > 100:
            config.temperature = 0.1  # More deterministic for complex schemas
            config.max_tokens = min(config.max_tokens * 2, 8000)
        
        # Large input document adjustments
        if input_size_bytes > 5_000_000:  # 5MB+
            config.use_chunking = True
            config.chunk_size = min(config.chunk_size, 2000)
        
        return config
    
    def _initialize_strategies(self) -> Dict[ExtractionStrategy, StrategyConfig]:
        """Initialize all available strategy configurations"""
        
        strategies = {}
        
        # Simple Prompt Strategy
        strategies[ExtractionStrategy.SIMPLE_PROMPT] = StrategyConfig(
            strategy=ExtractionStrategy.SIMPLE_PROMPT,
            description="Single LLM call with basic schema-guided prompt",
            use_chunking=False,
            chunk_size=0,
            max_parallel_agents=1,
            validation_rounds=1,
            primary_model="gpt-4",
            fallback_model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=4000,
            custom_params={
                "use_examples": True,
                "schema_truncation": False
            }
        )
        
        # Enhanced Prompt Strategy
        strategies[ExtractionStrategy.ENHANCED_PROMPT] = StrategyConfig(
            strategy=ExtractionStrategy.ENHANCED_PROMPT,
            description="Single LLM call with advanced prompting techniques",
            use_chunking=False,
            chunk_size=0,
            max_parallel_agents=1,
            validation_rounds=2,
            primary_model="gpt-4",
            fallback_model="gpt-3.5-turbo",
            temperature=0.2,
            max_tokens=6000,
            custom_params={
                "use_examples": True,
                "schema_simplification": True,
                "step_by_step": True,
                "confidence_scoring": True
            }
        )
        
        # Hierarchical Chunking Strategy
        strategies[ExtractionStrategy.HIERARCHICAL_CHUNKING] = StrategyConfig(
            strategy=ExtractionStrategy.HIERARCHICAL_CHUNKING,
            description="Break schema into chunks, process hierarchically",
            use_chunking=True,
            chunk_size=3000,
            max_parallel_agents=2,
            validation_rounds=2,
            primary_model="gpt-4",
            fallback_model="gpt-3.5-turbo",
            temperature=0.2,
            max_tokens=4000,
            custom_params={
                "preserve_context": True,
                "merge_strategy": "hierarchical",
                "dependency_tracking": True
            }
        )
        
        # Multi-Agent Parallel Strategy
        strategies[ExtractionStrategy.MULTI_AGENT_PARALLEL] = StrategyConfig(
            strategy=ExtractionStrategy.MULTI_AGENT_PARALLEL,
            description="Multiple agents process different schema sections simultaneously",
            use_chunking=True,
            chunk_size=2000,
            max_parallel_agents=4,
            validation_rounds=2,
            primary_model="gpt-4",
            fallback_model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=4000,
            custom_params={
                "agent_specialization": True,
                "cross_validation": True,
                "consensus_threshold": 0.8
            }
        )
        
        # Multi-Agent Sequential Strategy
        strategies[ExtractionStrategy.MULTI_AGENT_SEQUENTIAL] = StrategyConfig(
            strategy=ExtractionStrategy.MULTI_AGENT_SEQUENTIAL,
            description="Multiple agents process schema in dependent sequence",
            use_chunking=True,
            chunk_size=2500,
            max_parallel_agents=3,
            validation_rounds=3,
            primary_model="gpt-4",
            fallback_model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=5000,
            custom_params={
                "pipeline_stages": ["extraction", "validation", "refinement"],
                "inter_agent_feedback": True,
                "progressive_enhancement": True
            }
        )
        
        # Hybrid Approach Strategy
        strategies[ExtractionStrategy.HYBRID_APPROACH] = StrategyConfig(
            strategy=ExtractionStrategy.HYBRID_APPROACH,
            description="Adaptive combination of multiple strategies",
            use_chunking=True,
            chunk_size=3000,
            max_parallel_agents=5,
            validation_rounds=3,
            primary_model="gpt-4",
            fallback_model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=6000,
            custom_params={
                "adaptive_switching": True,
                "multi_strategy_voting": True,
                "quality_monitoring": True,
                "auto_fallback": True
            }
        )
        
        return strategies
    
    def get_strategy_explanation(self, config: StrategyConfig, schema_metrics: SchemaMetrics) -> str:
        """Generate human-readable explanation of selected strategy"""
        
        explanation = f"""
Strategy Selection: {config.strategy.value.upper().replace('_', ' ')}
=================================================================

Description: {config.description}

Why this strategy was chosen:
- Schema Complexity: {schema_metrics.complexity_level.value.upper()} (score: {schema_metrics.complexity_score:.2f})
- Nesting Depth: {schema_metrics.max_nesting_depth} levels
- Total Objects: {schema_metrics.total_objects}
- Conditional Schemas: {schema_metrics.conditional_schemas}

Processing Configuration:
- Use Chunking: {config.use_chunking}
- Chunk Size: {config.chunk_size} tokens
- Parallel Agents: {config.max_parallel_agents}
- Validation Rounds: {config.validation_rounds}
- Primary Model: {config.primary_model}
- Temperature: {config.temperature}

Estimated Performance:
- Model Calls: {schema_metrics.estimated_model_calls}
- Processing Time: {'High' if schema_metrics.complexity_score > 100 else 'Medium' if schema_metrics.complexity_score > 50 else 'Low'}
- Accuracy: {'Very High' if config.validation_rounds >= 3 else 'High' if config.validation_rounds >= 2 else 'Good'}
        """.strip()
        
        return explanation 