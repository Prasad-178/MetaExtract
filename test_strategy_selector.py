"""
Test script for Adaptive Strategy Selector

Demonstrates how the system selects strategies based on schema complexity.
"""

import json
import sys
sys.path.append('.')

from metaextract.core.schema_analyzer import SchemaComplexityAnalyzer
from metaextract.core.strategy_selector import AdaptiveStrategySelector

def load_schema(filename):
    """Load a JSON schema file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def test_strategy_selection():
    """Test strategy selection for all provided schemas"""
    analyzer = SchemaComplexityAnalyzer()
    selector = AdaptiveStrategySelector()
    
    schemas = [
        ("Resume Schema", "testcases/convert your resume to this schema.json"),
        ("GitHub Actions Schema", "testcases/github_actions_schema.json"), 
        ("Paper Citations Schema", "testcases/paper citations_schema.json")
    ]
    
    print("MetaExtract Adaptive Strategy Selection")
    print("=" * 60)
    
    for name, filename in schemas:
        try:
            print(f"\n🎯 Strategy Selection for: {name}")
            print("-" * 40)
            
            # Analyze schema complexity
            schema = load_schema(filename)
            metrics = analyzer.analyze_schema(schema)
            
            # Select optimal strategy
            strategy_config = selector.select_strategy(metrics)
            
            # Show strategy explanation
            explanation = selector.get_strategy_explanation(strategy_config, metrics)
            print(explanation)
            print()
            
        except Exception as e:
            print(f"❌ Error analyzing {name}: {e}")

def test_different_scenarios():
    """Test strategy selection for different scenarios"""
    analyzer = SchemaComplexityAnalyzer()
    selector = AdaptiveStrategySelector()
    
    print("\n" + "=" * 60)
    print("Testing Different Input Scenarios")
    print("=" * 60)
    
    # Load a complex schema
    schema = load_schema("testcases/paper citations_schema.json")
    metrics = analyzer.analyze_schema(schema)
    
    scenarios = [
        ("Default settings", {}),
        ("Large document (10MB)", {"input_size": 10_000_000}),
        ("Speed preference", {"user_prefs": {"prefer_speed": True}}),
        ("Accuracy preference", {"user_prefs": {"prefer_accuracy": True}})
    ]
    
    for scenario_name, params in scenarios:
        print(f"\n🔧 Scenario: {scenario_name}")
        print("-" * 30)
        
        input_size = params.get("input_size", 0)
        user_prefs = params.get("user_prefs", None)
        
        strategy_config = selector.select_strategy(
            metrics, 
            input_size_bytes=input_size,
            user_preferences=user_prefs
        )
        
        print(f"Selected Strategy: {strategy_config.strategy.value.upper().replace('_', ' ')}")
        print(f"Description: {strategy_config.description}")
        print(f"Parallel Agents: {strategy_config.max_parallel_agents}")
        print(f"Validation Rounds: {strategy_config.validation_rounds}")
        print(f"Use Chunking: {strategy_config.use_chunking}")
        if strategy_config.use_chunking:
            print(f"Chunk Size: {strategy_config.chunk_size} tokens")

if __name__ == "__main__":
    test_strategy_selection()
    test_different_scenarios() 