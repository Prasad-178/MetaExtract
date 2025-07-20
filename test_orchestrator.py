"""
Test script for Multi-Agent Orchestrator

Demonstrates complete extraction pipeline from unstructured text to structured JSON.
"""

import json
import asyncio
import sys
sys.path.append('.')

from metaextract.core.schema_analyzer import SchemaComplexityAnalyzer
from metaextract.core.strategy_selector import AdaptiveStrategySelector
from metaextract.core.orchestrator import MultiAgentOrchestrator

def load_schema(filename):
    """Load a JSON schema file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_input_text(filename):
    """Load input text file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

async def test_complete_pipeline():
    """Test the complete extraction pipeline"""
    print("MetaExtract: Complete Extraction Pipeline Test")
    print("=" * 60)
    
    # Initialize components
    analyzer = SchemaComplexityAnalyzer()
    selector = AdaptiveStrategySelector()
    orchestrator = MultiAgentOrchestrator()
    
    # Test cases
    test_cases = [
        {
            "name": "GitHub Actions Extraction",
            "schema_file": "testcases/github_actions_schema.json",
            "input_file": "testcases/github actions sample input.md",
            "description": "Extract GitHub Action metadata from natural language requirements"
        },
        # Note: We don't have resume input text, so we'll create a mock one
        {
            "name": "Resume Extraction (Mock)",
            "schema_file": "testcases/convert your resume to this schema.json",
            "input_text": """
            John Doe
            Software Engineer
            Email: john.doe@email.com
            Phone: (555) 123-4567
            
            Experience:
            - Senior Software Engineer at Tech Corp (2020-2023)
              Developed scalable web applications using React and Node.js
              Led a team of 5 developers
              Increased system performance by 40%
            
            - Software Developer at StartupXYZ (2018-2020)
              Built mobile applications using React Native
              Implemented CI/CD pipelines
            
            Education:
            - BS Computer Science, University of Technology (2014-2018)
              GPA: 3.8/4.0
            
            Skills: JavaScript, Python, React, Node.js, AWS, Docker
            """,
            "description": "Extract structured resume data from unstructured text"
        }
    ]
    
    for test_case in test_cases:
        await run_extraction_test(test_case, analyzer, selector, orchestrator)

async def run_extraction_test(test_case, analyzer, selector, orchestrator):
    """Run a single extraction test"""
    print(f"\n🚀 Test Case: {test_case['name']}")
    print(f"Description: {test_case['description']}")
    print("-" * 50)
    
    try:
        # Load schema
        schema = load_schema(test_case['schema_file'])
        
        # Load or use provided input text
        if 'input_file' in test_case:
            input_text = load_input_text(test_case['input_file'])
        else:
            input_text = test_case['input_text']
        
        print(f"📄 Input text length: {len(input_text)} characters")
        
        # Step 1: Analyze schema complexity
        print("\n📊 Step 1: Analyzing Schema Complexity...")
        metrics = analyzer.analyze_schema(schema)
        print(f"   Complexity Level: {metrics.complexity_level.value.upper()}")
        print(f"   Complexity Score: {metrics.complexity_score:.2f}")
        print(f"   Estimated Model Calls: {metrics.estimated_model_calls}")
        
        # Step 2: Select strategy
        print("\n🎯 Step 2: Selecting Extraction Strategy...")
        strategy_config = selector.select_strategy(metrics, len(input_text.encode('utf-8')))
        print(f"   Selected Strategy: {strategy_config.strategy.value.upper().replace('_', ' ')}")
        print(f"   Parallel Agents: {strategy_config.max_parallel_agents}")
        print(f"   Validation Rounds: {strategy_config.validation_rounds}")
        
        # Step 3: Execute extraction
        print("\n⚙️  Step 3: Executing Multi-Agent Extraction...")
        report = await orchestrator.process_extraction(
            input_text=input_text,
            schema=schema,
            strategy_config=strategy_config,
            schema_metrics=metrics
        )
        
        # Step 4: Display results
        print("\n📋 Step 4: Extraction Results")
        print(f"   Overall Confidence: {report.overall_confidence:.2f}")
        print(f"   Processing Time: {report.total_processing_time:.2f}s")
        print(f"   Total Model Calls: {report.total_model_calls}")
        print(f"   Total Tokens Used: {report.total_tokens:,}")
        print(f"   Agents Used: {', '.join(report.agents_used)}")
        
        if report.low_confidence_fields:
            print(f"   🚨 Low Confidence Fields: {', '.join(report.low_confidence_fields)}")
        
        if report.validation_notes:
            print(f"   📝 Validation Notes: {len(report.validation_notes)} notes")
        
        # Show sample of extracted data
        print(f"\n📦 Sample Extracted Data:")
        sample_data = dict(list(report.final_result.items())[:3])  # First 3 fields
        for key, value in sample_data.items():
            print(f"   {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
        
        print(f"\n✅ Extraction completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in extraction test: {e}")
        import traceback
        traceback.print_exc()

async def test_strategy_comparison():
    """Compare different strategies on the same input"""
    print("\n" + "=" * 60)
    print("Strategy Comparison Test")
    print("=" * 60)
    
    orchestrator = MultiAgentOrchestrator()
    
    # Simple test schema
    simple_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "title": {"type": "string"},
            "email": {"type": "string"},
            "skills": {"type": "array", "items": {"type": "string"}}
        }
    }
    
    input_text = "John Smith is a Senior Developer at TechCorp. His email is john@tech.com. He specializes in Python, JavaScript, and Machine Learning."
    
    # Test different strategies
    from metaextract.core.strategy_selector import ExtractionStrategy, StrategyConfig
    
    strategies_to_test = [
        ("Simple Prompt", ExtractionStrategy.SIMPLE_PROMPT),
        ("Enhanced Prompt", ExtractionStrategy.ENHANCED_PROMPT),
        ("Multi-Agent Sequential", ExtractionStrategy.MULTI_AGENT_SEQUENTIAL)
    ]
    
    for strategy_name, strategy_enum in strategies_to_test:
        print(f"\n🔄 Testing: {strategy_name}")
        print("-" * 30)
        
        # Create mock config
        config = StrategyConfig(
            strategy=strategy_enum,
            description=f"Testing {strategy_name}",
            use_chunking=False,
            chunk_size=0,
            max_parallel_agents=1,
            validation_rounds=1,
            primary_model="gpt-4",
            fallback_model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=4000,
            custom_params={}
        )
        
        report = await orchestrator.process_extraction(
            input_text=input_text,
            schema=simple_schema,
            strategy_config=config,
            schema_metrics=None  # Not needed for this test
        )
        
        print(f"   Confidence: {report.overall_confidence:.2f}")
        print(f"   Time: {report.total_processing_time:.2f}s")
        print(f"   Model Calls: {report.total_model_calls}")
        print(f"   Extracted: {report.final_result}")

async def main():
    """Main test function"""
    await test_complete_pipeline()
    await test_strategy_comparison()

if __name__ == "__main__":
    asyncio.run(main()) 