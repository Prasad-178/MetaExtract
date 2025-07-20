"""
Test script for Schema Complexity Analyzer

Tests the analyzer with the provided schemas from Metaforms assignment.
"""

import json
import sys
sys.path.append('.')

from metaextract.core.schema_analyzer import SchemaComplexityAnalyzer

def load_schema(filename):
    """Load a JSON schema file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def test_schemas():
    """Test the analyzer with all provided schemas"""
    analyzer = SchemaComplexityAnalyzer()
    
    schemas = [
        ("Resume Schema", "testcases/convert your resume to this schema.json"),
        ("GitHub Actions Schema", "testcases/github_actions_schema.json"), 
        ("Paper Citations Schema", "testcases/paper citations_schema.json")
    ]
    
    print("MetaExtract Schema Complexity Analysis")
    print("=" * 50)
    
    for name, filename in schemas:
        try:
            print(f"\n📋 Analyzing: {name}")
            print("-" * 30)
            
            schema = load_schema(filename)
            metrics = analyzer.analyze_schema(schema)
            
            print(analyzer.get_complexity_summary(metrics))
            print()
            
        except Exception as e:
            print(f"❌ Error analyzing {name}: {e}")

if __name__ == "__main__":
    test_schemas() 