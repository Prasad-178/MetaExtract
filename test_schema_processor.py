"""
Test script for Hierarchical Schema Processor

Demonstrates how complex schemas are broken down into manageable chunks.
"""

import json
import sys
sys.path.append('.')

from metaextract.core.schema_processor import HierarchicalSchemaProcessor

def load_schema(filename):
    """Load a JSON schema file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def test_schema_processing():
    """Test hierarchical processing for all provided schemas"""
    processor = HierarchicalSchemaProcessor(max_chunk_complexity=15.0)
    
    schemas = [
        ("Resume Schema", "testcases/convert your resume to this schema.json"),
        ("GitHub Actions Schema", "testcases/github_actions_schema.json"), 
        ("Paper Citations Schema", "testcases/paper citations_schema.json")
    ]
    
    print("MetaExtract: Hierarchical Schema Processing Test")
    print("=" * 60)
    
    for name, filename in schemas:
        try:
            print(f"\n🔧 Processing: {name}")
            print("-" * 40)
            
            # Load and process schema
            schema = load_schema(filename)
            plan = processor.process_schema(schema)
            
            # Display results
            print(f"📊 Processing Results:")
            print(f"   Total Chunks: {len(plan.chunks)}")
            print(f"   Estimated Complexity: {plan.estimated_total_complexity:.2f}")
            print(f"   Requires Merging: {plan.requires_merging}")
            print(f"   Parallel Groups: {len(plan.parallel_groups)}")
            
            # Show processing order
            print(f"\n🔄 Processing Order:")
            for i, chunk_id in enumerate(plan.processing_order, 1):
                chunk = plan.chunks[chunk_id]
                print(f"   {i}. {chunk_id} ({chunk.chunk_type.value}, complexity: {chunk.estimated_complexity:.2f})")
            
            # Show parallel groups
            print(f"\n⚡ Parallel Processing Groups:")
            for i, group in enumerate(plan.parallel_groups, 1):
                if len(group) > 1:
                    group_complexity = sum(plan.chunks[chunk_id].estimated_complexity for chunk_id in group)
                    print(f"   Group {i}: {', '.join(group)} (total complexity: {group_complexity:.2f})")
                else:
                    chunk = plan.chunks[group[0]]
                    print(f"   Group {i}: {group[0]} (complexity: {chunk.estimated_complexity:.2f})")
            
            # Show detailed chunk analysis
            print(f"\n📋 Detailed Chunk Analysis:")
            for chunk_id, chunk in plan.chunks.items():
                print(f"   🔹 {chunk_id}:")
                print(f"      Type: {chunk.chunk_type.value}")
                print(f"      Complexity: {chunk.estimated_complexity:.2f}")
                print(f"      Original Path: {chunk.original_path}")
                print(f"      Priority: {chunk.priority}")
                
                if chunk.dependencies:
                    dep_names = [dep.target_chunk_id for dep in chunk.dependencies]
                    print(f"      Dependencies: {', '.join(dep_names)}")
                
                if chunk.parent_chunk_id:
                    print(f"      Parent: {chunk.parent_chunk_id}")
                
                if chunk.child_chunk_ids:
                    print(f"      Children: {', '.join(chunk.child_chunk_ids)}")
                
                # Show sample of schema content
                content_str = json.dumps(chunk.schema_content, indent=2)[:200]
                print(f"      Sample Content: {content_str}{'...' if len(content_str) >= 200 else ''}")
                print()
            
            print(f"✅ Schema processing completed successfully!")
            
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
            import traceback
            traceback.print_exc()

def test_chunking_strategies():
    """Test different chunking strategies"""
    print("\n" + "=" * 60)
    print("Chunking Strategy Comparison")
    print("=" * 60)
    
    # Load a complex schema
    schema = load_schema("testcases/paper citations_schema.json")
    
    strategies = [
        ("Conservative (max complexity: 5)", 5.0),
        ("Balanced (max complexity: 15)", 15.0),
        ("Aggressive (max complexity: 30)", 30.0)
    ]
    
    for strategy_name, max_complexity in strategies:
        print(f"\n🎯 Strategy: {strategy_name}")
        print("-" * 30)
        
        processor = HierarchicalSchemaProcessor(max_chunk_complexity=max_complexity)
        plan = processor.process_schema(schema)
        
        print(f"   Chunks Created: {len(plan.chunks)}")
        print(f"   Total Complexity: {plan.estimated_total_complexity:.2f}")
        print(f"   Parallel Groups: {len(plan.parallel_groups)}")
        print(f"   Max Parallel: {max(len(group) for group in plan.parallel_groups)}")
        
        # Calculate efficiency metrics
        avg_complexity_per_chunk = plan.estimated_total_complexity / len(plan.chunks)
        parallelization_ratio = sum(len(group) for group in plan.parallel_groups) / len(plan.chunks)
        
        print(f"   Avg Complexity/Chunk: {avg_complexity_per_chunk:.2f}")
        print(f"   Parallelization Ratio: {parallelization_ratio:.2f}")

def test_dependency_tracking():
    """Test dependency tracking with a custom schema"""
    print("\n" + "=" * 60)
    print("Dependency Tracking Test")
    print("=" * 60)
    
    # Create a schema with complex dependencies
    test_schema = {
        "type": "object",
        "definitions": {
            "person": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "address": {"$ref": "#/definitions/address"}
                }
            },
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "country": {"type": "string"}
                }
            }
        },
        "properties": {
            "employees": {
                "type": "array",
                "items": {"$ref": "#/definitions/person"}
            },
            "company": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "headquarters": {"$ref": "#/definitions/address"},
                    "founded": {"type": "integer"}
                }
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "version": {"type": "string"},
                    "last_updated": {"type": "string"}
                }
            }
        }
    }
    
    processor = HierarchicalSchemaProcessor(max_chunk_complexity=8.0)
    plan = processor.process_schema(test_schema)
    
    print(f"📊 Dependency Analysis Results:")
    print(f"   Total Chunks: {len(plan.chunks)}")
    print(f"   Processing Order: {' → '.join(plan.processing_order)}")
    
    print(f"\n🔗 Dependency Details:")
    for chunk_id, chunk in plan.chunks.items():
        if chunk.dependencies:
            print(f"   {chunk_id} depends on:")
            for dep in chunk.dependencies:
                print(f"      - {dep.target_chunk_id} ({dep.dependency_type})")
        else:
            print(f"   {chunk_id} has no dependencies")

if __name__ == "__main__":
    test_schema_processing()
    test_chunking_strategies()
    test_dependency_tracking() 