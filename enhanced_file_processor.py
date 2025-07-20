#!/usr/bin/env python3
"""
MetaExtract Enhanced File Processor with Comprehensive Evaluation

Command-line tool with advanced metrics, low confidence flagging, and large document support.

Usage:
    python enhanced_file_processor.py <content_file> <schema_file> [output_file]

Features:
- Comprehensive evaluation metrics
- Low confidence field flagging  
- Large document support (50 pages to 10MB)
- Schema complexity-based strategy selection
- Detailed performance analysis
"""

import asyncio
import json
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv
import time
import psutil
import logging
from dataclasses import dataclass, asdict

# Load environment variables
load_dotenv()

# Import the existing MetaExtract system
from metaextract.simplified_extractor import SimplifiedMetaExtract, SchemaComplexity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics"""
    # Performance Metrics
    total_processing_time: float
    llm_call_time: float
    schema_analysis_time: float
    preprocessing_time: float
    postprocessing_time: float
    
    # LLM Usage Metrics
    total_llm_calls: int
    total_tokens_used: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    
    # Document Metrics
    input_document_size_chars: int
    input_document_size_mb: float
    chunks_processed: int
    strategy_used: str
    
    # Quality Metrics
    confidence_score: float
    validation_errors: List[str]
    low_confidence_fields: List[str]
    completeness_score: float
    
    # Schema Complexity Metrics
    schema_complexity_score: float
    schema_nesting_depth: int
    schema_total_properties: int
    schema_objects_count: int
    
    # System Resource Metrics
    peak_memory_usage_mb: float
    cpu_usage_percent: float


class EnhancedMetaExtract:
    """Enhanced MetaExtract with comprehensive evaluation capabilities"""
    
    def __init__(self):
        self.base_extractor = SimplifiedMetaExtract()
        self.evaluation_metrics = None
        self.process = psutil.Process()
        
        # Token pricing (approximate for GPT-4)
        self.input_token_cost = 0.03 / 1000  # $0.03 per 1K input tokens
        self.output_token_cost = 0.06 / 1000  # $0.06 per 1K output tokens
    
    async def extract_with_evaluation(self, 
                                    content: str, 
                                    schema: Dict[str, Any],
                                    strategy: str = "simple") -> Tuple[Any, EvaluationMetrics]:
        """Extract data with comprehensive evaluation metrics"""
        
        # Start timing and monitoring
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = self.process.cpu_percent()
        
        logger.info(f"Starting enhanced extraction with strategy: {strategy}")
        
        # Phase 1: Schema Analysis
        schema_start = time.time()
        complexity = self.base_extractor._analyze_schema_complexity(schema)
        recommended_strategy = self._analyze_and_recommend_strategy(complexity, len(content))
        schema_analysis_time = time.time() - schema_start
        
        logger.info(f"Schema complexity score: {complexity.complexity_score:.2f}")
        logger.info(f"Recommended strategy: {recommended_strategy}")
        
        # Use recommended strategy if 'auto' is specified
        if strategy == "auto":
            strategy = recommended_strategy
        
        # Phase 2: Preprocessing
        preprocess_start = time.time()
        document_size_chars = len(content)
        document_size_mb = len(content.encode('utf-8')) / 1024 / 1024
        preprocessed_content = self._preprocess_large_document(content, strategy)
        preprocessing_time = time.time() - preprocess_start
        
        # Phase 3: Main Extraction
        llm_start = time.time()
        result = await self.base_extractor.extract(
            input_text=preprocessed_content,
            schema=schema,
            strategy=strategy
        )
        llm_call_time = time.time() - llm_start
        
        # Phase 4: Post-processing and Quality Analysis
        postprocess_start = time.time()
        quality_metrics = self._analyze_quality(result, content, schema)
        low_confidence_fields = self._flag_low_confidence_fields(result, threshold=0.6)
        postprocessing_time = time.time() - postprocess_start
        
        # Calculate final metrics
        total_time = time.time() - start_time
        peak_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        cpu_usage = self.process.cpu_percent()
        
        # Estimate costs
        estimated_cost = self._calculate_estimated_cost(result.token_usage)
        
        # Create comprehensive metrics
        metrics = EvaluationMetrics(
            # Performance
            total_processing_time=total_time,
            llm_call_time=llm_call_time,
            schema_analysis_time=schema_analysis_time,
            preprocessing_time=preprocessing_time,
            postprocessing_time=postprocessing_time,
            
            # LLM Usage
            total_llm_calls=1,  # For simple strategy, more for chunked/hierarchical
            total_tokens_used=result.token_usage,
            input_tokens=int(result.token_usage * 0.7),  # Estimate
            output_tokens=int(result.token_usage * 0.3),  # Estimate
            estimated_cost_usd=estimated_cost,
            
            # Document
            input_document_size_chars=document_size_chars,
            input_document_size_mb=document_size_mb,
            chunks_processed=result.chunks_processed,
            strategy_used=result.strategy_used,
            
            # Quality
            confidence_score=result.confidence_score,
            validation_errors=result.validation_errors,
            low_confidence_fields=low_confidence_fields,
            completeness_score=quality_metrics['completeness'],
            
            # Schema Complexity
            schema_complexity_score=complexity.complexity_score,
            schema_nesting_depth=complexity.nesting_depth,
            schema_total_properties=complexity.total_properties,
            schema_objects_count=complexity.total_objects,
            
            # System Resources
            peak_memory_usage_mb=peak_memory - start_memory,
            cpu_usage_percent=cpu_usage
        )
        
        self.evaluation_metrics = metrics
        return result, metrics
    
    def _analyze_and_recommend_strategy(self, complexity: SchemaComplexity, content_length: int) -> str:
        """Analyze schema complexity and document size to recommend optimal strategy"""
        
        # Large document thresholds
        large_doc_threshold = 100_000  # 100KB characters (~50 pages)
        very_large_doc_threshold = 1_000_000  # 1MB characters
        
        # Schema complexity thresholds
        simple_complexity_threshold = 20
        complex_complexity_threshold = 50
        
        logger.info(f"Document size: {content_length:,} chars")
        logger.info(f"Schema complexity: {complexity.complexity_score:.2f}")
        
        # Decision logic based on both factors
        if content_length > very_large_doc_threshold and complexity.complexity_score > complex_complexity_threshold:
            return "hierarchical"
        elif content_length > large_doc_threshold or complexity.complexity_score > complex_complexity_threshold:
            return "chunked"
        else:
            return "simple"
    
    def _preprocess_large_document(self, content: str, strategy: str) -> str:
        """Preprocess large documents for optimal extraction"""
        
        content_size = len(content.encode('utf-8'))
        max_size_mb = 10
        
        if content_size > max_size_mb * 1024 * 1024:
            logger.warning(f"Document size ({content_size/1024/1024:.2f} MB) exceeds recommended {max_size_mb} MB")
            
            # For very large documents, take a strategic sample
            if strategy == "simple":
                # Take first 50% and last 10% for simple strategy
                mid_point = len(content) // 2
                end_sample = int(len(content) * 0.1)
                content = content[:mid_point] + "\n...[content truncated]...\n" + content[-end_sample:]
                logger.info("Large document: Using strategic sampling for simple strategy")
        
        return content
    
    def _analyze_quality(self, result, original_content: str, schema: Dict[str, Any]) -> Dict[str, float]:
        """Analyze extraction quality"""
        
        if not result.success or not result.extracted_data:
            return {'completeness': 0.0, 'accuracy': 0.0}
        
        # Calculate completeness based on required fields
        required_fields = schema.get('required', [])
        if required_fields:
            present_required = sum(1 for field in required_fields 
                                 if field in result.extracted_data and result.extracted_data[field])
            completeness = present_required / len(required_fields)
        else:
            completeness = 1.0
        
        # Simple accuracy check - see if extracted values appear in source
        accuracy = self._calculate_accuracy(result.extracted_data, original_content)
        
        return {
            'completeness': completeness,
            'accuracy': accuracy
        }
    
    def _calculate_accuracy(self, extracted_data: Dict[str, Any], original_content: str) -> float:
        """Calculate accuracy by checking if extracted values appear in source"""
        
        content_lower = original_content.lower()
        total_values = 0
        accurate_values = 0
        
        def check_values(obj):
            nonlocal total_values, accurate_values
            
            if isinstance(obj, dict):
                for value in obj.values():
                    check_values(value)
            elif isinstance(obj, list):
                for item in obj:
                    check_values(item)
            elif isinstance(obj, str) and obj.strip():
                total_values += 1
                if obj.lower().strip() in content_lower:
                    accurate_values += 1
        
        check_values(extracted_data)
        return accurate_values / total_values if total_values > 0 else 1.0
    
    def _flag_low_confidence_fields(self, result, threshold: float = 0.6) -> List[str]:
        """Flag fields with low confidence for human review"""
        
        low_confidence = []
        
        # Add validation errors as low confidence
        if result.validation_errors:
            low_confidence.extend([f"validation_error_{i}" for i in range(len(result.validation_errors))])
        
        # Add already flagged low confidence fields
        if result.low_confidence_fields:
            low_confidence.extend(result.low_confidence_fields)
        
        # Add overall low confidence if score is below threshold
        if result.confidence_score < threshold:
            low_confidence.append("overall_confidence_low")
        
        return list(set(low_confidence))
    
    def _calculate_estimated_cost(self, total_tokens: int) -> float:
        """Calculate estimated API cost"""
        input_tokens = int(total_tokens * 0.7)
        output_tokens = int(total_tokens * 0.3)
        
        cost = (input_tokens * self.input_token_cost + 
                output_tokens * self.output_token_cost)
        return round(cost, 4)


def load_content_file(file_path: str) -> str:
    """Load content from a file with large file support"""
    try:
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / 1024 / 1024
        
        print(f"✅ Loading content file: {file_path} ({file_size_mb:.2f} MB)")
        
        if file_size_mb > 50:
            print(f"⚠️  Large file detected ({file_size_mb:.2f} MB). Processing may take longer.")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"✅ Loaded {len(content):,} characters")
        return content
    except FileNotFoundError:
        print(f"❌ Content file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading content file: {e}")
        sys.exit(1)


def load_schema_file(file_path: str) -> Dict[str, Any]:
    """Load JSON schema from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        print(f"✅ Loaded schema file: {file_path}")
        return schema
    except FileNotFoundError:
        print(f"❌ Schema file not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in schema file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading schema file: {e}")
        sys.exit(1)


def save_output_files(data: Dict[str, Any], metrics: EvaluationMetrics, output_path: str) -> None:
    """Save extracted data and evaluation metrics"""
    
    # Save main extracted data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Extracted data saved to: {output_path}")
    
    # Save evaluation metrics
    metrics_path = output_path.replace('.json', '_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)
    print(f"✅ Evaluation metrics saved to: {metrics_path}")
    
    # Save low confidence fields report if any
    if metrics.low_confidence_fields:
        review_path = output_path.replace('.json', '_review.json')
        review_data = {
            "low_confidence_fields": metrics.low_confidence_fields,
            "validation_errors": metrics.validation_errors,
            "overall_confidence": metrics.confidence_score,
            "recommendation": "Human review recommended for flagged fields" if metrics.low_confidence_fields else "No issues detected"
        }
        with open(review_path, 'w', encoding='utf-8') as f:
            json.dump(review_data, f, indent=2, ensure_ascii=False)
        print(f"⚠️  Human review report saved to: {review_path}")


def print_comprehensive_summary(result, metrics: EvaluationMetrics) -> None:
    """Print comprehensive evaluation summary"""
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE EVALUATION SUMMARY")
    print("="*80)
    
    # Performance Metrics
    print(f"\n🚀 PERFORMANCE METRICS:")
    print(f"   Total Processing Time: {metrics.total_processing_time:.2f}s")
    print(f"   LLM Call Time: {metrics.llm_call_time:.2f}s ({metrics.llm_call_time/metrics.total_processing_time*100:.1f}%)")
    print(f"   Schema Analysis Time: {metrics.schema_analysis_time:.2f}s")
    print(f"   Preprocessing Time: {metrics.preprocessing_time:.2f}s")
    print(f"   Postprocessing Time: {metrics.postprocessing_time:.2f}s")
    
    # LLM Usage
    print(f"\n🤖 LLM USAGE METRICS:")
    print(f"   Total LLM Calls: {metrics.total_llm_calls}")
    print(f"   Total Tokens Used: {metrics.total_tokens_used:,}")
    print(f"   Estimated Cost: ${metrics.estimated_cost_usd:.4f}")
    print(f"   Strategy Used: {metrics.strategy_used}")
    
    # Document Analysis
    print(f"\n📄 DOCUMENT ANALYSIS:")
    print(f"   Document Size: {metrics.input_document_size_chars:,} chars ({metrics.input_document_size_mb:.2f} MB)")
    print(f"   Chunks Processed: {metrics.chunks_processed}")
    
    # Quality Metrics
    print(f"\n📈 QUALITY METRICS:")
    print(f"   Overall Confidence: {metrics.confidence_score:.2f}")
    print(f"   Completeness Score: {metrics.completeness_score:.2f}")
    print(f"   Validation Errors: {len(metrics.validation_errors)}")
    
    # Schema Complexity
    print(f"\n🔧 SCHEMA COMPLEXITY:")
    print(f"   Complexity Score: {metrics.schema_complexity_score:.2f}")
    print(f"   Nesting Depth: {metrics.schema_nesting_depth}")
    print(f"   Total Properties: {metrics.schema_total_properties}")
    print(f"   Object Count: {metrics.schema_objects_count}")
    
    # System Resources
    print(f"\n💻 SYSTEM RESOURCES:")
    print(f"   Peak Memory Usage: {metrics.peak_memory_usage_mb:.2f} MB")
    print(f"   CPU Usage: {metrics.cpu_usage_percent:.1f}%")
    
    # Low Confidence Fields Warning
    if metrics.low_confidence_fields:
        print(f"\n⚠️  LOW CONFIDENCE FIELDS ({len(metrics.low_confidence_fields)}):")
        for field in metrics.low_confidence_fields[:5]:  # Show first 5
            print(f"   - {field}")
        if len(metrics.low_confidence_fields) > 5:
            print(f"   ... and {len(metrics.low_confidence_fields) - 5} more")
        print(f"   🔍 Human review recommended!")
    
    # Performance Assessment
    print(f"\n🎯 PERFORMANCE ASSESSMENT:")
    efficiency_score = min(100, max(0, 100 - (metrics.total_processing_time * 10)))
    cost_efficiency = "Excellent" if metrics.estimated_cost_usd < 0.10 else "Good" if metrics.estimated_cost_usd < 0.50 else "High"
    quality_rating = "Excellent" if metrics.confidence_score > 0.85 else "Good" if metrics.confidence_score > 0.70 else "Needs Review"
    
    print(f"   Efficiency Score: {efficiency_score:.0f}/100")
    print(f"   Cost Efficiency: {cost_efficiency}")
    print(f"   Quality Rating: {quality_rating}")


async def main():
    """Main function with enhanced capabilities"""
    parser = argparse.ArgumentParser(
        description="Enhanced MetaExtract with comprehensive evaluation metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features:
  - Comprehensive evaluation metrics
  - Low confidence field flagging
  - Large document support (50 pages to 10MB)
  - Schema complexity-based strategy selection
  - Detailed performance analysis

Examples:
  python enhanced_file_processor.py document.txt schema.json
  python enhanced_file_processor.py --strategy auto large_doc.txt complex_schema.json
  python enhanced_file_processor.py --save-metrics report.txt schema.json output.json
        """
    )
    
    parser.add_argument("content_file", help="Path to the content file to process")
    parser.add_argument("schema_file", help="Path to the JSON schema file")
    parser.add_argument("output_file", nargs="?", help="Path to save the output JSON (optional)")
    parser.add_argument("--strategy", choices=["simple", "chunked", "hierarchical", "auto"], 
                       default="auto", help="Extraction strategy (auto = intelligent selection)")
    parser.add_argument("--save-metrics", action="store_true", help="Save detailed evaluation metrics")
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold for flagging fields")
    
    args = parser.parse_args()
    
    # Print banner
    print("""
╔══════════════════════════════════════════════════════════════╗
║          MetaExtract Enhanced File Processor v2.0           ║
║                                                              ║
║    Advanced AI extraction with comprehensive evaluation     ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check prerequisites
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Load files
    print("📁 Loading files...")
    content = load_content_file(args.content_file)
    schema = load_schema_file(args.schema_file)
    
    # Process with enhanced evaluation
    try:
        extractor = EnhancedMetaExtract()
        result, metrics = await extractor.extract_with_evaluation(
            content, schema, args.strategy
        )
        
        if result.success:
            # Determine output file path
            if args.output_file:
                output_path = args.output_file
            else:
                content_name = Path(args.content_file).stem
                output_path = f"{content_name}_extracted.json"
            
            # Save results and metrics
            save_output_files(result.extracted_data, metrics, output_path)
            
            # Print comprehensive summary
            print_comprehensive_summary(result, metrics)
            
            print(f"\n🎉 Processing completed successfully!")
            
            if metrics.low_confidence_fields:
                print(f"⚠️  {len(metrics.low_confidence_fields)} fields flagged for human review")
            
        else:
            print(f"\n❌ Extraction failed: {result.error_message}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        logger.exception("Extraction failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 