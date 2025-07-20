#!/usr/bin/env python3
"""
MetaExtract Simple File Processor

Command-line tool to process content files using JSON schemas with the existing MetaExtract system.

Usage:
    python simple_file_processor.py <content_file> <schema_file> [output_file]

Examples:
    python simple_file_processor.py resume.txt resume_schema.json output.json
    python simple_file_processor.py document.md schema.json
"""

import asyncio
import json
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Import the existing MetaExtract system
from metaextract.simplified_extractor import SimplifiedMetaExtract


def load_content_file(file_path: str) -> str:
    """Load content from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✅ Loaded content file: {file_path} ({len(content)} characters)")
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


def save_output_file(data: Dict[str, Any], file_path: str) -> None:
    """Save extracted data to a JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Output saved to: {file_path}")
    except Exception as e:
        print(f"❌ Error saving output file: {e}")
        sys.exit(1)


def print_summary(result, processing_time: float) -> None:
    """Print a summary of the extraction results"""
    print("\n" + "="*60)
    print("📊 EXTRACTION SUMMARY")
    print("="*60)
    
    print(f"🎯 Success: {result.success}")
    print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
    print(f"📈 Confidence Score: {result.confidence_score:.2f}")
    print(f"🧠 Strategy Used: {result.strategy_used}")
    print(f"🔢 Token Usage: {result.token_usage}")
    print(f"📦 Chunks Processed: {result.chunks_processed}")
    
    if result.validation_errors:
        print(f"⚠️  Validation Errors: {len(result.validation_errors)}")
        for error in result.validation_errors[:3]:  # Show first 3 errors
            print(f"   - {error}")
    
    if result.low_confidence_fields:
        print(f"⚠️  Low Confidence Fields: {len(result.low_confidence_fields)}")
        for field in result.low_confidence_fields[:3]:  # Show first 3 fields
            print(f"   - {field}")
    
    # Show a preview of extracted data
    if result.extracted_data:
        print(f"\n📄 Extracted Data Preview:")
        preview = json.dumps(result.extracted_data, indent=2)
        if len(preview) > 500:
            preview = preview[:500] + "...\n}"
        print(preview)


def check_prerequisites() -> bool:
    """Check if all prerequisites are met"""
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found.")
        print("   Please set OPENAI_API_KEY in your .env file or environment variables.")
        print("   Get your API key from: https://platform.openai.com/api-keys")
        return False
    
    # Check if MetaExtract system can be imported
    try:
        from metaextract.simplified_extractor import SimplifiedMetaExtract
        return True
    except ImportError as e:
        print(f"❌ MetaExtract system not available: {e}")
        print("   Please ensure the metaextract module is available.")
        return False


async def process_file_with_metaextract(content: str, schema: Dict[str, Any], strategy: str = "auto") -> Any:
    """Process content using the existing MetaExtract system"""
    print(f"\n🤖 Initializing MetaExtract System...")
    
    # Create the extractor
    extractor = SimplifiedMetaExtract()
    print(f"   📋 Using strategy: {strategy}")
    
    print("\n🚀 Starting Data Extraction...")
    start_time = time.time()
    
    # Perform extraction
    result = await extractor.extract(
        input_text=content,
        schema=schema,
        strategy=strategy
    )
    
    processing_time = time.time() - start_time
    
    return result, processing_time


def create_sample_files():
    """Create sample content and schema files for testing"""
    
    # Sample content file
    sample_content = """
John Smith
Senior Software Engineer

Contact Information:
Email: john.smith@example.com
Phone: (555) 123-4567
Location: San Francisco, CA

Professional Summary:
Experienced software engineer with 8+ years in full-stack development.
Led multiple successful projects and teams.

Work Experience:
- Senior Software Engineer at TechCorp (2020-Present)
  • Led development of microservices architecture
  • Managed team of 5 developers
  • Technologies: Python, React, AWS

- Software Engineer at StartupXYZ (2018-2020)
  • Built real-time analytics dashboard
  • Improved performance by 40%
  • Technologies: Node.js, MongoDB

Education:
Bachelor of Science in Computer Science
University of California, Berkeley (2012-2016)

Skills:
Programming: Python, JavaScript, TypeScript
Frameworks: React, Node.js, Django
Databases: PostgreSQL, MongoDB
Cloud: AWS, Azure
"""
    
    # Sample schema file
    sample_schema = {
        "type": "object",
        "properties": {
            "personal_info": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                    "location": {"type": "string"}
                },
                "required": ["name", "email"]
            },
            "summary": {"type": "string"},
            "work_experience": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "company": {"type": "string"},
                        "duration": {"type": "string"},
                        "responsibilities": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "technologies": {
                            "type": "array", 
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["title", "company"]
                }
            },
            "education": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "degree": {"type": "string"},
                        "institution": {"type": "string"},
                        "duration": {"type": "string"}
                    },
                    "required": ["degree", "institution"]
                }
            },
            "skills": {
                "type": "object",
                "properties": {
                    "programming": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "frameworks": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "databases": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "cloud": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        },
        "required": ["personal_info", "work_experience", "skills"]
    }
    
    # Write sample files
    with open("sample_resume.txt", "w") as f:
        f.write(sample_content)
    
    with open("sample_resume_schema.json", "w") as f:
        json.dump(sample_schema, f, indent=2)
    
    print("📄 Created sample files:")
    print("   - sample_resume.txt (content file)")
    print("   - sample_resume_schema.json (schema file)")
    print("\nTry: python simple_file_processor.py sample_resume.txt sample_resume_schema.json")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Process content files using JSON schemas with MetaExtract",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_file_processor.py resume.txt schema.json
  python simple_file_processor.py document.md schema.json output.json
  python simple_file_processor.py --create-samples
  python simple_file_processor.py --strategy hierarchical content.txt schema.json
        """
    )
    
    parser.add_argument("content_file", nargs="?", help="Path to the content file to process")
    parser.add_argument("schema_file", nargs="?", help="Path to the JSON schema file")
    parser.add_argument("output_file", nargs="?", help="Path to save the output JSON (optional)")
    parser.add_argument("--strategy", choices=["simple", "chunked", "hierarchical", "auto"], 
                       default="simple", help="Extraction strategy to use")
    parser.add_argument("--create-samples", action="store_true", help="Create sample content and schema files")
    
    args = parser.parse_args()
    
    # Print banner
    print("""
╔══════════════════════════════════════════════════════════════╗
║              MetaExtract Simple File Processor              ║
║                                                              ║
║    Transform any content into structured JSON using AI      ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create sample files if requested
    if args.create_samples:
        create_sample_files()
        return
    
    # Check if files are provided
    if not args.content_file or not args.schema_file:
        print("❌ Please provide both content file and schema file")
        print("   Usage: python simple_file_processor.py <content_file> <schema_file> [output_file]")
        print("   Or: python simple_file_processor.py --create-samples (to create sample files)")
        sys.exit(1)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Load files
    print("📁 Loading files...")
    content = load_content_file(args.content_file)
    schema = load_schema_file(args.schema_file)
    
    # Process with MetaExtract
    try:
        result, processing_time = await process_file_with_metaextract(
            content, 
            schema, 
            args.strategy
        )
        
        if result.success:
            # Determine output file path
            if args.output_file:
                output_path = args.output_file
            else:
                # Auto-generate output filename
                content_name = Path(args.content_file).stem
                output_path = f"{content_name}_extracted.json"
            
            # Save results
            save_output_file(result.extracted_data, output_path)
            
            # Print summary
            print_summary(result, processing_time)
            
            print(f"\n🎉 Processing completed successfully!")
            print(f"📄 Results saved to: {output_path}")
            
        else:
            print(f"\n❌ Extraction failed: {result.error_message}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 