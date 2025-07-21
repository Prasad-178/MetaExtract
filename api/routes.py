"""
API routes for MetaExtract.
"""
import os
import json
import pandas as pd
from io import BytesIO
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from .models import ExtractionRequest, ExtractionResult, HealthResponse
from .config import settings
from metaextract.simplified_extractor import SimplifiedMetaExtract

# Import agentic components
from agentic import AgenticMetaExtract, AgenticExtractionRequest, AgenticExtractionResult

# PDF processing
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

router = APIRouter()

# Initialize extractor with lazy loading for Railway compatibility
def get_extractor():
    """Get or create extractor instance with proper error handling"""
    try:
        # Check for OpenAI API key in multiple possible env var names
        api_key = (os.getenv("OPENAI_API_KEY") or 
                  os.getenv("OPENAI_API_KEY_SECRET"))
        
        if not api_key:
            print("No OpenAI API key found in environment variables")
            return None, False
            
        print(f"Found OpenAI API key: {api_key[:8]}...")  # Log first 8 chars for debugging
        extractor = SimplifiedMetaExtract(api_key)
        print("Successfully initialized MetaExtract")
        return extractor, True
        
    except ValueError as e:
        print(f"ValueError initializing extractor: {e}")
        return None, False
    except Exception as e:
        print(f"Unexpected error initializing extractor: {e}")
        import traceback
        traceback.print_exc()
        return None, False

# Initialize on startup but don't fail if not available
extractor, has_openai_key = get_extractor()

# Initialize agentic extractor
def get_agentic_extractor():
    """Get or create agentic extractor instance"""
    try:
        api_key = (os.getenv("OPENAI_API_KEY") or 
                  os.getenv("OPENAI_API_KEY_SECRET"))
        
        if not api_key:
            return None, False
            
        agentic_extractor = AgenticMetaExtract(api_key)
        return agentic_extractor, True
        
    except Exception as e:
        print(f"Error initializing agentic extractor: {e}")
        return None, False

agentic_extractor, has_agentic_key = get_agentic_extractor()


def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract text from various file formats"""
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext == '.pdf':
        return extract_text_from_pdf(file_content)
    elif file_ext == '.csv':
        return extract_text_from_csv(file_content)
    else:
        # Try to decode as text
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return file_content.decode('latin-1')
            except:
                raise HTTPException(status_code=400, detail=f"Could not read file as text: {filename}")


def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    if not PDF_AVAILABLE:
        raise HTTPException(status_code=400, detail="PDF processing not available. Install PyPDF2 and pdfplumber.")
    
    try:
        # Try pdfplumber first (better for complex layouts)
        with pdfplumber.open(BytesIO(file_content)) as pdf:
            text_parts = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            if text_parts:
                return '\n\n'.join(text_parts)
    except Exception as e:
        print(f"pdfplumber failed: {e}, trying PyPDF2...")
    
    try:
        # Fallback to PyPDF2
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        text_parts = []
        
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        if text_parts:
            return '\n\n'.join(text_parts)
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not extract text from PDF: {str(e)}")
    
    raise HTTPException(status_code=400, detail="PDF appears to be empty or text could not be extracted")


def extract_text_from_csv(file_content: bytes) -> str:
    """Extract text from CSV file"""
    try:
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(BytesIO(file_content), encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise HTTPException(status_code=400, detail="Could not decode CSV file")
        
        # Convert DataFrame to structured text
        text_parts = []
        text_parts.append(f"CSV file with {len(df)} rows and {len(df.columns)} columns")
        text_parts.append(f"Columns: {', '.join(df.columns)}")
        text_parts.append("\nData sample:")
        
        # Add sample rows
        sample_size = min(10, len(df))
        for i, row in df.head(sample_size).iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
            text_parts.append(f"Row {i+1}: {row_text}")
        
        if len(df) > sample_size:
            text_parts.append(f"... and {len(df) - sample_size} more rows")
        
        return '\n'.join(text_parts)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process CSV file: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running."""
    # Re-check OpenAI key availability dynamically
    current_extractor, current_has_key = get_extractor()
    status = "healthy" if current_has_key else "no_openai_key"
    
    return HealthResponse(
        status=status,
        version=settings.api_version
    )


@router.post("/extract", response_model=ExtractionResult)
async def extract_from_text(request: ExtractionRequest):
    """Extract structured data from text."""
    current_extractor, current_has_key = get_extractor()
    if not current_has_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not configured")
    
    try:
        result = await current_extractor.extract(
            input_text=request.text,
            schema=request.json_schema,
            strategy=None if request.strategy == "auto" else request.strategy
        )
        
        # Always return data, even if confidence is low - just flag it
        if not result.success:
            raise HTTPException(
                status_code=400, 
                detail=f"Extraction failed: {result.error_message or 'Unknown error'}"
            )
        
        return ExtractionResult(
            data=result.extracted_data or {},
            confidence=result.confidence_score,
            field_confidences=result.field_confidences,
            low_confidence_fields=result.low_confidence_fields,
            processing_time=result.processing_time,
            strategy_used=result.strategy_used
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract/file")
async def extract_from_file(
    file: UploadFile = File(...),
    schema: str = Form(...)
):
    """Extract structured data from uploaded file."""
    current_extractor, current_has_key = get_extractor()
    if not current_has_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not configured")
    
    # Read file content and extract text based on file type
    try:
        content = await file.read()
        text = extract_text_from_file(content, file.filename or "unknown")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process file: {str(e)}")
    
    # Parse schema
    try:
        schema_dict = json.loads(schema)
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON schema")
    
    # Extract data
    try:
        result = await current_extractor.extract(
            input_text=text,
            schema=schema_dict,
            strategy=None
        )
        
        # Always return data, even if confidence is low - just flag it
        if not result.success:
            raise HTTPException(
                status_code=400, 
                detail=f"Extraction failed: {result.error_message or 'Unknown error'}"
            )
        
        return {
            "data": result.extracted_data or {},
            "confidence": result.confidence_score,
            "field_confidences": result.field_confidences,
            "low_confidence_fields": result.low_confidence_fields,
            "processing_time": result.processing_time,
            "strategy_used": result.strategy_used,
            "filename": file.filename
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test")
async def test_extraction(
    content_file: UploadFile = File(...),
    schema_file: UploadFile = File(...)
):
    """Test endpoint: upload content file and schema file, get extracted JSON."""
    current_extractor, current_has_key = get_extractor()
    if not current_has_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not configured")
    
    try:
        # Read content file and extract text based on file type
        content = await content_file.read()
        text = extract_text_from_file(content, content_file.filename or "unknown")
        
        # Read schema file
        schema_content = await schema_file.read()
        schema = json.loads(schema_content.decode('utf-8'))
        
        # Extract data
        result = await current_extractor.extract(
            input_text=text,
            schema=schema,
            strategy=None
        )
        
        # Always return data, even if confidence is low - just flag it
        if not result.success:
            raise HTTPException(
                status_code=400, 
                detail=f"Extraction failed: {result.error_message or 'Unknown error'}"
            )
        
        return {
            "extracted_data": result.extracted_data or {},
            "confidence": result.confidence_score,
            "field_confidences": result.field_confidences,
            "low_confidence_fields": result.low_confidence_fields,
            "processing_time": result.processing_time,
            "strategy_used": result.strategy_used,
            "content_filename": content_file.filename,
            "schema_filename": schema_file.filename
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in schema file")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Could not read files as text")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.get("/strategies")
async def get_strategies():
    """Get available extraction strategies."""
    return {
        "strategies": ["simple", "chunked", "hierarchical", "auto"],
        "default": "auto",
        "description": {
            "simple": "Direct extraction for simple documents",
            "chunked": "Process large documents in chunks",
            "hierarchical": "Multi-level processing for complex schemas",
            "auto": "Automatically choose the best strategy"
        },
        "supported_file_types": {
            "text": [".txt", ".md", ".json"],
            "pdf": [".pdf"] if PDF_AVAILABLE else [],
            "data": [".csv"],
            "bibliography": [".bib"]
        }
    }


@router.post("/extract/agentic")
async def agentic_extract_from_text(request: AgenticExtractionRequest):
    """Extract structured data using agentic AI approach with multiple collaborative agents."""
    current_agentic_extractor, current_has_key = get_agentic_extractor()
    if not current_has_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not configured for agentic extraction")
    
    try:
        result = await current_agentic_extractor.extract(request)
        
        if not result.success:
            raise HTTPException(
                status_code=400, 
                detail="Agentic extraction failed"
            )
        
        return {
            "data": result.final_data or {},
            "confidence": result.overall_confidence,
            "strategy_used": result.strategy_used,
            "agents_used": result.agents_used,
            "processing_time": result.total_processing_time,
            "total_tokens_used": result.total_tokens_used,
            "consensus_fields": result.consensus_fields,
            "conflicting_fields": result.conflicting_fields,
            "validation_errors": result.validation_errors,
            "performance_metrics": result.performance_metrics,
            "agent_details": [
                {
                    "agent_id": agent.agent_id,
                    "agent_role": agent.agent_role.value,
                    "confidence": agent.confidence,
                    "reasoning": agent.reasoning,
                    "processing_time": agent.processing_time,
                    "tokens_used": agent.tokens_used
                }
                for agent in result.agent_results
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agentic extraction failed: {str(e)}")


@router.post("/extract/agentic/file")
async def agentic_extract_from_file(
    file: UploadFile = File(...),
    schema: str = Form(...),
    strategy: str = Form("adaptive"),
    enable_rag: bool = Form(False),
    max_agents: int = Form(5),
    confidence_threshold: float = Form(0.7),
    enable_cross_validation: bool = Form(True)
):
    """Extract structured data from uploaded file using agentic AI approach."""
    current_agentic_extractor, current_has_key = get_agentic_extractor()
    if not current_has_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not configured for agentic extraction")
    
    # Read file content and extract text
    try:
        content = await file.read()
        text = extract_text_from_file(content, file.filename or "unknown")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process file: {str(e)}")
    
    # Parse schema
    try:
        schema_dict = json.loads(schema)
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON schema")
    
    # Create agentic request
    agentic_request = AgenticExtractionRequest(
        text=text,
        json_schema=schema_dict,
        strategy=strategy,
        enable_rag=enable_rag,
        max_agents=max_agents,
        confidence_threshold=confidence_threshold,
        enable_cross_validation=enable_cross_validation
    )
    
    # Extract data
    try:
        result = await current_agentic_extractor.extract(agentic_request)
        
        if not result.success:
            raise HTTPException(
                status_code=400, 
                detail="Agentic extraction failed"
            )
        
        return {
            "data": result.final_data or {},
            "confidence": result.overall_confidence,
            "strategy_used": result.strategy_used,
            "agents_used": result.agents_used,
            "processing_time": result.total_processing_time,
            "total_tokens_used": result.total_tokens_used,
            "consensus_fields": result.consensus_fields,
            "conflicting_fields": result.conflicting_fields,
            "validation_errors": result.validation_errors,
            "performance_metrics": result.performance_metrics,
            "filename": file.filename,
            "agent_details": [
                {
                    "agent_id": agent.agent_id,
                    "agent_role": agent.agent_role.value,
                    "confidence": agent.confidence,
                    "reasoning": agent.reasoning,
                    "processing_time": agent.processing_time,
                    "tokens_used": agent.tokens_used
                }
                for agent in result.agent_results
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agentic extraction failed: {str(e)}")


@router.get("/agentic/strategies")
async def get_agentic_strategies():
    """Get available agentic extraction strategies and their descriptions."""
    return {
        "strategies": ["parallel", "hierarchical", "collaborative", "adaptive"],
        "default": "adaptive",
        "description": {
            "parallel": "Run multiple extraction agents in parallel for speed and consensus",
            "hierarchical": "Use supervisor agent to coordinate specialist agents for complex schemas",
            "collaborative": "Agents cross-validate and improve each other's work",
            "adaptive": "Automatically choose the best strategy based on document and schema complexity"
        },
        "features": {
            "agent_roles": [
                "schema_analyzer", "text_processor", "extraction_specialist", 
                "quality_assurance", "aggregator", "knowledge_retriever"
            ],
            "capabilities": [
                "Multi-agent collaboration", "Consensus building", "Cross-validation",
                "Schema complexity analysis", "Text chunking", "RAG integration",
                "Performance tracking", "Conflict resolution"
            ]
        },
        "advanced_options": {
            "enable_rag": "Enable Retrieval-Augmented Generation for domain knowledge",
            "max_agents": "Maximum number of agents to use (1-7)",
            "confidence_threshold": "Minimum confidence threshold for acceptance",
            "enable_cross_validation": "Enable agents to validate each other's work"
        }
    }


@router.get("/agentic/performance")
async def get_agentic_performance():
    """Get performance statistics for agentic extraction."""
    current_agentic_extractor, current_has_key = get_agentic_extractor()
    if not current_has_key:
        raise HTTPException(status_code=400, detail="Agentic extractor not available")
    
    try:
        stats = current_agentic_extractor.get_performance_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not retrieve performance stats: {str(e)}")


@router.post("/test/agentic")
async def test_agentic_extraction(
    content_file: UploadFile = File(...),
    schema_file: UploadFile = File(...)
):
    """Simplified test endpoint: upload content file and schema file, get agentic extracted JSON."""
    current_agentic_extractor, current_has_key = get_agentic_extractor()
    if not current_has_key:
        raise HTTPException(status_code=400, detail="Agentic extractor not available")
    
    try:
        # Read content file and extract text
        content = await content_file.read()
        text = extract_text_from_file(content, content_file.filename or "unknown")
        
        # Read schema file
        schema_content = await schema_file.read()
        schema = json.loads(schema_content.decode('utf-8'))
        
        # Create simplified agentic request
        agentic_request = AgenticExtractionRequest(
            text=text,
            json_schema=schema,
            strategy="simplified_fast",
            enable_rag=False,
            max_agents=3,
            confidence_threshold=0.7,
            enable_cross_validation=False
        )
        
        # Extract data
        result = await current_agentic_extractor.extract(agentic_request)
        
        if not result.success:
            raise HTTPException(
                status_code=400, 
                detail="Agentic extraction failed"
            )
        
        return {
            "extracted_data": result.final_data or {},
            "confidence": result.overall_confidence,
            "strategy_used": result.strategy_used,
            "agents_used": result.agents_used,
            "processing_time": result.total_processing_time,
            "total_tokens_used": result.total_tokens_used,
            "validation_errors": result.validation_errors,
            "content_filename": content_file.filename,
            "schema_filename": schema_file.filename
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in schema file")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Could not read files as text")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agentic extraction failed: {str(e)}")


@router.get("/compare/traditional-vs-agentic")
async def compare_extraction_approaches():
    """Compare traditional vs agentic extraction approaches."""
    return {
        "comparison": {
            "traditional": {
                "description": "Single LLM approach with strategy-based processing",
                "strategies": ["simple", "chunked", "hierarchical"],
                "pros": [
                    "Fast execution",
                    "Lower token usage", 
                    "Simpler architecture",
                    "Predictable behavior"
                ],
                "cons": [
                    "Limited error correction",
                    "No consensus building",
                    "Single point of failure",
                    "Less robust for complex cases"
                ],
                "best_for": [
                    "Simple schemas",
                    "Small to medium documents",
                    "Quick processing needs",
                    "Cost-sensitive applications"
                ]
            },
            "agentic": {
                "description": "Multi-agent collaborative approach with specialized roles",
                "strategies": ["parallel", "hierarchical", "collaborative", "adaptive"],
                "pros": [
                    "Higher accuracy through consensus",
                    "Self-correction and validation",
                    "Specialized agent expertise",
                    "Robust error handling",
                    "Performance tracking",
                    "Conflict resolution"
                ],
                "cons": [
                    "Higher token usage",
                    "Slower execution",
                    "More complex architecture", 
                    "Higher computational cost"
                ],
                "best_for": [
                    "Complex schemas",
                    "Large documents",
                    "High accuracy requirements",
                    "Critical applications",
                    "Quality-sensitive use cases"
                ]
            }
        },
        "recommendations": {
            "use_traditional_when": [
                "Schema has < 20 properties",
                "Document is < 10 pages",
                "Speed is critical",
                "Budget constraints exist"
            ],
            "use_agentic_when": [
                "Schema has > 50 properties", 
                "Document is > 20 pages",
                "Accuracy is critical",
                "Complex validation needed",
                "Multiple data types involved"
            ]
        }
    } 