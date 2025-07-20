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
                  os.getenv("OPENAI_API_KEY_SECRET") or 
                  os.getenv("OPENAI_KEY"))
        
        if not api_key:
            return None, False
            
        return SimplifiedMetaExtract(api_key), True
    except Exception as e:
        print(f"Failed to initialize extractor: {e}")
        return None, False

# Initialize on startup but don't fail if not available
extractor, has_openai_key = get_extractor()


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
            schema=request.schema,
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