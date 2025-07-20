"""
API route handlers for MetaExtract.
"""
import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, status
from fastapi.responses import JSONResponse
import aiofiles
import os

from .models import (
    ExtractionRequest, FileExtractionRequest, ExtractionResult, 
    SchemaAnalysisRequest, SchemaAnalysisResult, HealthResponse,
    ExtractionStatus, FieldConfidence, ExtractionStrategy
)
from .config import settings

# Import simplified extractor
from metaextract.simplified_extractor import SimplifiedMetaExtract, SchemaComplexity

router = APIRouter()

# In-memory job storage (in production, use Redis or database)
extraction_jobs: Dict[str, Dict[str, Any]] = {}


class MetaExtractService:
    """Simplified service class using the new simplified extractor."""
    
    def __init__(self):
        try:
            self.extractor = SimplifiedMetaExtract()
            self.has_api_key = True
        except ValueError as e:
            # Handle missing API key gracefully - create with dummy key for schema analysis
            print(f"Warning: {e}")
            try:
                self.extractor = SimplifiedMetaExtract("dummy-key")  # For schema analysis only
                self.has_api_key = False
                print("Schema analysis available without API key")
            except Exception as e2:
                print(f"Failed to create analyzer: {e2}")
                self.extractor = None
                self.has_api_key = False
    
    async def extract_from_text(self, request: ExtractionRequest) -> ExtractionResult:
        """Extract structured data from text input using simplified extractor."""
        if not self.extractor:
            raise HTTPException(status_code=503, detail="Extractor not available. Please check system configuration.")
        if not self.has_api_key:
            raise HTTPException(status_code=503, detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
        
        start_time = time.time()
        
        try:
            # Get input text
            if request.input_text:
                input_text = request.input_text
            elif request.input_url:
                # In a real implementation, fetch content from URL
                raise HTTPException(status_code=501, detail="URL extraction not implemented yet")
            else:
                raise HTTPException(status_code=400, detail="Either input_text or input_url must be provided")
            
            # Convert strategy enum to string if needed
            strategy_str = request.strategy.value if hasattr(request.strategy, 'value') else str(request.strategy)
            if strategy_str == "auto":
                strategy_str = None  # Let the extractor auto-select
            
            # Use simplified extractor
            result = await self.extractor.extract(
                input_text=input_text,
                schema=request.schema,
                strategy=strategy_str
            )
            
            # Convert result to API format
            if result.success:
                # Build field confidences in expected format
                field_confidences = []
                # Note: result.low_confidence_fields is just a list of field names
                # We'll create minimal field confidence objects
                for field_path in result.low_confidence_fields:
                    field_confidences.append(FieldConfidence(
                        field_path=field_path,
                        value=self._get_nested_value(result.extracted_data, field_path),
                        confidence=0.5,  # Default low confidence value
                        validation_errors=[]
                    ))
                
                # Create schema complexity info
                extractor_complexity = self.extractor._analyze_schema_complexity(request.schema)
                complexity_dict = {
                    "complexity_level": "complex" if extractor_complexity.is_complex else "simple",
                    "max_nesting_depth": extractor_complexity.nesting_depth,
                    "total_objects": extractor_complexity.total_objects,
                    "total_properties": extractor_complexity.total_properties,
                    "complexity_score": extractor_complexity.complexity_score
                }
                
                return ExtractionResult(
                    success=True,
                    extracted_data=result.extracted_data,
                    strategy_used=result.strategy_used,
                    schema_complexity=complexity_dict,
                    processing_time=result.processing_time,
                    chunk_count=result.chunks_processed,
                    agent_count=1,  # Simplified approach uses 1 agent
                    overall_confidence=result.confidence_score,
                    field_confidences=field_confidences,
                    validation_errors=result.validation_errors,
                    low_confidence_fields=result.low_confidence_fields,
                    warnings=[]
                )
            else:
                return ExtractionResult(
                    success=False,
                    extracted_data=None,
                    strategy_used=result.strategy_used,
                    schema_complexity={},
                    processing_time=result.processing_time,
                    chunk_count=0,
                    agent_count=0,
                    overall_confidence=0.0,
                    field_confidences=[],
                    validation_errors=[],
                    low_confidence_fields=[],
                    error_message=result.error_message
                )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ExtractionResult(
                success=False,
                extracted_data=None,
                strategy_used=request.strategy.value if hasattr(request.strategy, 'value') else str(request.strategy),
                schema_complexity={},
                processing_time=processing_time,
                chunk_count=0,
                agent_count=0,
                overall_confidence=0.0,
                field_confidences=[],
                validation_errors=[],
                low_confidence_fields=[],
                error_message=str(e)
            )
    
    def analyze_schema(self, schema: Dict[str, Any]) -> SchemaAnalysisResult:
        """Analyze schema complexity using simplified analyzer."""
        if not self.extractor:
            raise HTTPException(status_code=503, detail="Schema analyzer not available. Please check system configuration.")
        
        try:
            complexity = self.extractor._analyze_schema_complexity(schema)
            
            # Map to strategy
            if complexity.is_complex:
                if complexity.nesting_depth > 5 or complexity.total_objects > 50:
                    recommended_strategy = "hierarchical"
                else:
                    recommended_strategy = "chunked"
            else:
                recommended_strategy = "simple"
            
            # Estimate processing time based on complexity
            estimated_time = min(2.0 + (complexity.complexity_score / 25), 30.0)
            
            return SchemaAnalysisResult(
                complexity_metrics={
                    "complexity_level": "complex" if complexity.is_complex else "simple",
                    "max_nesting_depth": complexity.nesting_depth,
                    "total_objects": complexity.total_objects,
                    "total_properties": complexity.total_properties,
                    "complexity_score": complexity.complexity_score,
                    "estimated_tokens": complexity.estimated_tokens,
                    "has_complex_types": complexity.has_complex_types
                },
                recommended_strategy=recommended_strategy,
                estimated_processing_time=estimated_time,
                resource_requirements={
                    "agents_needed": 1,
                    "memory_estimate": f"{max(256, complexity.estimated_tokens // 100)}MB",
                    "chunking_recommended": complexity.is_complex
                }
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Schema analysis failed: {str(e)}")
    
    def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get nested value from data using field path."""
        try:
            keys = field_path.split('.')
            value = data
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value
        except:
            return None


# Global service instance
service = MetaExtractService()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        components={
            "simplified_extractor": "healthy" if service.extractor else "missing_api_key",
            "openai_integration": "configured" if service.extractor else "not_configured"
        }
    )


@router.post("/extract", response_model=ExtractionResult)
async def extract_text(request: ExtractionRequest):
    """Extract structured data from text input."""
    try:
        result = await service.extract_from_text(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.post("/extract/file", response_model=ExtractionResult)
async def extract_file(
    file: UploadFile = File(...),
    schema: str = None,
    strategy: str = "auto",
    chunk_size: int = 4000,
    overlap_size: int = 200,
    confidence_threshold: float = 0.7,
    max_agents: int = 5
):
    """Extract structured data from uploaded file."""
    try:
        # Validate file
        if file.size > settings.max_file_size:
            raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {settings.max_file_size} bytes")
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in settings.allowed_file_types:
            raise HTTPException(status_code=400, detail=f"File type {file_ext} not supported")
        
        # Parse schema
        if not schema:
            raise HTTPException(status_code=400, detail="Schema parameter is required")
        
        try:
            schema_dict = json.loads(schema)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON schema")
        
        # Read file content
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Create extraction request
        request = ExtractionRequest(
            input_text=text_content,
            schema=schema_dict,
            strategy=ExtractionStrategy(strategy),
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            confidence_threshold=confidence_threshold,
            max_agents=max_agents
        )
        
        # Perform extraction
        result = await service.extract_from_text(request)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File extraction failed: {str(e)}")


@router.post("/analyze-schema", response_model=SchemaAnalysisResult)
async def analyze_schema(request: SchemaAnalysisRequest):
    """Analyze schema complexity and get recommendations."""
    try:
        result = service.analyze_schema(request.schema)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schema analysis failed: {str(e)}")


@router.post("/extract/async", response_model=Dict[str, str])
async def extract_async(request: ExtractionRequest, background_tasks: BackgroundTasks):
    """Start asynchronous extraction job."""
    job_id = str(uuid.uuid4())
    
    # Store job info
    extraction_jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "result": None
    }
    
    # Start background task
    background_tasks.add_task(run_async_extraction, job_id, request)
    
    return {"job_id": job_id, "status": "accepted"}


@router.get("/extract/status/{job_id}", response_model=ExtractionStatus)
async def get_extraction_status(job_id: str):
    """Get status of async extraction job."""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = extraction_jobs[job_id]
    
    return ExtractionStatus(
        job_id=job_id,
        status=job_data["status"],
        progress=job_data["progress"],
        result=job_data.get("result"),
        created_at=job_data["created_at"],
        updated_at=job_data["updated_at"]
    )


async def run_async_extraction(job_id: str, request: ExtractionRequest):
    """Background task for async extraction."""
    try:
        # Update status
        extraction_jobs[job_id].update({
            "status": "processing",
            "progress": 0.1,
            "updated_at": datetime.utcnow().isoformat()
        })
        
        # Simulate progress updates
        for progress in [0.3, 0.5, 0.7]:
            await asyncio.sleep(0.5)  # Shorter delays for demo
            extraction_jobs[job_id].update({
                "progress": progress,
                "updated_at": datetime.utcnow().isoformat()
            })
        
        # Perform actual extraction
        result = await service.extract_from_text(request)
        
        # Update with final result
        extraction_jobs[job_id].update({
            "status": "completed",
            "progress": 1.0,
            "updated_at": datetime.utcnow().isoformat(),
            "result": result
        })
        
    except Exception as e:
        extraction_jobs[job_id].update({
            "status": "failed",
            "updated_at": datetime.utcnow().isoformat(),
            "result": ExtractionResult(
                success=False,
                extracted_data=None,
                strategy_used="unknown",
                schema_complexity={},
                processing_time=0,
                chunk_count=0,
                agent_count=0,
                overall_confidence=0.0,
                field_confidences=[],
                validation_errors=[],
                low_confidence_fields=[],
                error_message=str(e)
            )
        })


@router.get("/strategies")
async def list_strategies():
    """Get available extraction strategies and their descriptions."""
    return {
        "simple": {
            "name": "Simple",
            "description": "Single LLM call for straightforward schemas",
            "best_for": "Simple schemas with 1-3 nesting levels",
            "estimated_time": "Fast (1-5s)"
        },
        "chunked": {
            "name": "Chunked",
            "description": "Break large documents into manageable chunks",
            "best_for": "Large documents or moderate complexity schemas",
            "estimated_time": "Medium (5-15s)"
        },
        "hierarchical": {
            "name": "Hierarchical", 
            "description": "Process complex schemas in sections",
            "best_for": "Complex schemas with 4+ nesting levels or 20+ objects",
            "estimated_time": "Slower (10-30s)"
        }
    }


@router.get("/complexity-thresholds")
async def get_complexity_thresholds():
    """Get schema complexity thresholds for strategy selection."""
    return {
        "simple_threshold": {
            "max_nesting_depth": 3,
            "max_objects": 20,
            "max_complexity_score": 50
        },
        "chunked_threshold": {
            "max_document_size": "100KB",
            "recommended_for": "Large documents or moderate complexity"
        },
        "hierarchical_threshold": {
            "min_nesting_depth": 4,
            "min_objects": 20,
            "min_complexity_score": 50,
            "recommended_for": "Complex schemas with many nested objects"
        }
    } 