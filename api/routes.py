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
from metaextract.core.schema_analyzer import SchemaComplexityAnalyzer
from metaextract.core.strategy_selector import AdaptiveStrategySelector, StrategyConfig
from metaextract.core.orchestrator import MultiAgentOrchestrator
from metaextract.core.schema_processor import HierarchicalSchemaProcessor
from metaextract.core.document_chunker import LargeDocumentChunker
from metaextract.core.validation_engine import ValidationEngine

router = APIRouter()

# In-memory job storage (in production, use Redis or database)
extraction_jobs: Dict[str, Dict[str, Any]] = {}


class MetaExtractService:
    """Service class that orchestrates all MetaExtract components."""
    
    def __init__(self):
        self.analyzer = SchemaComplexityAnalyzer()
        self.strategy_selector = AdaptiveStrategySelector()
        self.orchestrator = MultiAgentOrchestrator()
        self.schema_processor = HierarchicalSchemaProcessor()
        self.document_chunker = LargeDocumentChunker()
        self.validator = ValidationEngine()
    
    async def extract_from_text(self, request: ExtractionRequest) -> ExtractionResult:
        """Extract structured data from text input."""
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
            
            # Analyze schema complexity
            complexity = self.analyzer.analyze_complexity(request.schema)
            
            # Select strategy if auto
            if request.strategy == ExtractionStrategy.AUTO:
                config = StrategyConfig(
                    chunk_size=request.chunk_size,
                    overlap_size=request.overlap_size,
                    confidence_threshold=request.confidence_threshold,
                    max_agents=request.max_agents
                )
                strategy_result = self.strategy_selector.select_strategy(
                    request.schema, len(input_text), config
                )
                selected_strategy = strategy_result.recommended_strategy
            else:
                selected_strategy = request.strategy.value
            
            # Process the extraction based on strategy
            if selected_strategy in ["hierarchical_chunking", "multi_agent_parallel", "multi_agent_sequential", "hybrid"]:
                # Use document chunking for complex strategies
                chunks = self.document_chunker.chunk_document(input_text, request.chunk_size, request.overlap_size)
                chunk_count = len(chunks.text_chunks) + len(chunks.table_chunks) + len(chunks.structured_chunks)
            else:
                chunk_count = 1
            
            # Perform extraction using orchestrator
            extraction_result = await self.orchestrator.extract_data(
                input_text, request.schema, selected_strategy, {
                    "chunk_size": request.chunk_size,
                    "overlap_size": request.overlap_size,
                    "confidence_threshold": request.confidence_threshold,
                    "max_agents": request.max_agents
                }
            )
            
            # Validate results
            validation_result = self.validator.validate_data(extraction_result["extracted_data"], request.schema)
            
            # Build field confidences
            field_confidences = []
            for field_path, confidence in extraction_result.get("field_confidences", {}).items():
                field_confidences.append(FieldConfidence(
                    field_path=field_path,
                    value=self._get_nested_value(extraction_result["extracted_data"], field_path),
                    confidence=confidence,
                    validation_errors=validation_result.field_errors.get(field_path, [])
                ))
            
            processing_time = time.time() - start_time
            
            return ExtractionResult(
                success=True,
                extracted_data=extraction_result["extracted_data"],
                strategy_used=selected_strategy,
                schema_complexity=complexity.to_dict(),
                processing_time=processing_time,
                chunk_count=chunk_count,
                agent_count=extraction_result.get("agents_used", 1),
                overall_confidence=extraction_result.get("overall_confidence", 0.8),
                field_confidences=field_confidences,
                validation_errors=validation_result.validation_errors,
                low_confidence_fields=validation_result.low_confidence_fields,
                warnings=extraction_result.get("warnings", [])
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ExtractionResult(
                success=False,
                extracted_data=None,
                strategy_used=request.strategy.value,
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
            "schema_analyzer": "healthy",
            "strategy_selector": "healthy", 
            "orchestrator": "healthy",
            "validator": "healthy"
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
        complexity = service.analyzer.analyze_complexity(request.schema)
        
        # Get strategy recommendation
        config = StrategyConfig()
        strategy_result = service.strategy_selector.select_strategy(
            request.schema, 5000, config  # Assume medium-sized input
        )
        
        return SchemaAnalysisResult(
            complexity_metrics=complexity.to_dict(),
            recommended_strategy=strategy_result.recommended_strategy,
            estimated_processing_time=strategy_result.estimated_time_seconds,
            resource_requirements={
                "agents_needed": strategy_result.agents_needed,
                "memory_estimate": f"{strategy_result.memory_estimate_mb}MB",
                "chunking_recommended": strategy_result.chunking_recommended
            }
        )
        
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
        for progress in [0.3, 0.5, 0.7, 0.9]:
            await asyncio.sleep(1)  # Simulate processing time
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
    from .config import get_strategy_config
    return get_strategy_config()


@router.get("/complexity-thresholds")
async def get_complexity_thresholds():
    """Get schema complexity thresholds for strategy selection."""
    from .config import get_complexity_thresholds
    return get_complexity_thresholds() 