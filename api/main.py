#!/usr/bin/env python3
"""
Production FastAPI server for GPU-accelerated RAG
Designed for React frontend integration
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import sys
import time
import logging
from datetime import datetime
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import RAG pipeline
sys.path.insert(0, '/mnt/d/Roshidat_Msc_Project/AI_Project/AI_Project/CleanInferenceRAG')
from fast_rag_pipeline import FastRAGPipeline
from fast_rag_config import FastRAGConfig

# Initialize FastAPI
app = FastAPI(
    title="Caramella RAG API",
    description="GPU-accelerated Retrieval-Augmented Generation API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS Configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*",  # Allow all for development (restrict in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User question")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Number of documents to retrieve")
    include_sources: Optional[bool] = Field(True, description="Include source documents in response")

class SourceDocument(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceDocument] = []
    performance: Dict[str, float]
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    gpu_enabled: bool
    vector_db: str
    collection: str
    model: str
    uptime_seconds: float

class StatsResponse(BaseModel):
    total_queries: int
    avg_latency_ms: float
    avg_retrieval_ms: float
    avg_generation_ms: float
    gpu_enabled: bool

# Global state
class APIState:
    pipeline: Optional[FastRAGPipeline] = None
    start_time: float = time.time()
    query_count: int = 0
    total_latency: float = 0.0
    total_retrieval: float = 0.0
    total_generation: float = 0.0

state = APIState()

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup"""
    logger.info("🚀 Starting Caramella RAG API...")
    
    try:
        # Configure settings (GPU mode for <3s latency)
        config = FastRAGConfig()
        # Safer GPU defaults for 6GB VRAM unless overridden by env
        gpu_layers = int(os.getenv("RAG_GPU_LAYERS", os.getenv("LLM_GPU_LAYERS", "20")))
        batch_size = int(os.getenv("RAG_BATCH_SIZE", "256"))
        use_mlock = os.getenv("RAG_USE_MLOCK", "false").lower() == "true"

        config.LLM_GPU_LAYERS = gpu_layers
        config.LLM_BATCH_SIZE = batch_size
        config.LLM_USE_MLOCK = use_mlock
        config.EMBED_DEVICE = "cpu"  # Keep embeddings on CPU (stable)
        
        logger.info(f"📚 Loading vector database: {config.DB_PATH}/{config.COLLECTION_NAME}")
        logger.info(f"🤖 Loading model: {config.LLM_MODEL_PATH}")
        logger.info(f"⚡ GPU layers: {config.LLM_GPU_LAYERS} (GPU ENABLED)")
        logger.info(f"🧠 Batch size: {config.LLM_BATCH_SIZE} | mlock: {config.LLM_USE_MLOCK}")
        
        # Initialize pipeline
        state.pipeline = FastRAGPipeline(config=config, verbose=True)
        state.start_time = time.time()
        
        logger.info("✅ RAG Pipeline initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}")
        raise

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "name": "Caramella RAG API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/api/docs"
    }

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring"""
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    config = state.pipeline.config
    uptime = time.time() - state.start_time
    
    return HealthResponse(
        status="healthy",
        gpu_enabled=config.LLM_GPU_LAYERS > 0,
        vector_db=config.DB_PATH,
        collection=config.COLLECTION_NAME,
        model=config.LLM_MODEL_PATH,
        uptime_seconds=uptime
    )

@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(request: QueryRequest):
    """
    Query the RAG system
    
    - **query**: Your question (1-500 characters)
    - **top_k**: Number of documents to retrieve (1-10, default: 3)
    - **include_sources**: Include source documents in response (default: true)
    """
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        start_time = time.time()
        
        # Execute query
        result = state.pipeline.query(request.query, top_k=request.top_k)
        
        total_time = (time.time() - start_time) * 1000
        
        # Extract timing from pipeline result (uses 'latency' dict)
        latency = result.get('latency', {})
        retrieval_ms = latency.get('retrieval_ms', 0)
        generation_ms = latency.get('generation_ms', 0)
        
        # Update statistics
        state.query_count += 1
        state.total_latency += total_time
        state.total_retrieval += retrieval_ms
        state.total_generation += generation_ms
        
        # Format sources
        sources = []
        if request.include_sources and 'sources' in result:
            for src in result['sources']:
                sources.append(SourceDocument(
                    content=src.get('text', src.get('content', '')),
                    score=src.get('score', src.get('similarity', 0.0)),
                    metadata=src.get('metadata', {})
                ))
        
        # Performance metrics
        performance = {
            "total_ms": round(total_time, 2),
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
        }
        
        logger.info(f"Query processed in {total_time:.0f}ms (R: {performance['retrieval_ms']}ms, G: {performance['generation_ms']}ms)")
        
        return QueryResponse(
            query=request.query,
            answer=result.get('answer', ''),
            sources=sources,
            performance=performance,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/api/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats():
    """Get API statistics"""
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return StatsResponse(
        total_queries=state.query_count,
        avg_latency_ms=round(state.total_latency / max(state.query_count, 1), 2),
        avg_retrieval_ms=round(state.total_retrieval / max(state.query_count, 1), 2),
        avg_generation_ms=round(state.total_generation / max(state.query_count, 1), 2),
        gpu_enabled=state.pipeline.config.LLM_GPU_LAYERS > 0
    )

@app.post("/api/stats/reset", tags=["Stats"])
async def reset_stats():
    """Reset statistics"""
    state.query_count = 0
    state.total_latency = 0.0
    state.total_retrieval = 0.0
    state.total_generation = 0.0
    return {"message": "Statistics reset successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable in production
        log_level="info"
    )
