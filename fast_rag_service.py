#!/usr/bin/env python3
"""
Fast RAG Service: Production API with an approximate 1 to 3 second latency target.

Usage:
    python fast_rag_service.py --profile balanced
    python fast_rag_service.py --port 8080 --workers 1
"""

import os
import sys
import time
from typing import Dict, Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
import uvicorn

from fast_rag_pipeline import FastRAGPipeline
from fast_rag_config import FastRAGConfig, DeploymentProfiles


# ============================================================================ #
# API Models                                                                   #
# ============================================================================ #

class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., min_length=1, max_length=500, description="User query")
    top_k: Optional[int] = Field(
        None,
        ge=1,
        le=FastRAGConfig.TOP_K,
        description="Number of documents to retrieve"
    )
    timeout: Optional[float] = Field(
        None,
        ge=0.5,
        le=5.0,
        description="Soft timeout requested by client in seconds"
    )


class SourceDocument(BaseModel):
    """Retrieved source document."""
    rank: int
    text: str
    score: float
    metadata: Dict


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    sources: List[SourceDocument]
    latency: Dict[str, float]
    status: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    uptime_seconds: float
    total_queries: int
    avg_latency_ms: float
    config: Dict


class StatsResponse(BaseModel):
    """Statistics response."""
    total_queries: int
    avg_retrieval_ms: float
    avg_generation_ms: float
    avg_total_ms: float
    cache_hits: int
    config: Dict


# ============================================================================ #
# FastAPI Application                                                          #
# ============================================================================ #

app = FastAPI(
    title="Fast RAG Service",
    description="RAG service with a 1 to 3 second latency target",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
pipeline: Optional[FastRAGPipeline] = None
start_time = time.time()


# ============================================================================ #
# Startup and Shutdown                                                         #
# ============================================================================ #

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup."""
    global pipeline

    print("\n" + "=" * 70)
    print("Starting Fast RAG Service")
    print("=" * 70)

    try:
        pipeline = FastRAGPipeline(verbose=False)
        print("\nService is ready to handle requests")
        print("API docs at http://localhost:8000/docs")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        sys.exit(1)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("\nShutting down Fast RAG Service")


# ============================================================================ #
# Endpoints                                                                    #
# ============================================================================ #

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Fast RAG Service",
        "version": "1.0.0",
        "status": "running",
        "target_latency": "1 to 3 seconds",
        "endpoints": {
            "query": "/query (POST)",
            "health": "/health (GET)",
            "stats": "/stats (GET)",
            "docs": "/docs (GET)"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    stats = pipeline.get_stats()
    uptime = time.time() - start_time

    return HealthResponse(
        status="healthy",
        uptime_seconds=round(uptime, 1),
        total_queries=stats["total_queries"],
        avg_latency_ms=round(stats["avg_total_ms"], 1),
        config=stats["config"]
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get detailed statistics."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    stats = pipeline.get_stats()

    return StatsResponse(
        total_queries=stats["total_queries"],
        avg_retrieval_ms=round(stats["avg_retrieval_ms"], 1),
        avg_generation_ms=round(stats["avg_generation_ms"], 1),
        avg_total_ms=round(stats["avg_total_ms"], 1),
        cache_hits=stats["cache_hits"],
        config=stats["config"]
    )


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Process a RAG query with a strong latency constraint.

    Effective timeout is clipped to the configured service timeout.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        default_timeout = FastRAGConfig.REQUEST_TIMEOUT
        effective_timeout = min(request.timeout or default_timeout, default_timeout)

        result = await run_in_threadpool(
            pipeline.query,
            request.query,
            request.top_k,
            effective_timeout
        )

        if result["latency"]["total_ms"] > 3000:
            print(
                f"Slow query "
                f"({result['latency']['total_ms']:.0f} ms): {request.query[:50]}"
            )

        sources = [
            SourceDocument(
                rank=doc["rank"],
                text=doc["text"][:500],
                score=doc["score"],
                metadata=doc.get("metadata", {})
            )
            for doc in result.get("sources", [])
        ]

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            latency=result["latency"],
            status=result.get("status", "success"),
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset-stats")
async def reset_stats():
    """Reset performance statistics."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    pipeline.reset_stats()
    return {"message": "Statistics reset successfully"}


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start = time.time()
    response = await call_next(request)
    process_time = (time.time() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.1f}"
    return response


# ============================================================================ #
# CLI                                                                          #
# ============================================================================ #

def main():
    """Run the Fast RAG service."""
    import argparse

    parser = argparse.ArgumentParser(description="Fast RAG Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument(
        "--profile",
        choices=["ultra_fast", "balanced", "quality", "gpu"],
        default="balanced",
        help="Performance profile"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto reload for development"
    )

    args = parser.parse_args()

    print(f"Applying profile: {args.profile}")
    if args.profile == "ultra_fast":
        DeploymentProfiles.ultra_fast()
        print("Target latency near 1.5 seconds")
    elif args.profile == "balanced":
        DeploymentProfiles.balanced()
        print("Target latency near 3 seconds")
    elif args.profile == "quality":
        DeploymentProfiles.quality()
        print("Target latency near 5 seconds")
    elif args.profile == "gpu":
        DeploymentProfiles.gpu_optimized()
        print("Target latency near 2 seconds")

    uvicorn.run(
        "fast_rag_service:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
