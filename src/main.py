"""Main FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from src.api.routes import router
from src.config import get_settings


def configure_logging():
    """Configure structured logging."""
    settings = get_settings()
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    configure_logging()
    logger = structlog.get_logger()
    logger.info("Starting Track 4 Optimizer API")
    yield
    logger.info("Shutting down Track 4 Optimizer API")


app = FastAPI(
    title="Track 4: Cost-Quality Optimizer",
    description="""
    A production-grade system for analyzing LLM cost-quality trade-offs 
    through historical prompt replay.
    
    ## Features
    
    - **Historical Replay**: Replay prompts through alternative models
    - **Multi-dimensional Evaluation**: Quality, cost, and latency metrics
    - **Pareto Analysis**: Find optimal cost-quality trade-offs
    - **Actionable Recommendations**: Quantified savings with quality impact
    
    ## Workflow
    
    1. POST `/api/v1/analyze` to start an analysis
    2. GET `/api/v1/analyze/{id}/status` to check progress
    3. GET `/api/v1/analyze/{id}/recommendation` to get results
    4. GET `/api/v1/analyze/{id}/report` for full HTML report
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Track 4: Cost-Quality Optimizer",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
