"""API request/response schemas."""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


# ============================================
# Request Schemas
# ============================================

class AnalysisRequest(BaseModel):
    """Request to start a new analysis."""
    
    data_source: str = Field(
        description="Data source type: 'portkey', 'csv', or 'json'"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Path to data file (for csv/json sources)"
    )
    start_date: Optional[datetime] = Field(
        default=None,
        description="Start date for Portkey log export"
    )
    end_date: Optional[datetime] = Field(
        default=None,
        description="End date for Portkey log export"
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum number of prompts to analyze"
    )
    models: Optional[list[str]] = Field(
        default=None,
        description="List of model IDs to compare (uses defaults if not specified)"
    )
    current_model: str = Field(
        default="gpt-4o",
        description="Current production model for comparison baseline"
    )
    include_bertscore: bool = Field(
        default=True,
        description="Include BERTScore semantic similarity"
    )
    include_deepeval: bool = Field(
        default=False,
        description="Include DeepEval LLM-as-judge (slower, more expensive)"
    )


class ReportRequest(BaseModel):
    """Request to generate a report."""
    
    analysis_id: str = Field(description="ID of the analysis to generate report for")
    format: str = Field(
        default="html",
        description="Report format: 'html', 'json', or 'markdown'"
    )
    include_charts: bool = Field(
        default=True,
        description="Include interactive Plotly charts"
    )


# ============================================
# Response Schemas
# ============================================

class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"


class AnalysisStartResponse(BaseModel):
    """Response when analysis is started."""
    
    analysis_id: str
    status: str = "started"
    message: str
    estimated_time_seconds: Optional[int] = None


class AnalysisStatusResponse(BaseModel):
    """Response for analysis status check."""
    
    analysis_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress_percent: Optional[float] = None
    current_step: Optional[str] = None
    error: Optional[str] = None
    completed_at: Optional[datetime] = None


class ModelComparisonResponse(BaseModel):
    """Response with model comparison data."""
    
    model_id: str
    provider: str
    sample_size: int
    mean_quality_score: float
    quality_ci_lower: float
    quality_ci_upper: float
    total_cost_usd: float
    mean_cost_per_request: float
    mean_latency_ms: float
    is_pareto_optimal: bool


class RecommendationResponse(BaseModel):
    """Response with optimization recommendation."""
    
    analysis_id: str
    executive_summary: str
    current_monthly_cost: float
    recommended_monthly_cost: float
    cost_reduction_percent: float
    quality_impact_percent: float
    recommended_routing: dict
    segments: list[dict]


class ParetoChartResponse(BaseModel):
    """Response with Pareto chart data."""
    
    chart_html: str  # Interactive Plotly HTML
    chart_data: dict  # Raw data for custom rendering


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================
# Portkey Routing Config Schema
# ============================================

class PortkeyRoutingConfig(BaseModel):
    """Portkey Gateway routing configuration."""
    
    strategy: dict = Field(
        description="Routing strategy configuration"
    )
    retry: Optional[dict] = Field(
        default=None,
        description="Retry configuration"
    )
    cache: Optional[dict] = Field(
        default=None,
        description="Cache configuration"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "strategy": {
                    "mode": "conditional",
                    "conditions": [
                        {
                            "query": "$.metadata.complexity == 'simple'",
                            "then": "@openai/gpt-4o-mini"
                        },
                        {
                            "query": "$.metadata.complexity == 'complex'",
                            "then": "@openai/gpt-4o"
                        }
                    ],
                    "default": "@openai/gpt-4o-mini"
                },
                "retry": {
                    "attempts": 3,
                    "on_status_codes": [429, 500, 502, 503]
                },
                "cache": {
                    "mode": "semantic",
                    "max_age": 3600
                }
            }
        }


def generate_portkey_routing_config(
    simple_model: str = "@openai/gpt-4o-mini",
    medium_model: str = "@anthropic/claude-sonnet-4-5",
    complex_model: str = "@openai/gpt-4o",
    default_model: str = "@openai/gpt-4o-mini",
) -> dict:
    """
    Generate Portkey routing configuration from optimization recommendation.
    Can be directly imported into Portkey Gateway.
    
    Args:
        simple_model: Model slug for simple prompts
        medium_model: Model slug for medium complexity prompts
        complex_model: Model slug for complex prompts
        default_model: Default model when complexity is unknown
        
    Returns:
        Portkey-compatible routing configuration dict
    """
    return {
        "strategy": {
            "mode": "conditional",
            "conditions": [
                {
                    "query": "$.metadata.complexity == 'simple'",
                    "then": simple_model
                },
                {
                    "query": "$.metadata.complexity == 'medium'",
                    "then": medium_model
                },
                {
                    "query": "$.metadata.complexity == 'complex'",
                    "then": complex_model
                }
            ],
            "default": default_model
        },
        "retry": {
            "attempts": 3,
            "on_status_codes": [429, 500, 502, 503]
        },
        "cache": {
            "mode": "semantic",
            "max_age": 3600
        }
    }
