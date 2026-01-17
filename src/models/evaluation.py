"""Evaluation result schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CostResult(BaseModel):
    """Cost calculation result for a model response."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    model_id: str
    pricing_version: str = "2025-01"  # Track pricing version used


class MetricScore(BaseModel):
    """Individual metric evaluation result."""

    metric_name: str
    score: float = Field(ge=0.0, le=1.0)
    reason: str | None = None  # Self-explaining reason from LLM-as-judge
    passed: bool = True
    threshold: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Complete evaluation result for a single replay."""

    prompt_id: str
    model_id: str
    
    # Quality metrics
    metrics: dict[str, MetricScore] = Field(default_factory=dict)
    composite_score: float = Field(ge=0.0, le=1.0)
    
    # Cost & performance
    cost: CostResult
    latency_ms: float
    
    # Refusal tracking
    refused: bool = False
    refusal_reason: str | None = None
    
    # Metadata
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    evaluator_version: str = "1.0.0"

    @property
    def quality_score(self) -> float:
        """Alias for composite_score."""
        return self.composite_score


class ReplayResult(BaseModel):
    """Result of replaying a prompt through a model."""

    # Identifiers
    prompt_id: str
    replay_id: str
    model_id: str
    provider: str

    # Response
    completion: str
    finish_reason: str = "stop"

    # Token usage
    input_tokens: int
    output_tokens: int
    total_tokens: int

    # Performance
    latency_ms: float
    time_to_first_token_ms: float | None = None

    # Cost
    cost_usd: float

    # Status
    success: bool = True
    error: str | None = None
    refused: bool = False

    # Metadata
    replayed_at: datetime = Field(default_factory=datetime.utcnow)
    api_version: str | None = None
    temperature: float = 1.0


class ModelAggregateResult(BaseModel):
    """Aggregated results for a model across all prompts."""

    model_id: str
    provider: str
    
    # Sample info
    sample_size: int
    successful_replays: int
    failed_replays: int
    refusal_count: int
    refusal_rate: float

    # Quality (aggregated)
    mean_quality_score: float
    quality_std: float
    quality_ci_lower: float
    quality_ci_upper: float

    # Cost (aggregated)
    total_cost_usd: float
    mean_cost_per_request: float
    mean_input_tokens: float
    mean_output_tokens: float

    # Latency (aggregated)
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Per-metric breakdown
    metric_scores: dict[str, float] = Field(default_factory=dict)
