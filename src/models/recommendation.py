"""Recommendation schemas."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ConfidenceInterval(BaseModel):
    """Statistical confidence interval."""

    value: float
    lower: float
    upper: float
    confidence_level: float = 0.95
    sample_size: int


class Verdict(str, Enum):
    """Recommendation verdict."""

    STRONG_RECOMMENDATION = "strong_recommendation"
    RECOMMENDATION = "recommendation"
    CONSIDER = "consider"
    NO_CHANGE = "no_change"
    NOT_RECOMMENDED = "not_recommended"


class ModelComparison(BaseModel):
    """Comparison between two models."""

    model_a: str
    model_b: str
    
    # Quality comparison
    quality_a: ConfidenceInterval
    quality_b: ConfidenceInterval
    quality_difference: ConfidenceInterval
    quality_difference_percent: float
    quality_is_significant: bool
    quality_p_value: float

    # Cost comparison
    cost_a_monthly: float
    cost_b_monthly: float
    cost_difference: float
    cost_difference_percent: float

    # Trade-off
    cost_per_quality_point: float  # How much cost savings per 1% quality loss
    
    # Verdict
    verdict: Verdict
    explanation: str


class SegmentRecommendation(BaseModel):
    """Recommendation for a specific segment."""

    segment_name: str
    segment_description: str
    volume_percent: float  # What % of total traffic is this segment
    volume_count: int

    # Current state
    current_model: str
    current_cost_monthly: float
    current_quality: ConfidenceInterval

    # Recommended state
    recommended_model: str
    recommended_cost_monthly: float
    recommended_quality: ConfidenceInterval

    # Impact
    cost_reduction_percent: float
    cost_savings_monthly: float
    quality_impact_percent: float

    # Confidence
    verdict: Verdict
    confidence: str  # "high", "medium", "low"
    reasoning: str

    # Risks
    risks: list[str] = Field(default_factory=list)
    mitigations: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    """Complete optimization recommendation."""

    # Analysis metadata
    analysis_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    total_prompts_analyzed: int
    models_evaluated: list[str]

    # Executive summary
    executive_summary: str
    
    # Current state
    current_monthly_cost: float
    current_avg_quality: ConfidenceInterval

    # Recommended state
    recommended_monthly_cost: float
    recommended_avg_quality: ConfidenceInterval

    # Overall impact
    total_cost_reduction_percent: float
    total_cost_savings_monthly: float
    total_quality_impact_percent: float

    # Segment breakdown
    segments: list[SegmentRecommendation] = Field(default_factory=list)

    # Model comparisons
    pareto_frontier_models: list[str]
    dominated_models: list[str]
    model_comparisons: list[ModelComparison] = Field(default_factory=list)

    # Implementation
    implementation_complexity: str  # "low", "medium", "high"
    implementation_steps: list[str] = Field(default_factory=list)
    estimated_implementation_time: str

    # Risks
    overall_risks: list[str] = Field(default_factory=list)
    overall_mitigations: list[str] = Field(default_factory=list)

    # Portkey routing config (if applicable)
    portkey_routing_config: dict | None = None


class AnalysisRequest(BaseModel):
    """Request to run a cost-quality analysis."""

    # Data source
    source_type: str = "portkey"  # "portkey", "csv", "json"
    source_path: str | None = None  # For file-based sources

    # Time range (for Portkey)
    start_date: datetime | None = None
    end_date: datetime | None = None

    # Sampling
    sample_size: int | None = None
    sampling_strategy: str = "random"  # "random", "stratified", "recent"

    # Models to evaluate
    models_to_test: list[str] | None = None  # None = use default set

    # Evaluation options
    enable_deepeval: bool = True
    enable_bertscore: bool = True
    deepeval_metrics: list[str] = Field(
        default_factory=lambda: ["answer_relevancy", "faithfulness"]
    )

    # Segmentation
    enable_segmentation: bool = True
    segmentation_method: str = "complexity"  # "complexity", "task_type", "custom"

    # Output options
    generate_routing_config: bool = True


class AnalysisStatus(BaseModel):
    """Status of a running analysis."""

    analysis_id: str
    status: str  # "pending", "ingesting", "replaying", "evaluating", "analyzing", "complete", "failed"
    progress_percent: float = 0.0
    current_step: str = ""
    
    # Counts
    prompts_ingested: int = 0
    replays_completed: int = 0
    evaluations_completed: int = 0
    
    # Timing
    started_at: datetime
    estimated_completion: datetime | None = None
    completed_at: datetime | None = None

    # Errors
    errors: list[str] = Field(default_factory=list)
