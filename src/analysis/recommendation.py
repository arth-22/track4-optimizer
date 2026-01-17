"""Recommendation engine for generating optimization recommendations."""

import uuid
from datetime import datetime
from typing import Any

import structlog

from src.analysis.pareto import ParetoAnalyzer
from src.analysis.segmentation import SegmentationEngine
from src.analysis.statistics import StatisticalAnalyzer
from src.models.canonical import CanonicalPrompt, ComplexityLevel
from src.models.evaluation import EvaluationResult, ModelAggregateResult
from src.models.recommendation import (
    ConfidenceInterval,
    ModelComparison,
    Recommendation,
    SegmentRecommendation,
    Verdict,
)
from src.replay.model_registry import ModelRegistry

logger = structlog.get_logger()


class RecommendationEngine:
    """
    Generates optimization recommendations from evaluation results.
    
    Combines:
    - Pareto analysis for optimal model selection
    - Segmentation for targeted recommendations
    - Statistical analysis for confidence
    """

    def __init__(self):
        self.stats = StatisticalAnalyzer()
        self.segmentation = SegmentationEngine()
        self.pareto = ParetoAnalyzer()

    def generate_recommendation(
        self,
        prompts: list[CanonicalPrompt],
        evaluations: list[EvaluationResult],
        current_model: str | None = None,
    ) -> Recommendation:
        """
        Generate comprehensive optimization recommendation.
        
        Args:
            prompts: Original prompts analyzed
            evaluations: Evaluation results for all model replays
            current_model: Current model in production (for comparison)
            
        Returns:
            Recommendation with segments and trade-offs
        """
        analysis_id = str(uuid.uuid4())

        # Aggregate results by model
        model_results = self._aggregate_by_model(evaluations)

        # Find Pareto frontier
        pareto_frontier = self.pareto.find_pareto_frontier(model_results)
        dominated = self.pareto.get_dominated_models(model_results, pareto_frontier)

        # Segment prompts and analyze each segment
        segments = self._analyze_segments(prompts, evaluations, current_model)

        # Calculate overall impact
        current_cost, recommended_cost = self._calculate_costs(
            segments, current_model, model_results
        )

        current_quality = self._get_model_quality(current_model, model_results)
        recommended_quality = self._get_recommended_quality(segments)

        # Generate executive summary
        summary = self._generate_summary(
            current_cost, recommended_cost,
            current_quality, recommended_quality,
            len(prompts), len(model_results),
        )

        # Generate model comparisons
        comparisons = self._generate_comparisons(model_results, current_model)

        # Generate Portkey routing config
        routing_config = self._generate_routing_config(segments)

        return Recommendation(
            analysis_id=analysis_id,
            total_prompts_analyzed=len(prompts),
            models_evaluated=[m.model_id for m in model_results],
            executive_summary=summary,
            current_monthly_cost=current_cost * 30,  # Assuming daily cost
            current_avg_quality=current_quality,
            recommended_monthly_cost=recommended_cost * 30,
            recommended_avg_quality=recommended_quality,
            total_cost_reduction_percent=self._calc_percent_change(
                current_cost, recommended_cost
            ),
            total_cost_savings_monthly=(current_cost - recommended_cost) * 30,
            total_quality_impact_percent=self._calc_percent_change(
                current_quality.value, recommended_quality.value
            ),
            segments=segments,
            pareto_frontier_models=[m.model_id for m in pareto_frontier],
            dominated_models=[m.model_id for m in dominated],
            model_comparisons=comparisons,
            implementation_complexity=self._assess_complexity(segments),
            implementation_steps=self._generate_steps(segments),
            estimated_implementation_time="1-2 weeks",
            overall_risks=self._identify_risks(segments),
            overall_mitigations=self._suggest_mitigations(),
            portkey_routing_config=routing_config,
        )

    def _aggregate_by_model(
        self,
        evaluations: list[EvaluationResult],
    ) -> list[ModelAggregateResult]:
        """Aggregate evaluation results by model."""
        from collections import defaultdict
        import numpy as np

        by_model = defaultdict(list)
        for eval_result in evaluations:
            by_model[eval_result.model_id].append(eval_result)

        aggregates = []
        for model_id, results in by_model.items():
            model_config = ModelRegistry.get_model(model_id)
            provider = model_config.provider.value if model_config else "unknown"

            quality_scores = [r.composite_score for r in results]
            costs = [r.cost.total_cost_usd for r in results]
            latencies = [r.latency_ms for r in results]
            refusals = sum(1 for r in results if r.refused)

            aggregates.append(ModelAggregateResult(
                model_id=model_id,
                provider=provider,
                sample_size=len(results),
                successful_replays=len(results) - refusals,
                failed_replays=0,  # Tracked separately
                refusal_count=refusals,
                refusal_rate=refusals / len(results) if results else 0,
                mean_quality_score=float(np.mean(quality_scores)),
                quality_std=float(np.std(quality_scores)),
                quality_ci_lower=float(np.percentile(quality_scores, 2.5)),
                quality_ci_upper=float(np.percentile(quality_scores, 97.5)),
                total_cost_usd=sum(costs),
                mean_cost_per_request=float(np.mean(costs)),
                mean_input_tokens=float(np.mean([r.cost.input_tokens for r in results])),
                mean_output_tokens=float(np.mean([r.cost.output_tokens for r in results])),
                mean_latency_ms=float(np.mean(latencies)),
                p50_latency_ms=float(np.percentile(latencies, 50)),
                p95_latency_ms=float(np.percentile(latencies, 95)),
                p99_latency_ms=float(np.percentile(latencies, 99)),
            ))

        return aggregates

    def _analyze_segments(
        self,
        prompts: list[CanonicalPrompt],
        evaluations: list[EvaluationResult],
        current_model: str | None,
    ) -> list[SegmentRecommendation]:
        """Analyze each segment and generate recommendations."""
        # Segment by complexity
        segments = self.segmentation.segment_by_complexity(prompts)
        segment_recs = []

        total_prompts = len(prompts)

        for complexity, segment_prompts in segments.items():
            if not segment_prompts:
                continue

            segment_ids = {p.id for p in segment_prompts}
            segment_evals = [e for e in evaluations if e.prompt_id in segment_ids]

            if not segment_evals:
                continue

            # Find best model for this segment
            model_results = self._aggregate_by_model(segment_evals)
            if not model_results:
                continue

            # Sort by quality-adjusted cost (heuristic)
            def score_model(m: ModelAggregateResult) -> float:
                # Higher quality, lower cost = better
                return m.mean_quality_score / (m.mean_cost_per_request + 0.0001)

            best_model = max(model_results, key=score_model)

            # Current model stats
            current = current_model or segment_prompts[0].completion.model_id
            current_results = [r for r in model_results if r.model_id == current]
            current_result = current_results[0] if current_results else best_model

            # Calculate impact
            cost_reduction = self._calc_percent_change(
                current_result.mean_cost_per_request,
                best_model.mean_cost_per_request,
            )
            quality_impact = self._calc_percent_change(
                current_result.mean_quality_score,
                best_model.mean_quality_score,
            )

            # Determine verdict
            verdict = self._determine_verdict(cost_reduction, quality_impact)

            segment_recs.append(SegmentRecommendation(
                segment_name=complexity.value.title(),
                segment_description=self._describe_segment(complexity),
                volume_percent=len(segment_prompts) / total_prompts * 100,
                volume_count=len(segment_prompts),
                current_model=current_result.model_id,
                current_cost_monthly=current_result.mean_cost_per_request * len(segment_prompts) * 30,
                current_quality=ConfidenceInterval(
                    value=current_result.mean_quality_score,
                    lower=current_result.quality_ci_lower,
                    upper=current_result.quality_ci_upper,
                    confidence_level=0.95,
                    sample_size=current_result.sample_size,
                ),
                recommended_model=best_model.model_id,
                recommended_cost_monthly=best_model.mean_cost_per_request * len(segment_prompts) * 30,
                recommended_quality=ConfidenceInterval(
                    value=best_model.mean_quality_score,
                    lower=best_model.quality_ci_lower,
                    upper=best_model.quality_ci_upper,
                    confidence_level=0.95,
                    sample_size=best_model.sample_size,
                ),
                cost_reduction_percent=cost_reduction,
                cost_savings_monthly=(
                    current_result.mean_cost_per_request - best_model.mean_cost_per_request
                ) * len(segment_prompts) * 30,
                quality_impact_percent=quality_impact,
                verdict=verdict,
                confidence="high" if best_model.sample_size >= 100 else "medium",
                reasoning=self._generate_reasoning(
                    complexity, current_result, best_model, cost_reduction, quality_impact
                ),
                risks=self._segment_risks(complexity, quality_impact),
                mitigations=self._segment_mitigations(complexity),
            ))

        return segment_recs

    def _describe_segment(self, complexity: ComplexityLevel) -> str:
        """Generate segment description."""
        descriptions = {
            ComplexityLevel.SIMPLE: "Short prompts with straightforward tasks",
            ComplexityLevel.MEDIUM: "Moderate complexity with multi-step reasoning",
            ComplexityLevel.COMPLEX: "Long context with complex requirements",
        }
        return descriptions.get(complexity, "Mixed complexity prompts")

    def _determine_verdict(
        self,
        cost_reduction: float,
        quality_impact: float,
    ) -> Verdict:
        """Determine recommendation verdict."""
        if cost_reduction > 50 and quality_impact > -5:
            return Verdict.STRONG_RECOMMENDATION
        elif cost_reduction > 20 and quality_impact > -10:
            return Verdict.RECOMMENDATION
        elif cost_reduction > 0 and quality_impact > -15:
            return Verdict.CONSIDER
        elif cost_reduction <= 0:
            return Verdict.NO_CHANGE
        else:
            return Verdict.NOT_RECOMMENDED

    def _generate_reasoning(
        self,
        complexity: ComplexityLevel,
        current: ModelAggregateResult,
        recommended: ModelAggregateResult,
        cost_reduction: float,
        quality_impact: float,
    ) -> str:
        """Generate human-readable reasoning."""
        if current.model_id == recommended.model_id:
            return f"Current model ({current.model_id}) is optimal for {complexity.value} prompts."

        reason = f"Switch from {current.model_id} to {recommended.model_id}. "
        reason += f"Cost reduction: {cost_reduction:.1f}%. "

        if quality_impact >= 0:
            reason += f"Quality improvement: {abs(quality_impact):.1f}%. "
        else:
            reason += f"Quality trade-off: {abs(quality_impact):.1f}% decrease. "

        reason += f"Based on {recommended.sample_size} samples."
        return reason

    def _segment_risks(
        self,
        complexity: ComplexityLevel,
        quality_impact: float,
    ) -> list[str]:
        """Identify segment-specific risks."""
        risks = []
        if quality_impact < -5:
            risks.append(f"Quality degradation of {abs(quality_impact):.1f}%")
        if complexity == ComplexityLevel.COMPLEX:
            risks.append("Complex prompts may require premium model capabilities")
        return risks

    def _segment_mitigations(self, complexity: ComplexityLevel) -> list[str]:
        """Suggest segment-specific mitigations."""
        mitigations = [
            "A/B test with 10% traffic before full rollout",
            "Monitor user feedback metrics closely",
        ]
        if complexity == ComplexityLevel.COMPLEX:
            mitigations.append("Maintain fallback to premium model for edge cases")
        return mitigations

    def _calc_percent_change(self, old: float, new: float) -> float:
        """Calculate percentage change."""
        if old == 0:
            return 0.0
        return ((old - new) / old) * 100

    def _calculate_costs(
        self,
        segments: list[SegmentRecommendation],
        current_model: str | None,
        model_results: list[ModelAggregateResult],
    ) -> tuple[float, float]:
        """Calculate current and recommended daily costs."""
        current_cost = sum(s.current_cost_monthly / 30 for s in segments)
        recommended_cost = sum(s.recommended_cost_monthly / 30 for s in segments)
        return current_cost, recommended_cost

    def _get_model_quality(
        self,
        model_id: str | None,
        results: list[ModelAggregateResult],
    ) -> ConfidenceInterval:
        """Get quality confidence interval for a model."""
        if not model_id or not results:
            return ConfidenceInterval(value=0.0, lower=0.0, upper=0.0, sample_size=0, confidence_level=0.95)

        for r in results:
            if r.model_id == model_id:
                return ConfidenceInterval(
                    value=r.mean_quality_score,
                    lower=r.quality_ci_lower,
                    upper=r.quality_ci_upper,
                    sample_size=r.sample_size,
                    confidence_level=0.95,
                )
        return ConfidenceInterval(value=0.0, lower=0.0, upper=0.0, sample_size=0, confidence_level=0.95)

    def _get_recommended_quality(
        self,
        segments: list[SegmentRecommendation],
    ) -> ConfidenceInterval:
        """Calculate weighted average quality from segment recommendations."""
        if not segments:
            return ConfidenceInterval(value=0.0, lower=0.0, upper=0.0, sample_size=0, confidence_level=0.95)

        total_volume = sum(s.volume_count for s in segments)
        weighted_quality = sum(
            s.recommended_quality.value * s.volume_count for s in segments
        ) / total_volume if total_volume > 0 else 0

        return ConfidenceInterval(
            value=weighted_quality,
            lower=weighted_quality * 0.95,  # Simplified
            upper=weighted_quality * 1.05,
            sample_size=total_volume,
            confidence_level=0.95,
        )

    def _generate_summary(
        self,
        current_cost: float,
        recommended_cost: float,
        current_quality: ConfidenceInterval,
        recommended_quality: ConfidenceInterval,
        prompt_count: int,
        model_count: int,
    ) -> str:
        """Generate executive summary."""
        cost_savings = (current_cost - recommended_cost) / current_cost * 100 if current_cost > 0 else 0
        quality_impact = (
            (recommended_quality.value - current_quality.value) / current_quality.value * 100
            if current_quality.value > 0 else 0
        )

        return (
            f"Analyzed {prompt_count:,} prompts across {model_count} models. "
            f"Recommended configuration saves {cost_savings:.1f}% on costs "
            f"with {abs(quality_impact):.1f}% quality {'improvement' if quality_impact >= 0 else 'trade-off'}. "
            f"Estimated monthly savings: ${(current_cost - recommended_cost) * 30:,.2f}."
        )

    def _generate_comparisons(
        self,
        results: list[ModelAggregateResult],
        current_model: str | None,
    ) -> list[ModelComparison]:
        """Generate pairwise model comparisons."""
        # Compare each model to current model
        comparisons = []
        if not current_model:
            return comparisons

        current = next((r for r in results if r.model_id == current_model), None)
        if not current:
            return comparisons

        for result in results:
            if result.model_id == current_model:
                continue

            cost_diff = (current.mean_cost_per_request - result.mean_cost_per_request)
            cost_diff_pct = cost_diff / current.mean_cost_per_request * 100 if current.mean_cost_per_request > 0 else 0

            quality_diff_pct = (
                (result.mean_quality_score - current.mean_quality_score)
                / current.mean_quality_score * 100
                if current.mean_quality_score > 0 else 0
            )

            # Cost per quality point
            cpq = abs(cost_diff_pct / quality_diff_pct) if quality_diff_pct != 0 else float('inf')

            comparisons.append(ModelComparison(
                model_a=current_model,
                model_b=result.model_id,
                quality_a=ConfidenceInterval(
                    value=current.mean_quality_score,
                    lower=current.quality_ci_lower,
                    upper=current.quality_ci_upper,
                    sample_size=current.sample_size,
                    confidence_level=0.95,
                ),
                quality_b=ConfidenceInterval(
                    value=result.mean_quality_score,
                    lower=result.quality_ci_lower,
                    upper=result.quality_ci_upper,
                    sample_size=result.sample_size,
                    confidence_level=0.95,
                ),
                quality_difference=ConfidenceInterval(
                    value=result.mean_quality_score - current.mean_quality_score,
                    lower=-0.1,  # Simplified
                    upper=0.1,
                    sample_size=min(current.sample_size, result.sample_size),
                    confidence_level=0.95,
                ),
                quality_difference_percent=quality_diff_pct,
                quality_is_significant=abs(quality_diff_pct) > 5,
                quality_p_value=0.05,  # Placeholder
                cost_a_monthly=current.mean_cost_per_request * 30 * 1000,  # Assuming 1000 requests/day
                cost_b_monthly=result.mean_cost_per_request * 30 * 1000,
                cost_difference=cost_diff * 30 * 1000,
                cost_difference_percent=cost_diff_pct,
                cost_per_quality_point=cpq,
                verdict=self._determine_verdict(cost_diff_pct, quality_diff_pct),
                explanation=f"{result.model_id} is {cost_diff_pct:.1f}% cheaper with {quality_diff_pct:.1f}% quality impact",
            ))

        return comparisons

    def _assess_complexity(self, segments: list[SegmentRecommendation]) -> str:
        """Assess implementation complexity."""
        model_changes = len(set(s.recommended_model for s in segments))
        if model_changes <= 1:
            return "low"
        elif model_changes <= 2:
            return "medium"
        else:
            return "high"

    def _generate_steps(self, segments: list[SegmentRecommendation]) -> list[str]:
        """Generate implementation steps."""
        return [
            "1. Deploy complexity classifier for routing",
            "2. Configure Portkey routing rules",
            "3. A/B test with 10% traffic",
            "4. Monitor quality metrics for 1 week",
            "5. Expand to 50% traffic if metrics stable",
            "6. Full rollout with monitoring",
        ]

    def _identify_risks(self, segments: list[SegmentRecommendation]) -> list[str]:
        """Identify overall risks."""
        risks = []
        for seg in segments:
            if seg.quality_impact_percent < -10:
                risks.append(f"Significant quality risk in {seg.segment_name} segment")
        if not risks:
            risks.append("Low overall risk - quality impact within acceptable bounds")
        return risks

    def _suggest_mitigations(self) -> list[str]:
        """Suggest overall mitigations."""
        return [
            "Implement gradual rollout with monitoring",
            "Maintain fallback to current model",
            "Set up alerting for quality degradation",
            "Review monthly for model updates",
        ]

    def _generate_routing_config(
        self,
        segments: list[SegmentRecommendation],
    ) -> dict[str, Any]:
        """Generate Portkey routing configuration."""
        conditions = []
        for seg in segments:
            conditions.append({
                "query": f"$.metadata.complexity == '{seg.segment_name.lower()}'",
                "then": seg.recommended_model,
            })

        # Default to first segment's model
        default_model = segments[0].recommended_model if segments else "gpt-4o"

        return {
            "strategy": {
                "mode": "conditional",
                "conditions": conditions,
                "default": default_model,
            },
            "retry": {
                "attempts": 3,
                "on_status_codes": [429, 500, 502, 503],
            },
            "cache": {
                "mode": "semantic",
                "max_age": 3600,
            },
        }
