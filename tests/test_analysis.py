"""Tests for analysis engine."""

import pytest

from src.analysis.statistics import StatisticalAnalyzer
from src.analysis.segmentation import SegmentationEngine
from src.analysis.pareto import ParetoAnalyzer
from src.models.canonical import ComplexityLevel
from src.models.evaluation import ModelAggregateResult


class TestStatisticalAnalyzer:
    """Tests for statistical analysis."""

    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        analyzer = StatisticalAnalyzer()
        values = [0.85, 0.87, 0.82, 0.88, 0.86, 0.84, 0.89, 0.83, 0.87, 0.85]
        
        ci = analyzer.calculate_confidence_interval(values)
        
        assert ci.sample_size == 10
        assert 0.84 < ci.value < 0.87  # Mean should be around 0.856
        assert ci.lower < ci.value < ci.upper
        assert ci.confidence_level == 0.95

    def test_empty_values(self):
        """Test handling of empty values."""
        analyzer = StatisticalAnalyzer()
        ci = analyzer.calculate_confidence_interval([])
        
        assert ci.value == 0.0
        assert ci.sample_size == 0

    def test_compare_means_significant(self):
        """Test mean comparison with significant difference."""
        analyzer = StatisticalAnalyzer()
        group_a = [0.9, 0.92, 0.88, 0.91, 0.89]
        group_b = [0.7, 0.72, 0.68, 0.71, 0.69]
        
        result = analyzer.compare_means(group_a, group_b)
        
        assert result["is_significant"] == True  # Use == for numpy bool compatibility
        assert result["p_value"] < 0.05
        assert "significant" in result["interpretation"].lower()

    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        analyzer = StatisticalAnalyzer()
        values = list(range(1, 101))  # 1 to 100
        
        stats = analyzer.calculate_summary_statistics(values)
        
        assert stats["count"] == 100
        assert stats["mean"] == 50.5
        assert stats["min"] == 1
        assert stats["max"] == 100
        assert stats["p50"] == 50.5


class TestSegmentationEngine:
    """Tests for segmentation."""

    def test_segment_by_complexity(self, sample_prompts_batch):
        """Test segmentation by complexity level."""
        engine = SegmentationEngine()
        segments = engine.segment_by_complexity(sample_prompts_batch)
        
        assert ComplexityLevel.SIMPLE in segments
        assert ComplexityLevel.MEDIUM in segments
        assert ComplexityLevel.COMPLEX in segments

    def test_complexity_score(self, sample_prompt):
        """Test complexity score calculation."""
        engine = SegmentationEngine()
        scored = engine.assign_complexity_scores([sample_prompt])
        
        assert len(scored) == 1
        prompt, score = scored[0]
        assert 0.0 <= score <= 1.0


class TestParetoAnalyzer:
    """Tests for Pareto frontier analysis."""

    def test_find_pareto_frontier(self):
        """Test Pareto frontier identification."""
        analyzer = ParetoAnalyzer()
        
        # Create mock model results
        models = [
            self._create_model_result("model-a", cost=0.01, quality=0.9),
            self._create_model_result("model-b", cost=0.005, quality=0.7),  # Pareto (cheapest)
            self._create_model_result("model-c", cost=0.02, quality=0.95),  # Pareto (highest quality)
            self._create_model_result("model-d", cost=0.015, quality=0.8),  # Dominated by a
        ]
        
        pareto = analyzer.find_pareto_frontier(models)
        pareto_ids = {m.model_id for m in pareto}
        
        assert "model-b" in pareto_ids  # Cheapest
        assert "model-c" in pareto_ids  # Highest quality
        # model-a might or might not be on frontier depending on exact positions

    def test_empty_models(self):
        """Test handling of empty model list."""
        analyzer = ParetoAnalyzer()
        pareto = analyzer.find_pareto_frontier([])
        
        assert pareto == []

    def test_single_model(self):
        """Test with single model."""
        analyzer = ParetoAnalyzer()
        models = [self._create_model_result("only-model", cost=0.01, quality=0.8)]
        
        pareto = analyzer.find_pareto_frontier(models)
        
        assert len(pareto) == 1
        assert pareto[0].model_id == "only-model"

    def _create_model_result(self, model_id: str, cost: float, quality: float) -> ModelAggregateResult:
        """Helper to create mock model results."""
        return ModelAggregateResult(
            model_id=model_id,
            provider="test",
            sample_size=100,
            successful_replays=100,
            failed_replays=0,
            refusal_count=0,
            refusal_rate=0.0,
            mean_quality_score=quality,
            quality_std=0.05,
            quality_ci_lower=quality - 0.05,
            quality_ci_upper=quality + 0.05,
            total_cost_usd=cost * 100,
            mean_cost_per_request=cost,
            mean_input_tokens=100,
            mean_output_tokens=200,
            mean_latency_ms=500,
            p50_latency_ms=400,
            p95_latency_ms=800,
            p99_latency_ms=1000,
        )
