"""Integration tests for the full pipeline."""

import pytest
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.adapters.csv_adapter import CSVAdapter
from src.replay.engine import ReplayEngine
from src.replay.model_registry import ModelRegistry
from src.evaluation.composite import CompositeEvaluator
from src.evaluation.cost_tracker import CostTracker
from src.analysis.statistics import StatisticalAnalyzer
from src.analysis.segmentation import SegmentationEngine
from src.analysis.pareto import ParetoAnalyzer
from src.analysis.recommendation import RecommendationEngine
from src.models.evaluation import EvaluationResult, MetricScore, CostResult


class TestEndToEndPipeline:
    """End-to-end integration tests."""

    @pytest.fixture
    def sample_evaluations(self, sample_prompts_batch):
        """Create sample evaluation results."""
        evaluations = []
        models = ["gpt-4o", "gpt-4o-mini"]
        
        for i, prompt in enumerate(sample_prompts_batch):
            for model_id in models:
                # Simulate quality difference (premium model slightly better)
                quality = 0.85 if model_id == "gpt-4o" else 0.78
                cost = 0.002 if model_id == "gpt-4o" else 0.0003
                
                evaluations.append(EvaluationResult(
                    prompt_id=prompt.id,
                    model_id=model_id,
                    metrics={
                        "semantic_similarity": MetricScore(
                            metric_name="semantic_similarity",
                            score=quality + (i % 3) * 0.02,
                            passed=True,
                        ),
                    },
                    composite_score=quality + (i % 3) * 0.02,
                    cost=CostResult(
                        input_tokens=50,
                        output_tokens=100,
                        total_tokens=150,
                        input_cost_usd=cost * 0.3,
                        output_cost_usd=cost * 0.7,
                        total_cost_usd=cost,
                        model_id=model_id,
                    ),
                    latency_ms=500 + i * 50,
                ))
        
        return evaluations

    @pytest.mark.asyncio
    async def test_data_ingestion(self, sample_json_data):
        """Test data ingestion from JSON file."""
        adapter = CSVAdapter(sample_json_data)
        prompts = await adapter.fetch_prompts()
        
        assert len(prompts) == 2
        assert prompts[0].id == "json-001"
        assert prompts[0].messages is not None

    def test_statistical_analysis(self, sample_evaluations):
        """Test statistical analysis on evaluation results."""
        analyzer = StatisticalAnalyzer()
        
        # Extract quality scores for GPT-4o
        gpt4_scores = [
            e.composite_score for e in sample_evaluations
            if e.model_id == "gpt-4o"
        ]
        
        ci = analyzer.calculate_confidence_interval(gpt4_scores)
        
        assert ci.sample_size == len(gpt4_scores)
        assert ci.lower <= ci.value <= ci.upper
        assert ci.confidence_level == 0.95

    def test_segmentation(self, sample_prompts_batch):
        """Test prompt segmentation."""
        engine = SegmentationEngine()
        segments = engine.segment_by_complexity(sample_prompts_batch)
        
        # Should have at least 2 different complexity levels
        assert len(segments) >= 2

    def test_pareto_analysis(self, sample_evaluations):
        """Test Pareto frontier calculation."""
        # Aggregate by model
        model_results = {}
        for e in sample_evaluations:
            if e.model_id not in model_results:
                model_results[e.model_id] = {"scores": [], "costs": []}
            model_results[e.model_id]["scores"].append(e.composite_score)
            model_results[e.model_id]["costs"].append(e.cost.total_cost_usd)
        
        # Create mock model aggregate results
        from src.models.evaluation import ModelAggregateResult
        import numpy as np
        
        aggregates = []
        for model_id, data in model_results.items():
            aggregates.append(ModelAggregateResult(
                model_id=model_id,
                provider="test",
                sample_size=len(data["scores"]),
                successful_replays=len(data["scores"]),
                failed_replays=0,
                refusal_count=0,
                refusal_rate=0.0,
                mean_quality_score=np.mean(data["scores"]),
                quality_std=np.std(data["scores"]),
                quality_ci_lower=np.mean(data["scores"]) - 0.05,
                quality_ci_upper=np.mean(data["scores"]) + 0.05,
                total_cost_usd=sum(data["costs"]),
                mean_cost_per_request=np.mean(data["costs"]),
                mean_input_tokens=50,
                mean_output_tokens=100,
                mean_latency_ms=500,
                p50_latency_ms=450,
                p95_latency_ms=800,
                p99_latency_ms=1000,
            ))
        
        analyzer = ParetoAnalyzer()
        pareto = analyzer.find_pareto_frontier(aggregates)
        
        # At least one model should be on the frontier
        assert len(pareto) >= 1

    def test_recommendation_generation(self, sample_prompts_batch, sample_evaluations):
        """Test recommendation engine."""
        engine = RecommendationEngine()
        
        recommendation = engine.generate_recommendation(
            prompts=sample_prompts_batch,
            evaluations=sample_evaluations,
            current_model="gpt-4o",
        )
        
        assert recommendation is not None
        assert recommendation.total_prompts_analyzed > 0
        assert len(recommendation.models_evaluated) > 0
        assert recommendation.executive_summary is not None

    def test_cost_tracker(self):
        """Test cost tracking accuracy."""
        tracker = CostTracker()
        
        # Test with known token counts
        result = tracker.calculate(
            model_id="gpt-4o",
            input_tokens=1000,
            output_tokens=500
        )
        
        # GPT-4o: $2.50/1M input, $10.00/1M output
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        
        assert abs(result.total_cost_usd - expected) < 0.0001


class TestAPIIntegration:
    """Tests for API endpoints."""

    @pytest.fixture
    def test_client(self):
        """Create test client for FastAPI."""
        from fastapi.testclient import TestClient
        from src.main import app
        
        return TestClient(app)

    def test_health_check(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_root_endpoint(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "Track 4" in data.get("title", "")


class TestPortkeyIntegration:
    """Integration tests with Portkey (requires API key)."""

    @pytest.fixture
    def portkey_api_key(self):
        """Get Portkey API key from environment."""
        key = os.getenv("PORTKEY_API_KEY")
        if not key:
            pytest.skip("PORTKEY_API_KEY not set")
        return key

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_live_replay(self, portkey_api_key, sample_prompt):
        """Test live replay through Portkey (requires API key)."""
        engine = ReplayEngine(portkey_api_key=portkey_api_key)
        
        model = ModelRegistry.get_model("gpt-4o-mini")
        result = await engine.replay_single(sample_prompt, model)
        
        assert result.success is True
        assert result.completion != ""
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        
        await engine.close()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multiple_models(self, portkey_api_key, sample_prompt):
        """Test replay across multiple models."""
        engine = ReplayEngine(portkey_api_key=portkey_api_key)
        
        models_to_test = ["gpt-4o-mini", "claude-haiku-4-5-20251001"]
        results = []
        
        for model_id in models_to_test:
            model = ModelRegistry.get_model(model_id)
            if model:
                result = await engine.replay_single(sample_prompt, model)
                results.append(result)
        
        await engine.close()
        
        # All models should return results
        assert len(results) == len(models_to_test)
        assert all(r.success for r in results)


# Custom pytest marker for integration tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires API keys)"
    )
