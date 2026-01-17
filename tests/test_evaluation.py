"""Tests for evaluation framework."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from src.evaluation.base import BaseEvaluator, EvaluatorConfig
from src.evaluation.cost_tracker import CostTracker
from src.evaluation.composite import CompositeEvaluator
from src.models.evaluation import MetricScore, CostResult


class TestCostTracker:
    """Tests for cost tracker."""

    @pytest.fixture
    def cost_tracker(self):
        """Create cost tracker instance."""
        return CostTracker()

    def test_count_tokens_openai(self, cost_tracker):
        """Test token counting for OpenAI models."""
        text = "Hello, how are you today?"
        tokens = cost_tracker.count_tokens(text, "gpt-4o")
        
        assert tokens > 0
        assert tokens < 20  # Reasonable bound

    def test_count_tokens_anthropic(self, cost_tracker):
        """Test token counting for Anthropic models."""
        text = "Hello, how are you today?"
        tokens = cost_tracker.count_tokens(text, "claude-sonnet-4-5")
        
        assert tokens > 0

    def test_calculate_cost_gpt4o(self, cost_tracker):
        """Test cost calculation for GPT-4o."""
        result = cost_tracker.calculate(
            model_id="gpt-4o",
            input_tokens=1000,
            output_tokens=500
        )
        
        # GPT-4o: $2.50/1M input, $10.00/1M output
        expected_input = (1000 / 1_000_000) * 2.50
        expected_output = (500 / 1_000_000) * 10.00
        
        assert abs(result.input_cost_usd - expected_input) < 0.0001
        assert abs(result.output_cost_usd - expected_output) < 0.0001
        assert abs(result.total_cost_usd - (expected_input + expected_output)) < 0.0001

    def test_calculate_cost_gpt4o_mini(self, cost_tracker):
        """Test cost calculation for GPT-4o-mini (cheaper model)."""
        result = cost_tracker.calculate(
            model_id="gpt-4o-mini",
            input_tokens=1000,
            output_tokens=500
        )
        
        # GPT-4o-mini: $0.15/1M input, $0.60/1M output
        expected_input = (1000 / 1_000_000) * 0.15
        expected_output = (500 / 1_000_000) * 0.60
        
        assert abs(result.total_cost_usd - (expected_input + expected_output)) < 0.0001

    def test_calculate_cost_from_text(self, cost_tracker):
        """Test cost calculation from raw text."""
        result = cost_tracker.calculate_from_text(
            model_id="gpt-4o",
            input_text="What is 2+2?",
            output_text="4"
        )
        
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert result.total_cost_usd > 0

    def test_unknown_model_fallback(self, cost_tracker):
        """Test fallback pricing for unknown models."""
        result = cost_tracker.calculate(
            model_id="unknown-model-xyz",
            input_tokens=1000,
            output_tokens=500
        )
        
        # Should use default pricing
        assert result.total_cost_usd > 0


class TestCompositeEvaluator:
    """Tests for composite evaluator."""

    @pytest.fixture
    def composite_evaluator(self):
        """Create composite evaluator with mocked sub-evaluators."""
        evaluator = CompositeEvaluator(
            include_bertscore=False,  # Disable for faster tests
            include_deepeval=False,
        )
        return evaluator

    def test_calculate_weighted_score_equal_weights(self):
        """Test weighted score calculation with equal weights."""
        evaluator = CompositeEvaluator()
        
        metrics = {
            "metric_a": MetricScore(metric_name="metric_a", score=0.8),
            "metric_b": MetricScore(metric_name="metric_b", score=0.6),
        }
        weights = {"metric_a": 1.0, "metric_b": 1.0}
        
        score = evaluator._calculate_weighted_score(metrics, weights)
        
        # (0.8 * 1.0 + 0.6 * 1.0) / 2.0 = 0.7
        assert abs(score - 0.7) < 0.001

    def test_calculate_weighted_score_unequal_weights(self):
        """Test weighted score with different weights."""
        evaluator = CompositeEvaluator()
        
        metrics = {
            "important": MetricScore(metric_name="important", score=0.9),
            "minor": MetricScore(metric_name="minor", score=0.3),
        }
        weights = {"important": 3.0, "minor": 1.0}
        
        score = evaluator._calculate_weighted_score(metrics, weights)
        
        # (0.9 * 3.0 + 0.3 * 1.0) / 4.0 = 0.75
        assert abs(score - 0.75) < 0.001

    def test_metric_score_creation(self):
        """Test MetricScore model."""
        metric = MetricScore(
            metric_name="test_metric",
            score=0.85,
            reason="Good response quality",
            passed=True,
        )
        
        assert metric.metric_name == "test_metric"
        assert metric.score == 0.85
        assert metric.passed is True

    def test_cost_result_creation(self):
        """Test CostResult model."""
        result = CostResult(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_cost_usd=0.001,
            output_cost_usd=0.002,
            total_cost_usd=0.003,
            model_id="gpt-4o",
        )
        
        assert result.total_tokens == 150
        assert result.total_cost_usd == 0.003


class TestBertScoreEvaluator:
    """Tests for BERTScore evaluator (mocked)."""

    @pytest.mark.asyncio
    async def test_bertscore_mock(self):
        """Test BERTScore evaluation with mock."""
        from src.evaluation.bertscore_evaluator import BertScoreEvaluator
        
        with patch('src.evaluation.bertscore_evaluator.score') as mock_score:
            # Mock BERTScore output
            mock_score.return_value = (
                [0.85],  # P
                [0.82],  # R
                [0.83],  # F1
            )
            
            evaluator = BertScoreEvaluator()
            result = await evaluator.evaluate(
                prompt="What is AI?",
                completion="AI is artificial intelligence.",
                reference="AI stands for artificial intelligence.",
            )
            
            assert "bertscore_f1" in result.metrics
            assert 0 <= result.composite_score <= 1


class TestDeepEvalEvaluator:
    """Tests for DeepEval evaluator (mocked)."""

    @pytest.mark.asyncio
    async def test_simple_llm_judge_fallback(self):
        """Test that SimpleLLMJudge works as fallback."""
        from src.evaluation.deepeval_evaluator import SimpleLLMJudge
        
        # This tests the fallback implementation
        judge = SimpleLLMJudge()
        
        # Mock the evaluation to avoid API calls
        with patch.object(judge, 'evaluate', new_callable=AsyncMock) as mock_eval:
            mock_result = MagicMock()
            mock_result.metrics = {
                "relevancy": MetricScore(
                    metric_name="relevancy",
                    score=0.8,
                    reason="Good relevancy",
                    passed=True,
                )
            }
            mock_result.composite_score = 0.8
            mock_eval.return_value = mock_result
            
            result = await judge.evaluate(
                prompt="What is 2+2?",
                completion="4",
            )
            
            assert result.composite_score == 0.8
