"""Composite evaluator combining multiple evaluation methods."""

from datetime import datetime

import structlog

from src.config import get_settings
from src.evaluation.base import BaseEvaluator
from src.evaluation.cost_tracker import CostTracker
from src.evaluation.bertscore_evaluator import BertScoreEvaluator
from src.evaluation.deepeval_evaluator import DeepEvalEvaluator, SimpleLLMJudge
from src.models.canonical import CanonicalPrompt
from src.models.evaluation import EvaluationResult, MetricScore, ReplayResult

logger = structlog.get_logger()


class CompositeEvaluator:
    """
    Combines multiple evaluators into a single evaluation pipeline.
    
    Aggregates scores from:
    - BERTScore (semantic similarity, deterministic)
    - DeepEval (LLM-as-judge, self-explaining)
    - Cost tracker (economic metrics)
    
    Calculates composite score with configurable weights.
    """

    DEFAULT_WEIGHTS = {
        "bertscore_f1": 0.3,
        "deepeval_answer_relevancy": 0.4,
        "llm_judge_score": 0.3,
    }

    def __init__(
        self,
        enable_bertscore: bool = True,
        enable_deepeval: bool = True,
        use_simple_judge: bool = False,
        weights: dict[str, float] | None = None,
    ):
        settings = get_settings()

        self.evaluators: list[BaseEvaluator] = []
        self.cost_tracker = CostTracker()
        self.weights = weights or self.DEFAULT_WEIGHTS

        if enable_bertscore and settings.enable_bertscore:
            try:
                self.evaluators.append(BertScoreEvaluator())
                logger.info("BERTScore evaluator enabled")
            except Exception as e:
                logger.warning("Could not initialize BERTScore", error=str(e))

        if enable_deepeval and settings.enable_deepeval:
            try:
                if use_simple_judge:
                    self.evaluators.append(SimpleLLMJudge())
                    logger.info("Simple LLM judge enabled")
                else:
                    self.evaluators.append(DeepEvalEvaluator())
                    logger.info("DeepEval evaluator enabled")
            except Exception as e:
                logger.warning("Could not initialize DeepEval", error=str(e))
                # Fall back to simple judge
                try:
                    self.evaluators.append(SimpleLLMJudge())
                    logger.info("Falling back to simple LLM judge")
                except Exception as e2:
                    logger.warning("Could not initialize simple judge", error=str(e2))

    async def evaluate(
        self,
        prompt: CanonicalPrompt,
        replay: ReplayResult,
        reference: str | None = None,
    ) -> EvaluationResult:
        """
        Run all evaluators and aggregate results.
        
        Args:
            prompt: Original prompt
            replay: Replay result to evaluate
            reference: Optional reference output
            
        Returns:
            EvaluationResult with all metrics and composite score
        """
        all_metrics: dict[str, MetricScore] = {}

        # Run each evaluator
        for evaluator in self.evaluators:
            try:
                metrics = await evaluator.evaluate(prompt, replay, reference)
                all_metrics.update(metrics)
            except Exception as e:
                logger.warning(
                    f"Evaluator {evaluator.name} failed",
                    error=str(e),
                    prompt_id=prompt.id,
                )

        # Calculate cost
        cost = self.cost_tracker.calculate_cost(
            replay.model_id,
            replay.input_tokens,
            replay.output_tokens,
        )

        # Calculate composite score
        composite = self._calculate_composite_score(all_metrics)

        return EvaluationResult(
            prompt_id=prompt.id,
            model_id=replay.model_id,
            metrics=all_metrics,
            composite_score=composite,
            cost=cost,
            latency_ms=replay.latency_ms,
            refused=replay.refused,
            refusal_reason="Model refused to respond" if replay.refused else None,
            evaluated_at=datetime.utcnow(),
        )

    async def evaluate_batch(
        self,
        prompts: list[CanonicalPrompt],
        replays: list[ReplayResult],
        references: list[str | None] | None = None,
        max_concurrent: int = 10,
    ) -> list[EvaluationResult]:
        """
        Evaluate a batch of replays with parallel execution.
        
        Uses asyncio.gather for concurrent evaluation with semaphore
        to limit concurrency and respect rate limits.
        
        Args:
            prompts: List of prompts
            replays: List of replay results
            references: Optional reference outputs
            max_concurrent: Maximum concurrent evaluations (default: 10)
        """
        import asyncio
        
        refs = references or [None] * len(prompts)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def eval_with_limit(prompt, replay, ref) -> EvaluationResult:
            """Evaluate with concurrency limit."""
            async with semaphore:
                try:
                    return await self.evaluate(prompt, replay, ref)
                except Exception as e:
                    logger.error(
                        "Batch evaluation failed for prompt",
                        prompt_id=prompt.id,
                        error=str(e),
                    )
                    # Return failed evaluation
                    return EvaluationResult(
                        prompt_id=prompt.id,
                        model_id=replay.model_id,
                        metrics={},
                        composite_score=0.0,
                        cost=self.cost_tracker.calculate_cost(
                            replay.model_id,
                            replay.input_tokens,
                            replay.output_tokens,
                        ),
                        latency_ms=replay.latency_ms,
                        refused=True,
                        refusal_reason=f"Evaluation error: {str(e)}",
                    )
        
        # Create tasks for parallel execution
        tasks = [
            eval_with_limit(prompt, replay, ref)
            for prompt, replay, ref in zip(prompts, replays, refs)
        ]
        
        # Execute all evaluations concurrently
        results = await asyncio.gather(*tasks)
        
        return list(results)

    def _calculate_composite_score(
        self,
        metrics: dict[str, MetricScore],
    ) -> float:
        """
        Calculate weighted composite score from individual metrics.
        
        Uses configured weights, normalizing to handle missing metrics.
        """
        if not metrics:
            return 0.0

        weighted_sum = 0.0
        weight_sum = 0.0

        for metric_name, weight in self.weights.items():
            if metric_name in metrics:
                weighted_sum += metrics[metric_name].score * weight
                weight_sum += weight

        if weight_sum == 0:
            # Fall back to average of all available metrics
            scores = [m.score for m in metrics.values()]
            return sum(scores) / len(scores) if scores else 0.0

        return weighted_sum / weight_sum

    def get_available_metrics(self) -> list[str]:
        """Get list of all available metric names."""
        metrics = []
        for evaluator in self.evaluators:
            metrics.extend(evaluator.metrics)
        return metrics

    def get_config(self) -> dict:
        """Get evaluator configuration for reproducibility."""
        return {
            "evaluators": [e.get_config() for e in self.evaluators],
            "weights": self.weights,
        }
