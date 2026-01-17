"""Base evaluator interface."""

from abc import ABC, abstractmethod
from typing import Any

from src.models.canonical import CanonicalPrompt
from src.models.evaluation import MetricScore, ReplayResult


class BaseEvaluator(ABC):
    """
    Abstract base class for quality evaluators.
    
    All evaluators implement a consistent interface for scoring
    replay results against quality metrics.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the evaluator name."""
        ...

    @property
    @abstractmethod
    def metrics(self) -> list[str]:
        """Return list of metric names this evaluator provides."""
        ...

    @abstractmethod
    async def evaluate(
        self,
        prompt: CanonicalPrompt,
        replay: ReplayResult,
        reference: str | None = None,
    ) -> dict[str, MetricScore]:
        """
        Evaluate a replay result.
        
        Args:
            prompt: The original prompt
            replay: The replay result to evaluate
            reference: Optional reference output for comparison
            
        Returns:
            Dictionary of metric name to MetricScore
        """
        ...

    async def evaluate_batch(
        self,
        prompts: list[CanonicalPrompt],
        replays: list[ReplayResult],
        references: list[str | None] | None = None,
    ) -> list[dict[str, MetricScore]]:
        """
        Evaluate a batch of replay results.
        
        Default implementation calls evaluate() for each.
        Subclasses can override for batch optimization.
        """
        refs = references or [None] * len(prompts)
        results = []
        for prompt, replay, ref in zip(prompts, replays, refs):
            result = await self.evaluate(prompt, replay, ref)
            results.append(result)
        return results

    def requires_reference(self) -> bool:
        """Check if this evaluator requires a reference output."""
        return False

    def get_config(self) -> dict[str, Any]:
        """Get evaluator configuration for reproducibility."""
        return {"name": self.name, "metrics": self.metrics}
