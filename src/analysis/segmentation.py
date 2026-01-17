"""Segmentation engine for grouping prompts by characteristics."""

from collections import defaultdict
from typing import Callable

import structlog

from src.models.canonical import CanonicalPrompt, ComplexityLevel, TaskType

logger = structlog.get_logger()


class SegmentationEngine:
    """
    Groups prompts into segments for analysis.
    
    Segmentation strategies:
    - By complexity (simple, medium, complex)
    - By task type (summarization, QA, code, etc.)
    - By custom criteria (token count, domain, etc.)
    """

    def segment_by_complexity(
        self,
        prompts: list[CanonicalPrompt],
    ) -> dict[ComplexityLevel, list[CanonicalPrompt]]:
        """
        Segment prompts by complexity level.
        
        Uses metadata if available, otherwise estimates from content.
        """
        segments = defaultdict(list)

        for prompt in prompts:
            complexity = prompt.metadata.complexity
            segments[complexity].append(prompt)

        # Log segment distribution
        logger.info(
            "Segmented by complexity",
            simple=len(segments[ComplexityLevel.SIMPLE]),
            medium=len(segments[ComplexityLevel.MEDIUM]),
            complex=len(segments[ComplexityLevel.COMPLEX]),
        )

        return dict(segments)

    def segment_by_task_type(
        self,
        prompts: list[CanonicalPrompt],
    ) -> dict[TaskType, list[CanonicalPrompt]]:
        """
        Segment prompts by task type.
        
        Uses metadata if available.
        """
        segments = defaultdict(list)

        for prompt in prompts:
            task_type = prompt.metadata.task_type
            segments[task_type].append(prompt)

        # Log segment distribution
        for task_type, segment_prompts in segments.items():
            logger.debug(
                "Task type segment",
                task_type=task_type.value,
                count=len(segment_prompts),
            )

        return dict(segments)

    def segment_by_token_count(
        self,
        prompts: list[CanonicalPrompt],
        thresholds: list[int] | None = None,
    ) -> dict[str, list[CanonicalPrompt]]:
        """
        Segment prompts by input token count.
        
        Args:
            prompts: List of prompts
            thresholds: Token count thresholds, default [500, 2000]
            
        Returns:
            Dict with segment names as keys
        """
        thresholds = thresholds or [500, 2000]
        segments = defaultdict(list)

        for prompt in prompts:
            token_count = prompt.completion.input_tokens

            if token_count <= thresholds[0]:
                segment = f"short (<={thresholds[0]} tokens)"
            elif len(thresholds) > 1 and token_count <= thresholds[1]:
                segment = f"medium ({thresholds[0]}-{thresholds[1]} tokens)"
            else:
                segment = f"long (>{thresholds[-1]} tokens)"

            segments[segment].append(prompt)

        return dict(segments)

    def segment_by_custom(
        self,
        prompts: list[CanonicalPrompt],
        key_func: Callable[[CanonicalPrompt], str],
    ) -> dict[str, list[CanonicalPrompt]]:
        """
        Segment prompts by custom key function.
        
        Args:
            prompts: List of prompts
            key_func: Function that returns segment key for a prompt
            
        Returns:
            Dict with segment names as keys
        """
        segments = defaultdict(list)

        for prompt in prompts:
            try:
                key = key_func(prompt)
                segments[key].append(prompt)
            except Exception:
                segments["other"].append(prompt)

        return dict(segments)

    def get_segment_stats(
        self,
        segments: dict[str, list[CanonicalPrompt]],
    ) -> dict[str, dict]:
        """
        Calculate statistics for each segment.
        
        Returns volume, average tokens, average latency, etc.
        """
        total = sum(len(s) for s in segments.values())
        stats = {}

        for name, prompts in segments.items():
            if not prompts:
                continue

            input_tokens = [p.completion.input_tokens for p in prompts]
            output_tokens = [p.completion.output_tokens for p in prompts]
            latencies = [p.completion.latency_ms for p in prompts]
            costs = [p.completion.cost_usd for p in prompts]

            stats[name] = {
                "count": len(prompts),
                "volume_percent": len(prompts) / total * 100 if total > 0 else 0,
                "avg_input_tokens": sum(input_tokens) / len(input_tokens),
                "avg_output_tokens": sum(output_tokens) / len(output_tokens),
                "avg_latency_ms": sum(latencies) / len(latencies),
                "total_cost": sum(costs),
                "avg_cost": sum(costs) / len(costs),
            }

        return stats

    def assign_complexity_scores(
        self,
        prompts: list[CanonicalPrompt],
    ) -> list[tuple[CanonicalPrompt, float]]:
        """
        Assign numerical complexity scores for ML-based routing.
        
        Returns list of (prompt, score) tuples.
        Score range: 0.0 (simplest) to 1.0 (most complex)
        """
        scored = []

        for prompt in prompts:
            score = self._calculate_complexity_score(prompt)
            scored.append((prompt, score))

        return scored

    def _calculate_complexity_score(self, prompt: CanonicalPrompt) -> float:
        """
        Calculate numerical complexity score for a prompt.
        
        Heuristics based on:
        - Token count
        - Number of messages
        - Presence of constraints/requirements
        - Task type complexity
        """
        score = 0.0

        # Token count contribution (0.0 - 0.3)
        input_tokens = prompt.completion.input_tokens
        if input_tokens < 200:
            score += 0.05
        elif input_tokens < 500:
            score += 0.10
        elif input_tokens < 1000:
            score += 0.15
        elif input_tokens < 2000:
            score += 0.20
        else:
            score += 0.30

        # Message count contribution (0.0 - 0.2)
        msg_count = len(prompt.messages)
        if msg_count <= 2:
            score += 0.05
        elif msg_count <= 4:
            score += 0.10
        elif msg_count <= 6:
            score += 0.15
        else:
            score += 0.20

        # Task type contribution (0.0 - 0.3)
        task_scores = {
            TaskType.CLASSIFICATION: 0.05,
            TaskType.EXTRACTION: 0.10,
            TaskType.SUMMARIZATION: 0.10,
            TaskType.QA: 0.15,
            TaskType.TRANSLATION: 0.15,
            TaskType.CODE_GENERATION: 0.25,
            TaskType.REASONING: 0.30,
            TaskType.CREATIVE_WRITING: 0.20,
            TaskType.OTHER: 0.15,
        }
        score += task_scores.get(prompt.metadata.task_type, 0.15)

        # Explicit complexity from metadata (0.0 - 0.2)
        complexity_scores = {
            ComplexityLevel.SIMPLE: 0.05,
            ComplexityLevel.MEDIUM: 0.10,
            ComplexityLevel.COMPLEX: 0.20,
        }
        score += complexity_scores.get(prompt.metadata.complexity, 0.10)

        # Normalize to 0-1 range
        return min(1.0, max(0.0, score))
