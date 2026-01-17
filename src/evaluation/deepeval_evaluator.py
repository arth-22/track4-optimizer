"""DeepEval integration for LLM-as-judge evaluation."""

import structlog

from src.config import get_settings
from src.evaluation.base import BaseEvaluator
from src.models.canonical import CanonicalPrompt
from src.models.evaluation import MetricScore, ReplayResult

logger = structlog.get_logger()


class DeepEvalEvaluator(BaseEvaluator):
    """
    Evaluator using DeepEval for LLM-as-judge evaluation.
    
    DeepEval provides self-explaining metrics that return both
    scores and reasoning, making debugging easier.
    
    Metrics available:
    - Answer Relevancy: Is the answer relevant to the question?
    - Faithfulness: Is the answer faithful to the context?
    - Hallucination: Does the answer contain hallucinations?
    - Summarization: Quality of summarization
    """

    def __init__(
        self,
        judge_model: str | None = None,
        metrics: list[str] | None = None,
        threshold: float = 0.7,
    ):
        settings = get_settings()
        self.judge_model = judge_model or settings.deepeval_judge_model
        self.threshold = threshold
        self._enabled_metrics = metrics or ["answer_relevancy"]
        self._metrics_initialized = False
        self._metric_instances = {}

    @property
    def name(self) -> str:
        return "deepeval"

    @property
    def metrics(self) -> list[str]:
        return [f"deepeval_{m}" for m in self._enabled_metrics]

    def _initialize_metrics(self):
        """Lazy initialize DeepEval metrics."""
        if self._metrics_initialized:
            return

        try:
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                FaithfulnessMetric,
                HallucinationMetric,
            )

            metric_classes = {
                "answer_relevancy": AnswerRelevancyMetric,
                "faithfulness": FaithfulnessMetric,
                "hallucination": HallucinationMetric,
            }

            for metric_name in self._enabled_metrics:
                if metric_name in metric_classes:
                    self._metric_instances[metric_name] = metric_classes[metric_name](
                        model=self.judge_model,
                        threshold=self.threshold,
                    )
                    logger.debug(f"Initialized DeepEval metric: {metric_name}")

            self._metrics_initialized = True
            logger.info(
                "DeepEval metrics initialized",
                metrics=list(self._metric_instances.keys()),
                judge_model=self.judge_model,
            )

        except ImportError:
            logger.error("deepeval not installed. Run: pip install deepeval")
            raise

    async def evaluate(
        self,
        prompt: CanonicalPrompt,
        replay: ReplayResult,
        reference: str | None = None,
    ) -> dict[str, MetricScore]:
        """
        Evaluate replay using DeepEval LLM-as-judge.
        
        Args:
            prompt: Original prompt
            replay: Replay result to evaluate
            reference: Optional reference output
            
        Returns:
            Dict of DeepEval metrics with self-explaining reasons
        """
        self._initialize_metrics()

        if not replay.completion or replay.refused:
            return self._empty_scores()

        try:
            from deepeval.test_case import LLMTestCase

            # Create test case
            test_case = LLMTestCase(
                input=prompt.prompt_text,
                actual_output=replay.completion,
                expected_output=reference or prompt.completion.text,
                retrieval_context=None,  # Add context if RAG
            )

            results = {}
            for metric_name, metric in self._metric_instances.items():
                try:
                    # Measure is synchronous in DeepEval
                    import asyncio
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, lambda: metric.measure(test_case))

                    results[f"deepeval_{metric_name}"] = MetricScore(
                        metric_name=f"deepeval_{metric_name}",
                        score=max(0.0, min(1.0, metric.score or 0.0)),
                        reason=metric.reason or "No reason provided",
                        passed=metric.is_successful(),
                        threshold=self.threshold,
                        metadata={
                            "judge_model": self.judge_model,
                        },
                    )
                except Exception as e:
                    logger.warning(
                        f"DeepEval metric {metric_name} failed",
                        error=str(e),
                    )
                    results[f"deepeval_{metric_name}"] = MetricScore(
                        metric_name=f"deepeval_{metric_name}",
                        score=0.0,
                        reason=f"Evaluation error: {str(e)}",
                        passed=False,
                    )

            return results

        except Exception as e:
            logger.error("DeepEval evaluation failed", error=str(e))
            return self._empty_scores()

    def _empty_scores(self) -> dict[str, MetricScore]:
        """Return empty scores for failed evaluation."""
        return {
            f"deepeval_{m}": MetricScore(
                metric_name=f"deepeval_{m}",
                score=0.0,
                reason="Evaluation failed or refused output",
                passed=False,
            )
            for m in self._enabled_metrics
        }


class SimpleLLMJudge(BaseEvaluator):
    """
    Simplified LLM-as-judge without DeepEval dependency.
    
    Uses Portkey to call the judge model directly for cases
    where DeepEval is not available or too slow.
    """

    JUDGE_PROMPT = """You are evaluating the quality of an AI response.

Question/Prompt: {prompt}

AI Response: {response}

Reference Answer (if available): {reference}

Rate the response on a scale of 0.0 to 1.0 based on:
1. Relevance: Does it answer the question?
2. Accuracy: Is the information correct?
3. Completeness: Does it cover the key points?
4. Coherence: Is it well-organized and clear?

Respond with ONLY a JSON object:
{{"score": 0.0-1.0, "reason": "Brief explanation"}}
"""

    def __init__(
        self,
        judge_model: str = "gpt-4o-mini",
        portkey_api_key: str | None = None,
    ):
        self.judge_model = judge_model
        self.portkey_api_key = portkey_api_key or get_settings().portkey_api_key
        self._client = None

    @property
    def name(self) -> str:
        return "simple_llm_judge"

    @property
    def metrics(self) -> list[str]:
        return ["llm_judge_score"]

    async def evaluate(
        self,
        prompt: CanonicalPrompt,
        replay: ReplayResult,
        reference: str | None = None,
    ) -> dict[str, MetricScore]:
        """Evaluate using simple LLM-as-judge."""
        import json
        import httpx

        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url="https://api.portkey.ai/v1",
                headers={
                    "x-portkey-api-key": self.portkey_api_key,
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )

        judge_prompt = self.JUDGE_PROMPT.format(
            prompt=prompt.prompt_text,
            response=replay.completion,
            reference=reference or prompt.completion.text or "Not provided",
        )

        try:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self.judge_model,
                    "messages": [{"role": "user", "content": judge_prompt}],
                    "temperature": 0.0,
                    "max_tokens": 200,
                },
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            # Parse JSON from response
            result = json.loads(content)
            score = float(result.get("score", 0.0))
            reason = result.get("reason", "No reason provided")

            return {
                "llm_judge_score": MetricScore(
                    metric_name="llm_judge_score",
                    score=max(0.0, min(1.0, score)),
                    reason=reason,
                    passed=score >= 0.7,
                    threshold=0.7,
                    metadata={"judge_model": self.judge_model},
                )
            }

        except Exception as e:
            logger.error("LLM judge evaluation failed", error=str(e))
            return {
                "llm_judge_score": MetricScore(
                    metric_name="llm_judge_score",
                    score=0.0,
                    reason=f"Evaluation error: {str(e)}",
                    passed=False,
                )
            }
