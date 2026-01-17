"""BERTScore evaluator for semantic similarity."""

import hashlib
import structlog

from src.evaluation.base import BaseEvaluator
from src.models.canonical import CanonicalPrompt
from src.models.evaluation import MetricScore, ReplayResult

logger = structlog.get_logger()

# Module-level cache for BERTScore results (persists across instances)
_bertscore_cache: dict[str, dict[str, MetricScore]] = {}


class BertScoreEvaluator(BaseEvaluator):
    """
    Evaluator using BERTScore for semantic similarity.
    
    BERTScore uses contextual embeddings to measure similarity,
    making it more robust than token-overlap metrics like ROUGE.
    
    Features:
    - Automatic language detection
    - Caching for efficiency
    - Runs locally without LLM API calls
    """

    # BERTScore supported languages for rescale_with_baseline
    SUPPORTED_LANGS = {"en", "de", "fr", "es", "zh", "ja", "ko", "ru", "ar", "nl", "pt"}

    def __init__(
        self,
        model_type: str = "distilbert-base-uncased",  # Fast & small: ~250MB
        device: str | None = None,
        batch_size: int = 64,
        auto_detect_language: bool = True,
        enable_cache: bool = True,
    ):
        self.model_type = model_type
        self.device = device
        self.batch_size = batch_size
        self.auto_detect_language = auto_detect_language
        self.enable_cache = enable_cache
        self._scorers: dict[str, object] = {}  # Cache scorers per language
        self._default_lang = "en"

    @property
    def name(self) -> str:
        return "bertscore"

    @property
    def metrics(self) -> list[str]:
        return ["bertscore_precision", "bertscore_recall", "bertscore_f1"]

    def requires_reference(self) -> bool:
        return True

    def _detect_language(self, text: str) -> str:
        """Detect language of text, falling back to English."""
        if not self.auto_detect_language:
            return self._default_lang
        
        try:
            from langdetect import detect
            detected = detect(text[:500])  # Use first 500 chars for speed
            if detected in self.SUPPORTED_LANGS:
                return detected
            return self._default_lang
        except Exception:
            return self._default_lang

    def _get_cache_key(self, candidate: str, reference: str) -> str:
        """Generate cache key from candidate-reference pair."""
        combined = f"{candidate}|||{reference}"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    def _get_scorer(self, lang: str = "en"):
        """Lazy load BERTScore scorer for specified language."""
        if lang not in self._scorers:
            try:
                from bert_score import BERTScorer
                self._scorers[lang] = BERTScorer(
                    model_type=self.model_type,
                    device=self.device,
                    batch_size=self.batch_size,
                    lang=lang,
                    rescale_with_baseline=True,
                )
                logger.info("BERTScore scorer initialized", model=self.model_type, lang=lang)
            except ImportError:
                logger.error("bert-score not installed. Run: pip install bert-score")
                raise
        return self._scorers[lang]

    async def evaluate(
        self,
        prompt: CanonicalPrompt,
        replay: ReplayResult,
        reference: str | None = None,
    ) -> dict[str, MetricScore]:
        """
        Evaluate replay using BERTScore.
        
        Args:
            prompt: Original prompt
            replay: Replay result to evaluate
            reference: Reference output (uses original completion if None)
            
        Returns:
            Dict of BERTScore metrics
        """
        # Use original completion as reference if not provided
        if reference is None:
            reference = prompt.completion.text

        if not reference or not replay.completion:
            return self._empty_scores()

        try:
            scores = await self._compute_bertscore([replay.completion], [reference])
            return scores[0]
        except Exception as e:
            logger.error("BERTScore evaluation failed", error=str(e))
            return self._empty_scores()

    async def evaluate_batch(
        self,
        prompts: list[CanonicalPrompt],
        replays: list[ReplayResult],
        references: list[str | None] | None = None,
    ) -> list[dict[str, MetricScore]]:
        """Batch evaluation for efficiency."""
        # Prepare references
        refs = []
        for i, prompt in enumerate(prompts):
            if references and references[i]:
                refs.append(references[i])
            else:
                refs.append(prompt.completion.text)

        candidates = [r.completion for r in replays]

        try:
            return await self._compute_bertscore(candidates, refs)
        except Exception as e:
            logger.error("Batch BERTScore evaluation failed", error=str(e))
            return [self._empty_scores() for _ in prompts]

    async def _compute_bertscore(
        self,
        candidates: list[str],
        references: list[str],
    ) -> list[dict[str, MetricScore]]:
        """Compute BERTScore for candidate-reference pairs."""
        import asyncio

        # BERTScore is CPU/GPU intensive, run in thread pool
        def compute():
            scorer = self._get_scorer()
            P, R, F1 = scorer.score(candidates, references)
            return P.tolist(), R.tolist(), F1.tolist()

        loop = asyncio.get_event_loop()
        P, R, F1 = await loop.run_in_executor(None, compute)

        results = []
        for p, r, f1 in zip(P, R, F1):
            results.append({
                "bertscore_precision": MetricScore(
                    metric_name="bertscore_precision",
                    score=max(0.0, min(1.0, p)),
                    reason="Semantic precision via BERTScore",
                    passed=p >= 0.5,
                    threshold=0.5,
                ),
                "bertscore_recall": MetricScore(
                    metric_name="bertscore_recall",
                    score=max(0.0, min(1.0, r)),
                    reason="Semantic recall via BERTScore",
                    passed=r >= 0.5,
                    threshold=0.5,
                ),
                "bertscore_f1": MetricScore(
                    metric_name="bertscore_f1",
                    score=max(0.0, min(1.0, f1)),
                    reason="Semantic F1 via BERTScore",
                    passed=f1 >= 0.5,
                    threshold=0.5,
                ),
            })

        return results

    def _empty_scores(self) -> dict[str, MetricScore]:
        """Return empty scores for failed evaluation."""
        return {
            "bertscore_precision": MetricScore(
                metric_name="bertscore_precision",
                score=0.0,
                reason="Evaluation failed or empty input",
                passed=False,
            ),
            "bertscore_recall": MetricScore(
                metric_name="bertscore_recall",
                score=0.0,
                reason="Evaluation failed or empty input",
                passed=False,
            ),
            "bertscore_f1": MetricScore(
                metric_name="bertscore_f1",
                score=0.0,
                reason="Evaluation failed or empty input",
                passed=False,
            ),
        }
