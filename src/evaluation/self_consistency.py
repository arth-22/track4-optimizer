"""Self-consistency evaluator using multiple response sampling."""

from typing import Any
import asyncio

import structlog

from openai import AsyncOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL

from src.config import get_settings
from src.evaluation.base import BaseEvaluator
from src.models.canonical import CanonicalPrompt
from src.models.evaluation import MetricScore, ReplayResult

logger = structlog.get_logger()


class SelfConsistencyEvaluator(BaseEvaluator):
    """
    Evaluates response quality by generating multiple samples and measuring agreement.
    
    High agreement across samples indicates confident, reliable response.
    Low agreement indicates uncertainty or ambiguity.
    """
    
    name = "self_consistency"
    metrics = ["consistency_score", "agreement_rate"]
    
    def __init__(
        self,
        num_samples: int = 3,
        temperature: float = 0.7,
        portkey_api_key: str | None = None,
    ):
        """
        Initialize self-consistency evaluator.
        
        Args:
            num_samples: Number of samples to generate for comparison
            temperature: Temperature for sampling (higher = more diverse)
            portkey_api_key: Portkey API key (uses settings if None)
        """
        settings = get_settings()
        self.num_samples = num_samples
        self.temperature = temperature
        
        self.client = AsyncOpenAI(
            base_url=PORTKEY_GATEWAY_URL,
            api_key=portkey_api_key or settings.portkey_api_key,
        )
    
    async def evaluate(
        self,
        prompt: CanonicalPrompt,
        replay: ReplayResult,
        reference: str | None = None,
    ) -> dict[str, MetricScore]:
        """
        Evaluate by generating multiple responses and measuring agreement.
        
        Args:
            prompt: Original prompt
            replay: Single replay result (will generate more samples)
            reference: Not used for self-consistency
            
        Returns:
            Dict with consistency_score and agreement_rate
        """
        try:
            # Generate additional samples
            samples = [replay.completion]  # Start with the original
            
            messages = prompt.to_openai_format()
            
            for _ in range(self.num_samples - 1):
                try:
                    response = await self.client.chat.completions.create(
                        model=replay.model_id,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=1000,
                    )
                    samples.append(response.choices[0].message.content or "")
                except Exception as e:
                    logger.warning("Failed to generate sample", error=str(e))
            
            if len(samples) < 2:
                return self._default_scores()
            
            # Calculate consistency metrics
            consistency_score = self._calculate_consistency(samples)
            agreement_rate = self._calculate_agreement(samples)
            
            return {
                "consistency_score": MetricScore(
                    name="consistency_score",
                    score=consistency_score,
                    passed=consistency_score >= 0.7,
                    reason=f"Based on {len(samples)} samples",
                    metadata={"num_samples": len(samples)},
                ),
                "agreement_rate": MetricScore(
                    name="agreement_rate",
                    score=agreement_rate,
                    passed=agreement_rate >= 0.6,
                    reason=f"Semantic agreement across responses",
                    metadata={"num_samples": len(samples)},
                ),
            }
            
        except Exception as e:
            logger.error("Self-consistency evaluation failed", error=str(e))
            return self._default_scores()
    
    def _calculate_consistency(self, samples: list[str]) -> float:
        """
        Calculate consistency score based on response length variance.
        
        Lower variance in length = higher consistency.
        """
        if len(samples) < 2:
            return 1.0
        
        lengths = [len(s) for s in samples]
        mean_length = sum(lengths) / len(lengths)
        
        if mean_length == 0:
            return 0.0
        
        # Coefficient of variation (CV)
        variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
        std_dev = variance ** 0.5
        cv = std_dev / mean_length
        
        # Convert CV to a 0-1 score (lower CV = higher consistency)
        # CV of 0 = perfect consistency (1.0)
        # CV of 1+ = low consistency (0.0)
        consistency = max(0.0, 1.0 - cv)
        
        return round(consistency, 3)
    
    def _calculate_agreement(self, samples: list[str]) -> float:
        """
        Calculate semantic agreement using word overlap (jaccard similarity).
        
        For production, could use embedding similarity instead.
        """
        if len(samples) < 2:
            return 1.0
        
        # Tokenize each sample
        token_sets = []
        for sample in samples:
            tokens = set(sample.lower().split())
            token_sets.append(tokens)
        
        # Calculate pairwise Jaccard similarity
        similarities = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                set_a = token_sets[i]
                set_b = token_sets[j]
                
                if not set_a and not set_b:
                    similarities.append(1.0)
                elif not set_a or not set_b:
                    similarities.append(0.0)
                else:
                    intersection = len(set_a & set_b)
                    union = len(set_a | set_b)
                    similarities.append(intersection / union)
        
        if not similarities:
            return 1.0
        
        return round(sum(similarities) / len(similarities), 3)
    
    def _default_scores(self) -> dict[str, MetricScore]:
        """Return default scores when evaluation fails."""
        return {
            "consistency_score": MetricScore(
                name="consistency_score",
                score=0.5,
                passed=False,
                reason="Insufficient samples for evaluation",
            ),
            "agreement_rate": MetricScore(
                name="agreement_rate",
                score=0.5,
                passed=False,
                reason="Insufficient samples for evaluation",
            ),
        }
    
    def get_config(self) -> dict[str, Any]:
        """Get evaluator configuration."""
        return {
            "name": self.name,
            "num_samples": self.num_samples,
            "temperature": self.temperature,
        }
