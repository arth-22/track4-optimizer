"""Statistical analysis for confidence intervals and significance testing."""

import numpy as np
from scipy import stats
import structlog

from src.config import get_settings
from src.models.recommendation import ConfidenceInterval

logger = structlog.get_logger()


class StatisticalAnalyzer:
    """
    Provides statistical rigor for cost-quality recommendations.
    
    Calculates:
    - Confidence intervals for metrics
    - Statistical significance between models
    - Effect sizes for practical significance
    """

    def __init__(self, confidence_level: float | None = None):
        settings = get_settings()
        self.confidence_level = confidence_level or settings.confidence_level

    def calculate_confidence_interval(
        self,
        values: list[float],
        confidence: float | None = None,
    ) -> ConfidenceInterval:
        """
        Calculate mean and confidence interval for a sample.
        
        Uses t-distribution for small samples.
        
        Args:
            values: Sample values
            confidence: Confidence level (default: from settings)
            
        Returns:
            ConfidenceInterval with mean, lower, upper bounds
        """
        confidence = confidence or self.confidence_level

        if not values:
            return ConfidenceInterval(
                value=0.0,
                lower=0.0,
                upper=0.0,
                confidence_level=confidence,
                sample_size=0,
            )

        n = len(values)
        mean = float(np.mean(values))

        if n < 2:
            return ConfidenceInterval(
                value=mean,
                lower=mean,
                upper=mean,
                confidence_level=confidence,
                sample_size=n,
            )

        # Standard error
        se = float(stats.sem(values))

        # t-value for confidence interval
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = se * t_value

        return ConfidenceInterval(
            value=mean,
            lower=mean - margin,
            upper=mean + margin,
            confidence_level=confidence,
            sample_size=n,
        )

    def compare_means(
        self,
        group_a: list[float],
        group_b: list[float],
        paired: bool = True,
    ) -> dict:
        """
        Compare means of two groups with statistical testing.
        
        Args:
            group_a: First group of values
            group_b: Second group of values
            paired: Whether samples are paired (same prompts)
            
        Returns:
            Dict with t-statistic, p-value, effect size, and interpretation
        """
        if len(group_a) < 2 or len(group_b) < 2:
            return {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "effect_size": 0.0,
                "is_significant": False,
                "interpretation": "Insufficient data for comparison",
            }

        try:
            if paired and len(group_a) == len(group_b):
                # Paired t-test (same prompts, different models)
                t_stat, p_value = stats.ttest_rel(group_a, group_b)
                
                # Effect size: Cohen's d for paired samples
                diff = np.array(group_a) - np.array(group_b)
                effect_size = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff) > 0 else 0.0
            else:
                # Independent t-test
                t_stat, p_value = stats.ttest_ind(group_a, group_b)
                
                # Effect size: Cohen's d for independent samples
                pooled_std = np.sqrt(
                    ((len(group_a) - 1) * np.var(group_a, ddof=1) +
                     (len(group_b) - 1) * np.var(group_b, ddof=1)) /
                    (len(group_a) + len(group_b) - 2)
                )
                effect_size = float((np.mean(group_a) - np.mean(group_b)) / pooled_std) if pooled_std > 0 else 0.0

            is_significant = p_value < (1 - self.confidence_level)
            interpretation = self._interpret_comparison(p_value, effect_size, is_significant)

            return {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "effect_size": effect_size,
                "is_significant": is_significant,
                "interpretation": interpretation,
            }

        except Exception as e:
            logger.warning("Statistical comparison failed", error=str(e))
            return {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "effect_size": 0.0,
                "is_significant": False,
                "interpretation": f"Comparison error: {str(e)}",
            }

    def _interpret_comparison(
        self,
        p_value: float,
        effect_size: float,
        is_significant: bool,
    ) -> str:
        """Generate human-readable interpretation of comparison."""
        if not is_significant:
            return "No statistically significant difference detected"

        # Interpret effect size
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            effect_desc = "negligible"
        elif abs_effect < 0.5:
            effect_desc = "small"
        elif abs_effect < 0.8:
            effect_desc = "medium"
        else:
            effect_desc = "large"

        direction = "better" if effect_size > 0 else "worse"

        return (
            f"Statistically significant difference (p={p_value:.4f}) "
            f"with {effect_desc} effect size ({direction})"
        )

    def calculate_percentile_ci(
        self,
        values: list[float],
        percentile: float = 50,
        confidence: float | None = None,
    ) -> ConfidenceInterval:
        """
        Calculate confidence interval for a percentile using bootstrap.
        
        Useful for latency P95, P99 confidence intervals.
        """
        confidence = confidence or self.confidence_level

        if len(values) < 10:
            percentile_value = float(np.percentile(values, percentile)) if values else 0.0
            return ConfidenceInterval(
                value=percentile_value,
                lower=percentile_value,
                upper=percentile_value,
                confidence_level=confidence,
                sample_size=len(values),
            )

        # Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_percentiles = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_percentiles.append(np.percentile(sample, percentile))

        alpha = 1 - confidence
        lower = float(np.percentile(bootstrap_percentiles, alpha / 2 * 100))
        upper = float(np.percentile(bootstrap_percentiles, (1 - alpha / 2) * 100))
        value = float(np.percentile(values, percentile))

        return ConfidenceInterval(
            value=value,
            lower=lower,
            upper=upper,
            confidence_level=confidence,
            sample_size=len(values),
        )

    def calculate_summary_statistics(
        self,
        values: list[float],
    ) -> dict:
        """Calculate comprehensive summary statistics."""
        if not values:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }
