"""LLM-powered anomaly detection in evaluation results."""

import structlog
from openai import AsyncOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL

from src.config import get_settings
from src.models.evaluation import EvaluationResult

logger = structlog.get_logger()


class AnomalyDetector:
    """
    Uses LLM to identify unusual patterns in evaluation results.
    
    Detects:
    - Models performing unexpectedly well/poorly
    - Unusual cost patterns
    - Quality anomalies
    - Statistical outliers
    """
    
    ANALYSIS_PROMPT = """You are an AI systems analyst. Analyze these model evaluation results and identify any anomalies or unusual patterns.

EVALUATION SUMMARY:
{summary}

Look for:
1. Models performing unexpectedly well or poorly
2. Unusual cost/quality trade-offs
3. High refusal rates
4. Latency outliers
5. Any patterns that warrant investigation

Respond with a JSON object:
{{
  "anomalies": [
    {{"type": "...", "model": "...", "description": "...", "severity": "low|medium|high"}}
  ],
  "insights": ["insight1", "insight2"],
  "recommended_actions": ["action1", "action2"]
}}
"""

    def __init__(self, api_key: str | None = None):
        settings = get_settings()
        self.api_key = api_key or settings.portkey_api_key
        
        self.client = AsyncOpenAI(
            base_url=PORTKEY_GATEWAY_URL,
            api_key=self.api_key,
        )
    
    async def detect_anomalies(
        self,
        evaluations: list[EvaluationResult],
    ) -> dict:
        """
        Analyze evaluation results for anomalies.
        
        Args:
            evaluations: List of evaluation results to analyze
            
        Returns:
            Dict with anomalies, insights, and recommended actions
        """
        import json
        
        # Build summary for LLM
        summary = self._build_summary(evaluations)
        
        prompt = self.ANALYSIS_PROMPT.format(summary=summary)
        
        try:
            response = await self.client.chat.completions.create(
                model="@openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500,
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(content[json_start:json_end])
                logger.info(
                    "Anomaly detection complete",
                    anomaly_count=len(result.get("anomalies", [])),
                )
                return result
                
        except Exception as e:
            logger.warning("Anomaly detection failed", error=str(e))
        
        return {"anomalies": [], "insights": [], "recommended_actions": []}
    
    def _build_summary(self, evaluations: list[EvaluationResult]) -> str:
        """Build a summary string for LLM analysis."""
        # Group by model
        by_model: dict[str, list] = {}
        for e in evaluations:
            if e.model_id not in by_model:
                by_model[e.model_id] = []
            by_model[e.model_id].append(e)
        
        lines = []
        for model_id, evals in by_model.items():
            scores = [e.composite_score for e in evals]
            costs = [e.cost for e in evals if e.cost]
            latencies = [e.latency_ms for e in evals if e.latency_ms]
            refusals = sum(1 for e in evals if e.refused)
            
            avg_score = sum(scores) / len(scores) if scores else 0
            avg_cost = sum(costs) / len(costs) if costs else 0
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            refusal_rate = refusals / len(evals) * 100 if evals else 0
            
            lines.append(f"""
Model: {model_id}
  - Evaluations: {len(evals)}
  - Avg Quality Score: {avg_score:.2f}
  - Avg Cost: ${avg_cost:.6f}
  - Avg Latency: {avg_latency:.0f}ms
  - Refusal Rate: {refusal_rate:.1f}%
""")
        
        return "\n".join(lines)
    
    async def get_quick_insights(
        self,
        evaluations: list[EvaluationResult],
    ) -> list[str]:
        """Get quick statistical insights without LLM."""
        insights = []
        
        # Group by model
        by_model: dict[str, list] = {}
        for e in evaluations:
            if e.model_id not in by_model:
                by_model[e.model_id] = []
            by_model[e.model_id].append(e)
        
        # Find best/worst performers
        model_scores = {
            model: sum(e.composite_score for e in evals) / len(evals)
            for model, evals in by_model.items()
        }
        
        if model_scores:
            best = max(model_scores, key=model_scores.get)
            worst = min(model_scores, key=model_scores.get)
            
            if model_scores[best] - model_scores[worst] > 0.2:
                insights.append(
                    f"Large quality gap: {best} ({model_scores[best]:.2f}) vs "
                    f"{worst} ({model_scores[worst]:.2f})"
                )
        
        # Check for high refusal rates
        for model, evals in by_model.items():
            refusal_rate = sum(1 for e in evals if e.refused) / len(evals)
            if refusal_rate > 0.1:
                insights.append(f"High refusal rate for {model}: {refusal_rate:.0%}")
        
        return insights
