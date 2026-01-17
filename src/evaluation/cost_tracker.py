"""Cost and token tracking using tiktoken."""

import tiktoken
import structlog

from src.models.evaluation import CostResult
from src.replay.model_registry import ModelRegistry

logger = structlog.get_logger()


class CostTracker:
    """
    Accurate token counting and cost calculation for all providers.
    
    Uses tiktoken for tokenization with fallback encoding for non-OpenAI models.
    """

    # Encoding mapping for models
    ENCODING_MAP = {
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "o1": "o200k_base",
        "o1-mini": "o200k_base",
        "gpt-3.5-turbo": "cl100k_base",
        # Non-OpenAI models use cl100k_base as approximation
        "claude": "cl100k_base",
        "gemini": "cl100k_base",
        "llama": "cl100k_base",
        "mistral": "cl100k_base",
    }

    def __init__(self):
        self._encoders: dict[str, tiktoken.Encoding] = {}

    def get_encoder(self, model_id: str) -> tiktoken.Encoding:
        """Get tokenizer encoder for a model."""
        # Try to get encoding from map
        encoding_name = None
        for prefix, enc_name in self.ENCODING_MAP.items():
            if prefix in model_id.lower():
                encoding_name = enc_name
                break

        if encoding_name is None:
            encoding_name = "cl100k_base"  # Default fallback

        # Cache encoders
        if encoding_name not in self._encoders:
            self._encoders[encoding_name] = tiktoken.get_encoding(encoding_name)

        return self._encoders[encoding_name]

    def count_tokens(self, text: str, model_id: str = "gpt-4") -> int:
        """
        Count tokens in text for a specific model.
        
        Args:
            text: Text to tokenize
            model_id: Model ID to use for tokenization
            
        Returns:
            Token count
        """
        if not text:
            return 0

        encoder = self.get_encoder(model_id)
        return len(encoder.encode(text))

    def count_message_tokens(
        self,
        messages: list[dict[str, str]],
        model_id: str = "gpt-4",
    ) -> int:
        """
        Count tokens for a list of messages (chat format).
        
        Includes overhead for message formatting.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model_id: Model ID to use for tokenization
            
        Returns:
            Total token count including overhead
        """
        encoder = self.get_encoder(model_id)
        tokens = 0

        # Per-message overhead varies by model
        tokens_per_message = 3  # OpenAI default
        tokens_per_name = 1

        for message in messages:
            tokens += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    tokens += len(encoder.encode(value))
                if key == "name":
                    tokens += tokens_per_name

        tokens += 3  # Priming tokens for assistant reply

        return tokens

    def calculate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostResult:
        """
        Calculate cost for a model usage.
        
        Args:
            model_id: Model ID
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            CostResult with detailed breakdown
        """
        pricing = ModelRegistry.get_pricing(model_id)

        if pricing is None:
            logger.warning("No pricing found for model", model_id=model_id)
            return CostResult(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                input_cost_usd=0.0,
                output_cost_usd=0.0,
                total_cost_usd=0.0,
                model_id=model_id,
                pricing_version="unknown",
            )

        input_cost = (input_tokens / 1_000_000) * pricing.input_per_million
        output_cost = (output_tokens / 1_000_000) * pricing.output_per_million

        return CostResult(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            total_cost_usd=input_cost + output_cost,
            model_id=model_id,
            pricing_version=pricing.effective_date,
        )

    def estimate_cost_from_text(
        self,
        model_id: str,
        input_text: str,
        output_text: str,
    ) -> CostResult:
        """
        Estimate cost from text (useful for pre-run estimation).
        
        Args:
            model_id: Model ID
            input_text: Input/prompt text
            output_text: Output/completion text
            
        Returns:
            CostResult with estimated costs
        """
        input_tokens = self.count_tokens(input_text, model_id)
        output_tokens = self.count_tokens(output_text, model_id)
        return self.calculate_cost(model_id, input_tokens, output_tokens)

    def project_monthly_cost(
        self,
        cost_per_request: float,
        requests_per_day: int,
    ) -> float:
        """
        Project monthly cost from per-request cost.
        
        Args:
            cost_per_request: Average cost per request
            requests_per_day: Expected daily request volume
            
        Returns:
            Projected monthly cost in USD
        """
        return cost_per_request * requests_per_day * 30

    def compare_model_costs(
        self,
        model_a: str,
        model_b: str,
        input_tokens: int,
        output_tokens: int,
    ) -> dict:
        """
        Compare costs between two models.
        
        Args:
            model_a: First model ID
            model_b: Second model ID
            input_tokens: Token count for input
            output_tokens: Token count for output
            
        Returns:
            Comparison dict with costs and savings
        """
        cost_a = self.calculate_cost(model_a, input_tokens, output_tokens)
        cost_b = self.calculate_cost(model_b, input_tokens, output_tokens)

        difference = cost_a.total_cost_usd - cost_b.total_cost_usd
        percent_diff = (difference / cost_a.total_cost_usd * 100) if cost_a.total_cost_usd > 0 else 0

        return {
            "model_a": model_a,
            "model_b": model_b,
            "cost_a": cost_a.total_cost_usd,
            "cost_b": cost_b.total_cost_usd,
            "difference": difference,
            "percent_difference": percent_diff,
            "cheaper_model": model_b if difference > 0 else model_a,
        }
