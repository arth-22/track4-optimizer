"""Model registry with pricing and capabilities for Portkey Model Catalog."""

from dataclasses import dataclass, field
from enum import Enum


class Provider(str, Enum):
    """LLM provider."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VERTEX = "vertex"  # Google via Vertex AI
    META = "meta"
    MISTRAL = "mistral"
    LOCAL = "local"


@dataclass
class PricingInfo:
    """Pricing information per 1M tokens."""

    input_per_million: float
    output_per_million: float
    currency: str = "USD"
    effective_date: str = "2025-01"

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD."""
        input_cost = (input_tokens / 1_000_000) * self.input_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_per_million
        return input_cost + output_cost


@dataclass
class ModelConfig:
    """Configuration for a model."""

    model_id: str
    provider: Provider
    pricing: PricingInfo
    max_context: int
    max_output: int = 4096
    capabilities: list[str] = field(default_factory=list)
    provider_slug: str = ""  # Portkey provider slug (e.g., "openai", "anthropic", "vertex")
    status: str = "active"  # active, deprecated, preview
    notes: str = ""

    def __post_init__(self):
        if not self.provider_slug:
            self.provider_slug = self.provider.value

    def supports_capability(self, capability: str) -> bool:
        """Check if model supports a capability."""
        if not self.capabilities:
            return True  # Assume all capabilities if not specified
        return capability in self.capabilities

    def get_model_slug(self) -> str:
        """
        Get the Model Catalog format: @provider_slug/model_id
        
        Example: @openai/gpt-4o, @anthropic/claude-sonnet-4-5
        """
        return f"@{self.provider_slug}/{self.model_id}"


class ModelRegistry:
    """
    Registry of available models with pricing and capabilities.
    
    Model slugs configured for Portkey Model Catalog (January 2025).
    """

    _models: dict[str, ModelConfig] = {
        # OpenAI Models
        "gpt-4o": ModelConfig(
            model_id="gpt-4o",
            provider=Provider.OPENAI,
            pricing=PricingInfo(input_per_million=2.50, output_per_million=10.00),
            max_context=128000,
            max_output=16384,
            capabilities=["chat", "function_calling", "vision", "json_mode"],
            provider_slug="openai",
        ),
        "gpt-4o-mini": ModelConfig(
            model_id="gpt-4o-mini",
            provider=Provider.OPENAI,
            pricing=PricingInfo(input_per_million=0.15, output_per_million=0.60),
            max_context=128000,
            max_output=16384,
            capabilities=["chat", "function_calling", "vision", "json_mode"],
            provider_slug="openai",
        ),
        # Anthropic Models
        "claude-sonnet-4-5": ModelConfig(
            model_id="claude-sonnet-4-5",
            provider=Provider.ANTHROPIC,
            pricing=PricingInfo(input_per_million=3.00, output_per_million=15.00),
            max_context=200000,
            max_output=16000,
            capabilities=["chat", "tool_use", "vision"],
            provider_slug="anthropic",
        ),
        "claude-haiku-4-5-20251001": ModelConfig(
            model_id="claude-haiku-4-5-20251001",
            provider=Provider.ANTHROPIC,
            pricing=PricingInfo(input_per_million=0.80, output_per_million=4.00),
            max_context=200000,
            max_output=8192,
            capabilities=["chat", "tool_use", "vision"],
            provider_slug="anthropic",
        ),
        # Google Models (via Vertex AI)
        "gemini-2.5-flash": ModelConfig(
            model_id="gemini-2.5-flash",
            provider=Provider.VERTEX,
            pricing=PricingInfo(input_per_million=0.10, output_per_million=0.40),
            max_context=1000000,
            max_output=8192,
            capabilities=["chat", "tool_use", "vision", "audio"],
            provider_slug="vertex",
        ),
    }

    @classmethod
    def get_model(cls, model_id: str) -> ModelConfig | None:
        """Get model configuration by ID."""
        return cls._models.get(model_id)

    @classmethod
    def list_models(
        cls,
        provider: Provider | None = None,
        capability: str | None = None,
        status: str = "active",
    ) -> list[ModelConfig]:
        """List models with optional filtering."""
        models = list(cls._models.values())

        if provider:
            models = [m for m in models if m.provider == provider]
        if capability:
            models = [m for m in models if m.supports_capability(capability)]
        if status:
            models = [m for m in models if m.status == status]

        return models

    @classmethod
    def get_default_models(cls) -> list[str]:
        """Get default set of models for evaluation."""
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "claude-sonnet-4-5",
            "claude-haiku-4-5-20251001",
            "gemini-2.5-flash",
        ]

    @classmethod
    def register_model(cls, config: ModelConfig) -> None:
        """Register a custom model configuration."""
        cls._models[config.model_id] = config

    @classmethod
    def get_pricing(cls, model_id: str) -> PricingInfo | None:
        """Get pricing info for a model."""
        model = cls.get_model(model_id)
        return model.pricing if model else None

    @classmethod
    def calculate_cost(
        cls, model_id: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost for a model usage."""
        pricing = cls.get_pricing(model_id)
        if pricing is None:
            return 0.0
        return pricing.calculate_cost(input_tokens, output_tokens)

    @classmethod
    def get_model_slug(cls, model_id: str) -> str | None:
        """Get the Portkey Model Catalog slug for a model."""
        model = cls.get_model(model_id)
        return model.get_model_slug() if model else None
