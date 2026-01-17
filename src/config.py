"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Portkey
    portkey_api_key: str = ""
    portkey_base_url: str = "https://api.portkey.ai/v1"

    # AI Provider Slugs (Model Catalog format)
    # These are the slugs you create in Portkey's Model Catalog
    openai_provider_slug: str = "openai"
    anthropic_provider_slug: str = "anthropic"
    google_provider_slug: str = "google"

    # Optional API keys (for direct provider access, e.g., DeepEval judge)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""

    # Application
    log_level: str = "INFO"
    database_url: str = "sqlite+aiosqlite:///./data/optimizer.db"

    # Replay settings
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 60
    retry_attempts: int = 3

    # Evaluation settings
    deepeval_judge_model: str = "gpt-4o"
    enable_bertscore: bool = True
    enable_deepeval: bool = True

    # Analysis settings
    confidence_level: float = 0.95
    min_sample_size: int = 30


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
