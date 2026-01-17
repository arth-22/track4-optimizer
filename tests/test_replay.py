"""Tests for replay engine."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.replay.engine import ReplayEngine
from src.replay.model_registry import ModelRegistry, ModelConfig, Provider, PricingInfo


class TestModelRegistry:
    """Tests for model registry."""

    def test_get_default_models(self):
        """Test default models are returned."""
        models = ModelRegistry.get_default_models()
        
        assert len(models) == 5
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models
        assert "claude-sonnet-4-5" in models
        assert "claude-haiku-4-5-20251001" in models
        assert "gemini-2.5-flash" in models

    def test_get_model(self):
        """Test getting a specific model."""
        model = ModelRegistry.get_model("gpt-4o")
        
        assert model is not None
        assert model.model_id == "gpt-4o"
        assert model.provider == Provider.OPENAI
        assert model.provider_slug == "openai"

    def test_get_model_not_found(self):
        """Test getting non-existent model."""
        model = ModelRegistry.get_model("nonexistent-model")
        
        assert model is None

    def test_get_model_slug(self):
        """Test Model Catalog slug generation."""
        model = ModelRegistry.get_model("gpt-4o")
        slug = model.get_model_slug()
        
        assert slug == "@openai/gpt-4o"

    def test_get_model_slug_vertex(self):
        """Test Vertex AI model slug."""
        model = ModelRegistry.get_model("gemini-2.5-flash")
        slug = model.get_model_slug()
        
        assert slug == "@vertex/gemini-2.5-flash"

    def test_list_models_by_provider(self):
        """Test filtering models by provider."""
        openai_models = ModelRegistry.list_models(provider=Provider.OPENAI)
        
        assert len(openai_models) >= 2
        assert all(m.provider == Provider.OPENAI for m in openai_models)

    def test_pricing_calculation(self):
        """Test cost calculation."""
        pricing = PricingInfo(input_per_million=2.50, output_per_million=10.00)
        
        # 1000 input tokens, 500 output tokens
        cost = pricing.calculate_cost(1000, 500)
        
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert abs(cost - expected) < 0.0001

    def test_calculate_cost_via_registry(self):
        """Test cost calculation through registry."""
        cost = ModelRegistry.calculate_cost("gpt-4o-mini", 1000, 500)
        
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        expected = (1000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
        assert abs(cost - expected) < 0.0001


class TestReplayEngine:
    """Tests for replay engine."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mocked client."""
        with patch('src.replay.engine.AsyncOpenAI') as mock_client:
            engine = ReplayEngine(portkey_api_key="test-key")
            engine.client = mock_client.return_value
            return engine

    @pytest.mark.asyncio
    async def test_replay_single_success(self, mock_engine, sample_prompt):
        """Test successful single prompt replay."""
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 100
        mock_response.usage.total_tokens = 150
        mock_response.model = "gpt-4o"
        
        mock_engine.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        model = ModelRegistry.get_model("gpt-4o")
        result = await mock_engine.replay_single(sample_prompt, model)
        
        assert result.success is True
        assert result.completion == "Test response"
        assert result.input_tokens == 50
        assert result.output_tokens == 100
        assert result.model_id == "gpt-4o"

    @pytest.mark.asyncio
    async def test_replay_single_failure(self, mock_engine, sample_prompt):
        """Test replay failure handling."""
        mock_engine.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        model = ModelRegistry.get_model("gpt-4o")
        result = await mock_engine.replay_single(sample_prompt, model)
        
        assert result.success is False
        assert "API Error" in result.error

    @pytest.mark.asyncio
    async def test_replay_uses_model_slug(self, mock_engine, sample_prompt):
        """Test that replay uses Model Catalog format."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 20
        mock_response.model = "gpt-4o"
        
        mock_engine.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        model = ModelRegistry.get_model("gpt-4o")
        await mock_engine.replay_single(sample_prompt, model)
        
        # Verify model slug was used
        call_args = mock_engine.client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "@openai/gpt-4o"

    def test_detect_refusal_content_filter(self, mock_engine):
        """Test refusal detection for content filter."""
        is_refused = mock_engine._detect_refusal("", "content_filter")
        assert is_refused is True

    def test_detect_refusal_patterns(self, mock_engine):
        """Test refusal detection for common patterns."""
        refusal_texts = [
            "I cannot help with that request.",
            "I'm not able to assist with this.",
            "I must decline this request.",
        ]
        
        for text in refusal_texts:
            is_refused = mock_engine._detect_refusal(text, "stop")
            assert is_refused is True, f"Should detect refusal: {text}"

    def test_no_refusal_normal_response(self, mock_engine):
        """Test normal responses are not flagged as refusals."""
        normal_texts = [
            "Here's the answer to your question.",
            "The capital of France is Paris.",
            "def hello(): print('hello')",
        ]
        
        for text in normal_texts:
            is_refused = mock_engine._detect_refusal(text, "stop")
            assert is_refused is False, f"Should not detect refusal: {text}"
