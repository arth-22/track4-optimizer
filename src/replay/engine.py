"""Core replay engine for running prompts through different models via Portkey."""

import asyncio
import time
import uuid
from typing import AsyncIterator

from openai import AsyncOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL
import structlog

from src.config import get_settings
from src.models.canonical import CanonicalPrompt
from src.models.evaluation import ReplayResult
from src.replay.model_registry import ModelRegistry, ModelConfig, Provider
from src.replay.rate_limiter import RateLimiter, ConcurrencyLimiter

logger = structlog.get_logger()


class ReplayEngine:
    """
    Engine for replaying prompts through different models.
    
    Uses Portkey Gateway with OpenAI SDK for provider-agnostic model access.
    Supports all providers via Model Catalog slugs (@openai/gpt-4o, etc.)
    """

    def __init__(
        self,
        portkey_api_key: str | None = None,
        rate_limiter: RateLimiter | None = None,
        concurrency_limiter: ConcurrencyLimiter | None = None,
    ):
        settings = get_settings()
        self.api_key = portkey_api_key or settings.portkey_api_key
        self.timeout = settings.request_timeout_seconds
        self.retry_attempts = settings.retry_attempts

        self.rate_limiter = rate_limiter or RateLimiter()
        self.concurrency_limiter = concurrency_limiter or ConcurrencyLimiter()

        # Use OpenAI SDK with Portkey Gateway
        self.client = AsyncOpenAI(
            base_url=PORTKEY_GATEWAY_URL,
            api_key=self.api_key,
            timeout=float(self.timeout),
        )

        logger.info(
            "ReplayEngine initialized",
            gateway=PORTKEY_GATEWAY_URL,
            timeout=self.timeout,
        )

    async def replay_single(
        self,
        prompt: CanonicalPrompt,
        model: ModelConfig,
        temperature: float = 1.0,
    ) -> ReplayResult:
        """
        Replay a single prompt through a specified model.
        
        Args:
            prompt: The prompt to replay
            model: Target model configuration
            temperature: Sampling temperature
            
        Returns:
            ReplayResult with completion and metrics
        """
        replay_id = str(uuid.uuid4())
        provider = model.provider

        # Acquire rate limit and concurrency permits
        await self.rate_limiter.acquire(provider)
        provider_sem, global_sem = await self.concurrency_limiter.acquire(provider)

        try:
            # Get Model Catalog slug: @provider/model_id
            model_slug = model.get_model_slug()
            messages = prompt.to_openai_format()

            # Make request with retry
            start_time = time.perf_counter()
            response = await self._make_request_with_retry(
                model_slug=model_slug,
                messages=messages,
                temperature=temperature,
                max_tokens=min(4096, model.max_output),
            )
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Parse response
            return self._parse_response(
                response=response,
                prompt_id=prompt.id,
                replay_id=replay_id,
                model=model,
                latency_ms=latency_ms,
                temperature=temperature,
            )

        except Exception as e:
            logger.error(
                "Replay failed",
                prompt_id=prompt.id,
                model=model.model_id,
                model_slug=model.get_model_slug(),
                error=str(e),
            )
            return ReplayResult(
                prompt_id=prompt.id,
                replay_id=replay_id,
                model_id=model.model_id,
                provider=model.provider.value,
                completion="",
                finish_reason="error",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                latency_ms=0,
                cost_usd=0,
                success=False,
                error=str(e),
            )

        finally:
            self.concurrency_limiter.release(provider_sem, global_sem)

    async def replay_batch(
        self,
        prompts: list[CanonicalPrompt],
        models: list[str] | None = None,
        temperature: float = 1.0,
    ) -> AsyncIterator[ReplayResult]:
        """
        Replay multiple prompts through multiple models.
        
        Args:
            prompts: List of prompts to replay
            models: List of model IDs (uses defaults if None)
            temperature: Sampling temperature
            
        Yields:
            ReplayResult for each prompt-model combination
        """
        model_ids = models or ModelRegistry.get_default_models()
        model_configs = [
            ModelRegistry.get_model(m) for m in model_ids
            if ModelRegistry.get_model(m) is not None
        ]

        total = len(prompts) * len(model_configs)
        completed = 0

        logger.info(
            "Starting batch replay",
            prompts=len(prompts),
            models=[m.get_model_slug() for m in model_configs],
            total_replays=total,
        )

        # Create tasks for all combinations
        async def replay_and_yield(prompt: CanonicalPrompt, model: ModelConfig):
            return await self.replay_single(prompt, model, temperature)

        # Process in batches to avoid overwhelming the system
        batch_size = 20
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            
            tasks = []
            for prompt in batch_prompts:
                for model in model_configs:
                    tasks.append(replay_and_yield(prompt, model))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                completed += 1
                if isinstance(result, Exception):
                    logger.warning("Batch replay exception", error=str(result))
                    continue
                if completed % 50 == 0:
                    logger.info("Replay progress", completed=completed, total=total)
                yield result

        logger.info("Batch replay complete", completed=completed)

    async def replay_single_streaming(
        self,
        prompt: CanonicalPrompt,
        model: ModelConfig,
        temperature: float = 1.0,
    ) -> AsyncIterator[str]:
        """
        Stream replay response chunks for real-time display.
        
        Yields text chunks as they arrive from the model.
        Does not return full ReplayResult - use for UX only.
        
        Args:
            prompt: The prompt to replay
            model: Target model configuration
            temperature: Sampling temperature
            
        Yields:
            String chunks as they arrive
        """
        model_slug = model.get_model_slug()
        messages = prompt.to_openai_messages()
        
        try:
            stream = await self.client.chat.completions.create(
                model=model_slug,
                messages=messages,
                temperature=temperature,
                max_tokens=model.max_output,
                stream=True,
                extra_headers={"x-portkey-provider": model.provider_slug},
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error("Streaming replay failed", error=str(e), model=model.model_id)
            yield f"[Error: {str(e)}]"

    async def _make_request_with_retry(
        self,
        model_slug: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """Make API request with exponential backoff retry."""
        last_error = None

        for attempt in range(self.retry_attempts):
            try:
                response = await self.client.chat.completions.create(
                    model=model_slug,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response

            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if rate limited or server error
                if "429" in error_str or "rate" in error_str:
                    wait_time = 2 ** attempt
                    logger.warning(
                        "Rate limited, waiting",
                        model=model_slug,
                        wait_seconds=wait_time,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(wait_time)
                elif "500" in error_str or "502" in error_str or "503" in error_str:
                    wait_time = 2 ** attempt
                    logger.warning(
                        "Server error, retrying",
                        model=model_slug,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(wait_time)
                elif "timeout" in error_str:
                    wait_time = 2 ** attempt
                    logger.warning(
                        "Request timeout, retrying",
                        model=model_slug,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    # Non-retryable error
                    raise

        raise last_error or Exception("Max retries exceeded")

    def _parse_response(
        self,
        response,
        prompt_id: str,
        replay_id: str,
        model: ModelConfig,
        latency_ms: float,
        temperature: float,
    ) -> ReplayResult:
        """Parse OpenAI SDK response into ReplayResult."""
        choice = response.choices[0] if response.choices else None
        
        if choice:
            completion = choice.message.content or ""
            finish_reason = choice.finish_reason or "stop"
        else:
            completion = ""
            finish_reason = "error"

        # Get usage info
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else input_tokens + output_tokens

        # Calculate cost
        cost = model.pricing.calculate_cost(input_tokens, output_tokens)

        # Check for refusal
        refused = self._detect_refusal(completion, finish_reason)

        return ReplayResult(
            prompt_id=prompt_id,
            replay_id=replay_id,
            model_id=model.model_id,
            provider=model.provider.value,
            completion=completion,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            success=True,
            refused=refused,
            temperature=temperature,
            api_version=response.model,
        )

    def _detect_refusal(self, completion: str, finish_reason: str) -> bool:
        """Detect if the model refused to respond."""
        if finish_reason == "content_filter":
            return True

        refusal_patterns = [
            "I cannot",
            "I'm not able to",
            "I am not able to",
            "I won't",
            "I will not",
            "I'm unable to",
            "I am unable to",
            "against my guidelines",
            "violates my guidelines",
            "I must decline",
            "I have to decline",
        ]

        completion_lower = completion.lower()
        return any(pattern.lower() in completion_lower for pattern in refusal_patterns)

    async def close(self):
        """Close the HTTP client."""
        await self.client.close()


async def test_connection(api_key: str | None = None) -> bool:
    """
    Test Portkey connection with a simple request.
    
    Args:
        api_key: Portkey API key (uses settings if None)
        
    Returns:
        True if connection successful
    """
    settings = get_settings()
    key = api_key or settings.portkey_api_key

    client = AsyncOpenAI(
        base_url=PORTKEY_GATEWAY_URL,
        api_key=key,
    )

    try:
        response = await client.chat.completions.create(
            model="@openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_tokens=10,
        )
        logger.info("Connection test successful", model=response.model)
        return True
    except Exception as e:
        logger.error("Connection test failed", error=str(e))
        return False
    finally:
        await client.close()
