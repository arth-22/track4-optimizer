"""Provider-aware rate limiter using leaky bucket algorithm."""

import asyncio
from typing import Dict

from aiolimiter import AsyncLimiter
import structlog

from src.replay.model_registry import Provider

logger = structlog.get_logger()


class RateLimiter:
    """
    Rate limiter with provider-specific limits.
    
    Uses leaky bucket algorithm via aiolimiter for smooth rate control.
    """

    # Default rate limits per provider (requests per minute)
    DEFAULT_LIMITS: Dict[Provider, tuple[int, int]] = {
        Provider.OPENAI: (500, 60),      # 500 requests per 60 seconds
        Provider.ANTHROPIC: (300, 60),   # 300 requests per 60 seconds
        Provider.VERTEX: (600, 60),      # 600 requests per 60 seconds (Google via Vertex)
        Provider.META: (100, 60),        # 100 requests per 60 seconds
        Provider.MISTRAL: (200, 60),     # 200 requests per 60 seconds
        Provider.LOCAL: (1000, 60),      # 1000 requests per 60 seconds (local)
    }

    def __init__(
        self,
        custom_limits: Dict[Provider, tuple[int, int]] | None = None,
        global_limit: int | None = None,
    ):
        """
        Initialize rate limiter.
        
        Args:
            custom_limits: Custom limits per provider (requests, time_period)
            global_limit: Global limit across all providers
        """
        self._limiters: Dict[Provider, AsyncLimiter] = {}
        self._global_limiter: AsyncLimiter | None = None

        # Set up provider-specific limiters
        limits = {**self.DEFAULT_LIMITS, **(custom_limits or {})}
        for provider, (max_rate, time_period) in limits.items():
            self._limiters[provider] = AsyncLimiter(max_rate, time_period)
            logger.debug(
                "Configured rate limit",
                provider=provider.value,
                max_rate=max_rate,
                time_period=time_period,
            )

        # Set up global limiter if specified
        if global_limit:
            self._global_limiter = AsyncLimiter(global_limit, 60)

    async def acquire(self, provider: Provider) -> None:
        """
        Acquire permission to make a request.
        
        Blocks until rate limit allows the request.
        
        Args:
            provider: The provider to acquire for
        """
        # Wait for provider-specific limit
        limiter = self._limiters.get(provider)
        if limiter:
            await limiter.acquire()

        # Wait for global limit if set
        if self._global_limiter:
            await self._global_limiter.acquire()

    async def acquire_multiple(self, provider: Provider, count: int = 1) -> None:
        """
        Acquire permission for multiple requests.
        
        Args:
            provider: The provider to acquire for
            count: Number of requests to acquire
        """
        for _ in range(count):
            await self.acquire(provider)

    def get_current_rate(self, provider: Provider) -> float:
        """Get current requests per second for a provider."""
        limiter = self._limiters.get(provider)
        if not limiter:
            return 0.0
        # aiolimiter doesn't expose current rate, so estimate
        return limiter.max_rate / limiter.time_period


class ConcurrencyLimiter:
    """
    Limits concurrent requests using semaphores.
    
    Use alongside RateLimiter for both rate and concurrency control.
    """

    DEFAULT_CONCURRENCY: Dict[Provider, int] = {
        Provider.OPENAI: 20,
        Provider.ANTHROPIC: 15,
        Provider.VERTEX: 20,  # Google via Vertex AI
        Provider.META: 10,
        Provider.MISTRAL: 10,
        Provider.LOCAL: 50,
    }

    def __init__(
        self,
        custom_limits: Dict[Provider, int] | None = None,
        global_limit: int = 50,
    ):
        self._semaphores: Dict[Provider, asyncio.Semaphore] = {}
        self._global_semaphore = asyncio.Semaphore(global_limit)

        limits = {**self.DEFAULT_CONCURRENCY, **(custom_limits or {})}
        for provider, limit in limits.items():
            self._semaphores[provider] = asyncio.Semaphore(limit)

    async def __aenter__(self) -> "ConcurrencyLimiter":
        return self

    async def __aexit__(self, *args) -> None:
        pass

    def get_semaphore(self, provider: Provider) -> asyncio.Semaphore:
        """Get semaphore for a provider."""
        return self._semaphores.get(provider) or asyncio.Semaphore(10)

    async def acquire(self, provider: Provider) -> tuple[asyncio.Semaphore, asyncio.Semaphore]:
        """Acquire both provider and global semaphores."""
        provider_sem = self.get_semaphore(provider)
        await provider_sem.acquire()
        await self._global_semaphore.acquire()
        return provider_sem, self._global_semaphore

    def release(
        self,
        provider_sem: asyncio.Semaphore,
        global_sem: asyncio.Semaphore,
    ) -> None:
        """Release both semaphores."""
        provider_sem.release()
        global_sem.release()
