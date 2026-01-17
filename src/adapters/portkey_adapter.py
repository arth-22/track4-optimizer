"""Portkey Log Export API adapter."""

import uuid
from datetime import datetime, timedelta
from typing import Any

import httpx
import structlog

from src.adapters.base import BaseAdapter
from src.config import get_settings
from src.models.canonical import (
    CanonicalPrompt,
    CompletionData,
    Message,
    MessageRole,
    PromptMetadata,
    TaskType,
    ComplexityLevel,
)

logger = structlog.get_logger()


class PortkeyLogAdapter(BaseAdapter):
    """
    Adapter for fetching historical prompt-completion data from Portkey.
    
    Uses Portkey's Log Export API to retrieve logs with:
    - Request/response data
    - Token usage
    - Cost information
    - Latency metrics
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.portkey.ai/v1",
    ):
        settings = get_settings()
        self.api_key = api_key or settings.portkey_api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-portkey-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    @property
    def source_name(self) -> str:
        return "portkey"

    async def fetch_prompts(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int | None = None,
    ) -> list[CanonicalPrompt]:
        """Fetch prompts from Portkey logs."""
        # Default to last 30 days if no date range specified
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        logger.info(
            "Fetching Portkey logs",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            limit=limit,
        )

        # Create export request
        export = await self._create_export(start_date, end_date)
        export_id = export.get("id")

        if not export_id:
            logger.error("Failed to create export", response=export)
            return []

        # Wait for export to complete and download
        logs = await self._download_export(export_id)

        # Apply limit
        if limit and len(logs) > limit:
            logs = logs[:limit]

        # Convert to canonical format
        prompts = []
        for log in logs:
            try:
                prompt = self._normalize_to_canonical(log)
                prompts.append(prompt)
            except Exception as e:
                logger.warning("Failed to parse log entry", error=str(e), log_id=log.get("id"))
                continue

        logger.info("Fetched prompts from Portkey", count=len(prompts))
        return prompts

    async def count_available(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> int:
        """Count available logs without fetching full data."""
        # For Portkey, we need to create an export to count
        # This is a limitation of the API
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        try:
            export = await self._create_export(start_date, end_date)
            # Return estimated count from export metadata if available
            return export.get("estimated_count", 0)
        except Exception:
            return 0

    async def _create_export(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """Create a log export request."""
        payload = {
            "filters": {
                "time_of_generation_min": start_date.isoformat(),
                "time_of_generation_max": end_date.isoformat(),
            },
            "requested_data": [
                "id",
                "trace_id",
                "created_at",
                "request",
                "response",
                "ai_org",
                "ai_model",
                "req_units",
                "res_units",
                "total_units",
                "cost",
                "cost_currency",
                "response_time",
                "response_status_code",
                "is_success",
                "metadata",
                "prompt_slug",
            ],
        }

        response = await self.client.post("/logs/exports", json=payload)
        response.raise_for_status()
        return response.json()

    async def _download_export(self, export_id: str) -> list[dict[str, Any]]:
        """Wait for export to complete and download results."""
        import asyncio

        # Poll for completion
        max_attempts = 60
        for _ in range(max_attempts):
            status = await self._get_export_status(export_id)
            if status.get("status") == "completed":
                break
            elif status.get("status") == "failed":
                logger.error("Export failed", export_id=export_id, status=status)
                return []
            await asyncio.sleep(2)
        else:
            logger.error("Export timed out", export_id=export_id)
            return []

        # Download export data
        response = await self.client.get(f"/logs/exports/{export_id}/download")
        response.raise_for_status()
        return response.json().get("data", [])

    async def _get_export_status(self, export_id: str) -> dict[str, Any]:
        """Get export status."""
        response = await self.client.get(f"/logs/exports/{export_id}")
        response.raise_for_status()
        return response.json()

    def _normalize_to_canonical(self, log: dict[str, Any]) -> CanonicalPrompt:
        """Convert Portkey log to canonical format."""
        # Parse request to extract messages
        request = log.get("request", {})
        messages_raw = request.get("messages", [])
        
        messages = []
        for msg in messages_raw:
            role_str = msg.get("role", "user")
            try:
                role = MessageRole(role_str)
            except ValueError:
                role = MessageRole.USER
            messages.append(Message(role=role, content=msg.get("content", "")))

        # Parse response
        response = log.get("response", {})
        choices = response.get("choices", [{}])
        completion_text = ""
        if choices:
            completion_text = choices[0].get("message", {}).get("content", "")

        # Extract model and provider
        model_id = log.get("ai_model", "unknown")
        provider = log.get("ai_org", "unknown")

        # Parse timestamps
        created_at_str = log.get("created_at")
        if created_at_str:
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        else:
            created_at = datetime.utcnow()

        # Calculate cost
        cost = log.get("cost", 0.0)
        if isinstance(cost, str):
            try:
                cost = float(cost)
            except ValueError:
                cost = 0.0

        # Build completion data
        completion = CompletionData(
            text=completion_text,
            model_id=model_id,
            provider=provider,
            input_tokens=log.get("req_units", 0),
            output_tokens=log.get("res_units", 0),
            total_tokens=log.get("total_units", 0),
            latency_ms=log.get("response_time", 0.0),
            cost_usd=cost,
            finish_reason=choices[0].get("finish_reason", "stop") if choices else "stop",
            created_at=created_at,
        )

        # Build metadata
        metadata = PromptMetadata(
            task_type=self._infer_task_type(messages),
            complexity=self._estimate_complexity(messages),
            custom=log.get("metadata", {}),
        )

        return CanonicalPrompt(
            id=log.get("id", str(uuid.uuid4())),
            trace_id=log.get("trace_id"),
            source="portkey",
            created_at=created_at,
            messages=messages,
            completion=completion,
            metadata=metadata,
        )

    def _infer_task_type(self, messages: list[Message]) -> TaskType:
        """Infer task type from message content."""
        all_content = " ".join(m.content.lower() for m in messages)
        
        if any(kw in all_content for kw in ["summarize", "summary", "brief"]):
            return TaskType.SUMMARIZATION
        elif any(kw in all_content for kw in ["code", "function", "implement", "write a program"]):
            return TaskType.CODE_GENERATION
        elif any(kw in all_content for kw in ["translate", "translation"]):
            return TaskType.TRANSLATION
        elif any(kw in all_content for kw in ["classify", "categorize", "label"]):
            return TaskType.CLASSIFICATION
        elif any(kw in all_content for kw in ["extract", "find", "identify"]):
            return TaskType.EXTRACTION
        elif "?" in all_content:
            return TaskType.QA
        else:
            return TaskType.OTHER

    def _estimate_complexity(self, messages: list[Message]) -> ComplexityLevel:
        """Estimate prompt complexity based on heuristics."""
        total_length = sum(len(m.content) for m in messages)
        num_messages = len(messages)

        # Simple heuristics
        if total_length < 500 and num_messages <= 2:
            return ComplexityLevel.SIMPLE
        elif total_length > 3000 or num_messages > 5:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.MEDIUM

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
