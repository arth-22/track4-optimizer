"""Portkey Log Export API adapter.

Supports two methods:
1. Direct logs API (/logs) - simpler, paginated
2. Export API (/logs/exports) - for large datasets

Also supports portkey-ai SDK if available.
"""

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

# Try to import portkey SDK
try:
    from portkey_ai import Portkey
    PORTKEY_SDK_AVAILABLE = True
except ImportError:
    PORTKEY_SDK_AVAILABLE = False

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
        workspace_id: str | None = None,
        base_url: str = "https://api.portkey.ai/v1",
    ):
        settings = get_settings()
        self.api_key = api_key or settings.portkey_api_key
        self.workspace_id = workspace_id or getattr(settings, 'portkey_workspace_id', None) or "ws-team-1-f28b06"
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-portkey-api-key": self.api_key,
                "x-portkey-workspace-id": self.workspace_id,
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
        """Fetch prompts from Portkey logs.
        
        Tries multiple methods in order:
        1. Direct logs API (simpler, paginated)
        2. Export API (for large datasets)
        """
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

        # Try direct logs API first (simpler)
        try:
            logs = await self._fetch_logs_direct(start_date, end_date, limit or 100)
            if logs:
                logger.info("Fetched via direct API", count=len(logs))
                return self._convert_logs_to_prompts(logs)
        except Exception as e:
            logger.warning("Direct logs API failed, trying export", error=str(e))

        # Fallback to export API
        try:
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

            return self._convert_logs_to_prompts(logs)

        except Exception as e:
            logger.error("Export API also failed", error=str(e))
            return []

    def _convert_logs_to_prompts(self, logs: list[dict]) -> list[CanonicalPrompt]:
        """Convert raw logs to canonical prompts."""
        prompts = []
        for log in logs:
            try:
                prompt = self._normalize_to_canonical(log)
                prompts.append(prompt)
            except Exception as e:
                logger.warning("Failed to parse log entry", error=str(e), log_id=log.get("id"))
                continue

        logger.info("Converted logs to prompts", count=len(prompts))
        return prompts

    async def _fetch_logs_direct(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch logs using direct /logs endpoint with pagination."""
        all_logs = []
        page = 1
        page_size = min(limit, 100)  # API max per page
        
        while len(all_logs) < limit:
            params = {
                "page": page,
                "page_size": page_size,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            }
            
            response = await self.client.get("/logs", params=params)
            response.raise_for_status()
            data = response.json()
            
            logs = data.get("data", data.get("logs", []))
            if not logs:
                break
                
            all_logs.extend(logs)
            
            # Check if more pages
            has_more = data.get("has_more", len(logs) == page_size)
            if not has_more:
                break
                
            page += 1
        
        return all_logs[:limit]

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
        """Create a log export request using Portkey SDK pattern."""
        # Use Portkey SDK if available for more reliable API calls
        if PORTKEY_SDK_AVAILABLE:
            return await self._create_export_via_sdk(start_date, end_date)
        
        # Fallback to httpx
        payload = {
            "description": f"Track 4 Cost-Quality Analysis Export {datetime.utcnow().isoformat()}",
            "filters": {
                "time_of_generation_min": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "time_of_generation_max": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
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
                "config",
                "prompt_slug",
                "metadata",
            ],
        }

        response = await self.client.post("/logs/exports", json=payload, timeout=120.0)
        response.raise_for_status()
        return response.json()

    async def _create_export_via_sdk(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """Create export using Portkey SDK (more reliable)."""
        from portkey_ai import Portkey
        import asyncio
        
        def _create():
            portkey = Portkey(api_key=self.api_key)
            export = portkey.logs.exports.create(
                filters={
                    "time_of_generation_min": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "time_of_generation_max": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
                description=f"Track 4 Export {datetime.utcnow().isoformat()}",
                requested_data=[
                    "id", "trace_id", "created_at", "request", "response",
                    "ai_org", "ai_model", "req_units", "res_units", "total_units",
                    "cost", "cost_currency", "response_time", "response_status_code",
                    "config", "prompt_slug", "metadata",
                ],
            )
            # Convert to dict
            if hasattr(export, 'model_dump'):
                return export.model_dump()
            elif hasattr(export, 'dict'):
                return export.dict()
            return dict(export)
        
        # Run sync SDK call in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _create)

    async def _download_export(self, export_id: str) -> list[dict[str, Any]]:
        """Wait for export to complete, start it, and download results."""
        import asyncio
        
        # CRITICAL: Must call start() after create()!
        await self._start_export(export_id)
        
        # Poll for completion
        max_attempts = 90  # 3 minutes max
        for attempt in range(max_attempts):
            status_data = await self._get_export_status(export_id)
            status = status_data.get("status", "")
            
            logger.debug("Export status", attempt=attempt, status=status, export_id=export_id)
            
            if status in ["completed", "success"]:
                logger.info("Export completed", export_id=export_id)
                break
            elif status in ["failed", "error"]:
                logger.error("Export failed", export_id=export_id, status=status_data)
                return []
                
            await asyncio.sleep(2)
        else:
            logger.error("Export timed out", export_id=export_id)
            return []

        # Download export data - get signed URL first
        return await self._download_export_data(export_id)

    async def _start_export(self, export_id: str) -> None:
        """Start the export job (required after create)."""
        try:
            if PORTKEY_SDK_AVAILABLE:
                await self._start_export_via_sdk(export_id)
            else:
                response = await self.client.post(f"/logs/exports/{export_id}/start", timeout=30.0)
                response.raise_for_status()
            logger.info("Export started", export_id=export_id)
        except Exception as e:
            logger.warning("Failed to start export (may already be running)", error=str(e))

    async def _start_export_via_sdk(self, export_id: str) -> None:
        """Start export using SDK."""
        from portkey_ai import Portkey
        import asyncio
        
        def _start():
            portkey = Portkey(api_key=self.api_key)
            portkey.logs.exports.start(export_id=export_id)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _start)

    async def _get_export_status(self, export_id: str) -> dict[str, Any]:
        """Get export status."""
        if PORTKEY_SDK_AVAILABLE:
            return await self._get_export_status_via_sdk(export_id)
        
        response = await self.client.get(f"/logs/exports/{export_id}", timeout=30.0)
        response.raise_for_status()
        return response.json()

    async def _get_export_status_via_sdk(self, export_id: str) -> dict[str, Any]:
        """Get export status using SDK."""
        from portkey_ai import Portkey
        import asyncio
        
        def _retrieve():
            portkey = Portkey(api_key=self.api_key)
            result = portkey.logs.exports.retrieve(export_id=export_id)
            if hasattr(result, 'model_dump'):
                return result.model_dump()
            elif hasattr(result, 'dict'):
                return result.dict()
            return dict(result)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _retrieve)

    async def _download_export_data(self, export_id: str) -> list[dict[str, Any]]:
        """Download export data via signed URL."""
        import json as json_module
        
        if PORTKEY_SDK_AVAILABLE:
            return await self._download_export_data_via_sdk(export_id)
        
        # Get download info (signed URL)
        response = await self.client.get(f"/logs/exports/{export_id}/download", timeout=60.0)
        response.raise_for_status()
        download_info = response.json()
        
        signed_url = download_info.get("signed_url") or download_info.get("url")
        
        if signed_url:
            # Download from signed URL
            data_response = await self.client.get(signed_url, timeout=120.0)
            content = data_response.text
            
            # Parse NDJSON (newline-delimited JSON)
            return self._parse_ndjson(content)
        else:
            # Data might be directly in response
            return download_info.get("data", [])

    async def _download_export_data_via_sdk(self, export_id: str) -> list[dict[str, Any]]:
        """Download export data using SDK."""
        from portkey_ai import Portkey
        import asyncio
        import requests
        import json as json_module
        
        def _download():
            portkey = Portkey(api_key=self.api_key)
            result = portkey.logs.exports.download(export_id=export_id)
            
            if hasattr(result, 'model_dump'):
                download_data = result.model_dump()
            elif hasattr(result, 'dict'):
                download_data = result.dict()
            else:
                download_data = dict(result)
            
            signed_url = download_data.get("signed_url") or download_data.get("url")
            
            if signed_url:
                # Download from signed URL
                resp = requests.get(signed_url, timeout=120)
                return resp.text
            
            return download_data.get("data", [])
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _download)
        
        if isinstance(result, str):
            return self._parse_ndjson(result)
        return result

    def _parse_ndjson(self, content: str) -> list[dict[str, Any]]:
        """Parse newline-delimited JSON (NDJSON) format."""
        import json as json_module
        
        records = []
        for line in content.strip().split('\n'):
            if line.strip():
                try:
                    records.append(json_module.loads(line))
                except json_module.JSONDecodeError as e:
                    logger.warning("Failed to parse NDJSON line", error=str(e))
                    continue
        
        logger.info("Parsed NDJSON records", count=len(records))
        return records

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
            
            # Handle different content formats
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle Anthropic-style list format: [{"type": "text", "text": "..."}]
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        text_parts.append(part.get("text", str(part)))
                    else:
                        text_parts.append(str(part))
                content = " ".join(text_parts)
            elif not isinstance(content, str):
                content = str(content)
            
            messages.append(Message(role=role, content=content))

        # Parse response
        response = log.get("response", {})
        choices = response.get("choices", [{}])
        completion_text = ""
        if choices:
            msg_content = choices[0].get("message", {}).get("content", "")
            # Handle list-format content in response too
            if isinstance(msg_content, list):
                text_parts = []
                for part in msg_content:
                    if isinstance(part, dict):
                        text_parts.append(part.get("text", str(part)))
                    else:
                        text_parts.append(str(part))
                completion_text = " ".join(text_parts)
            else:
                completion_text = msg_content or ""

        # Extract model and provider
        model_id = log.get("ai_model", "unknown")
        provider = log.get("ai_org", "unknown")

        # Parse timestamps - handle multiple formats
        created_at = self._parse_timestamp(log.get("created_at"))

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

    def _parse_timestamp(self, timestamp_str: str | None) -> datetime:
        """Parse timestamp from various formats."""
        if not timestamp_str:
            return datetime.utcnow()
        
        # Try ISO format first (most common)
        try:
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pass
        
        # Try JS Date format: "Sat Jan 17 2026 09:22:04 GMT+0000 (Coordinated Universal Time)"
        try:
            # Remove timezone name in parentheses
            if "(" in timestamp_str:
                timestamp_str = timestamp_str.split("(")[0].strip()
            
            # Parse the date part
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(timestamp_str)
        except Exception:
            pass
        
        # Try common formats
        formats = [
            "%a %b %d %Y %H:%M:%S GMT%z",  # JS Date with GMT offset
            "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO with milliseconds
            "%Y-%m-%dT%H:%M:%SZ",  # ISO without milliseconds
            "%Y-%m-%d %H:%M:%S",  # Simple datetime
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        # Fallback to current time
        logger.warning("Could not parse timestamp, using current time", timestamp=timestamp_str)
        return datetime.utcnow()

    def close(self):
        """Close the HTTP client (sync wrapper for async close)."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.client.aclose())
        except RuntimeError:
            # No running loop, run synchronously
            asyncio.run(self.client.aclose())

    async def aclose(self):
        """Close the HTTP client (async)."""
        await self.client.aclose()
