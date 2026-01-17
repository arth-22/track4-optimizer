"""CSV/JSON file adapter for data ingestion."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src.adapters.base import BaseAdapter
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


class CSVAdapter(BaseAdapter):
    """
    Adapter for loading prompt-completion data from CSV or JSON files.
    
    Supports formats:
    - CSV with columns: prompt, completion, model, tokens, cost, etc.
    - JSON array of prompt-completion objects
    - JSONL (one JSON object per line)
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self._cached_data: list[dict[str, Any]] | None = None

    @property
    def source_name(self) -> str:
        return f"file:{self.file_path.name}"

    async def fetch_prompts(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int | None = None,
    ) -> list[CanonicalPrompt]:
        """Load prompts from file."""
        data = self._load_data()

        # Filter by date if applicable
        if start_date or end_date:
            data = self._filter_by_date(data, start_date, end_date)

        # Apply limit
        if limit and len(data) > limit:
            data = data[:limit]

        # Convert to canonical format
        prompts = []
        for i, record in enumerate(data):
            try:
                prompt = self._normalize_to_canonical(record, index=i)
                prompts.append(prompt)
            except Exception as e:
                logger.warning("Failed to parse record", index=i, error=str(e))
                continue

        logger.info("Loaded prompts from file", path=str(self.file_path), count=len(prompts))
        return prompts

    async def count_available(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> int:
        """Count records in file."""
        data = self._load_data()
        if start_date or end_date:
            data = self._filter_by_date(data, start_date, end_date)
        return len(data)

    def _load_data(self) -> list[dict[str, Any]]:
        """Load and cache data from file."""
        if self._cached_data is not None:
            return self._cached_data

        suffix = self.file_path.suffix.lower()

        if suffix == ".csv":
            self._cached_data = self._load_csv()
        elif suffix == ".json":
            self._cached_data = self._load_json()
        elif suffix == ".jsonl":
            self._cached_data = self._load_jsonl()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        return self._cached_data

    def _load_csv(self) -> list[dict[str, Any]]:
        """Load data from CSV file."""
        import csv

        data = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        return data

    def _load_json(self) -> list[dict[str, Any]]:
        """Load data from JSON file."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "data" in data:
            return data["data"]
        else:
            return [data]

    def _load_jsonl(self) -> list[dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _filter_by_date(
        self,
        data: list[dict[str, Any]],
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> list[dict[str, Any]]:
        """Filter records by date range."""
        filtered = []
        for record in data:
            timestamp_str = record.get("created_at") or record.get("timestamp")
            if not timestamp_str:
                filtered.append(record)  # Include if no timestamp
                continue

            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if start_date and timestamp < start_date:
                    continue
                if end_date and timestamp > end_date:
                    continue
                filtered.append(record)
            except (ValueError, AttributeError):
                filtered.append(record)  # Include if can't parse

        return filtered

    def _normalize_to_canonical(
        self, record: dict[str, Any], index: int = 0
    ) -> CanonicalPrompt:
        """Convert file record to canonical format."""
        # Extract prompt/messages
        messages = []
        
        # Check for messages array first
        if "messages" in record:
            for msg in record["messages"]:
                role_str = msg.get("role", "user")
                try:
                    role = MessageRole(role_str)
                except ValueError:
                    role = MessageRole.USER
                messages.append(Message(role=role, content=msg.get("content", "")))
        else:
            # Fall back to single prompt field
            prompt_text = record.get("prompt") or record.get("input") or record.get("question", "")
            if prompt_text:
                messages.append(Message(role=MessageRole.USER, content=prompt_text))

            # Add system prompt if present
            system_prompt = record.get("system_prompt") or record.get("system")
            if system_prompt:
                messages.insert(0, Message(role=MessageRole.SYSTEM, content=system_prompt))

        # Extract completion
        completion_text = (
            record.get("completion")
            or record.get("response")
            or record.get("output")
            or record.get("answer")
            or ""
        )

        # Parse timestamp
        timestamp_str = record.get("created_at") or record.get("timestamp")
        if timestamp_str:
            try:
                created_at = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = datetime.utcnow()
        else:
            created_at = datetime.utcnow()

        # Extract model info
        model_id = record.get("model") or record.get("model_id") or "unknown"
        provider = record.get("provider") or self._infer_provider(model_id)

        # Extract token counts
        input_tokens = int(record.get("input_tokens") or record.get("prompt_tokens") or 0)
        output_tokens = int(record.get("output_tokens") or record.get("completion_tokens") or 0)
        total_tokens = int(record.get("total_tokens") or (input_tokens + output_tokens))

        # Extract cost
        cost = float(record.get("cost") or record.get("cost_usd") or 0.0)

        # Extract latency
        latency = float(record.get("latency_ms") or record.get("response_time") or 0.0)

        # Build completion data
        completion = CompletionData(
            text=completion_text,
            model_id=model_id,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency,
            cost_usd=cost,
            finish_reason=record.get("finish_reason", "stop"),
            created_at=created_at,
        )

        # Extract metadata
        task_type_str = record.get("task_type", "other")
        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            task_type = TaskType.OTHER

        complexity_str = record.get("complexity", "medium")
        try:
            complexity = ComplexityLevel(complexity_str)
        except ValueError:
            complexity = ComplexityLevel.MEDIUM

        metadata = PromptMetadata(
            user_id=record.get("user_id"),
            task_type=task_type,
            complexity=complexity,
            language=record.get("language"),
            domain=record.get("domain"),
            tags=record.get("tags", []),
            custom={k: v for k, v in record.items() if k not in self._known_fields()},
        )

        return CanonicalPrompt(
            id=record.get("id") or str(uuid.uuid4()),
            trace_id=record.get("trace_id"),
            source=self.source_name,
            created_at=created_at,
            messages=messages,
            completion=completion,
            metadata=metadata,
        )

    def _infer_provider(self, model_id: str) -> str:
        """Infer provider from model ID."""
        model_lower = model_id.lower()
        if "gpt" in model_lower or "o1" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "google"
        elif "llama" in model_lower or "mistral" in model_lower:
            return "meta"
        else:
            return "unknown"

    def _known_fields(self) -> set[str]:
        """Return set of known/parsed fields."""
        return {
            "id", "trace_id", "created_at", "timestamp",
            "prompt", "input", "question", "messages", "system_prompt", "system",
            "completion", "response", "output", "answer",
            "model", "model_id", "provider",
            "input_tokens", "prompt_tokens", "output_tokens", "completion_tokens", "total_tokens",
            "cost", "cost_usd", "latency_ms", "response_time",
            "finish_reason", "task_type", "complexity", "user_id", "language", "domain", "tags",
        }
