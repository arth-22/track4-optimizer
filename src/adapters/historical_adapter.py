"""Historical data adapter for demo mode.

Loads pre-created prompt data from JSON files for hackathon demos
and testing without requiring live Portkey API calls.
"""

import json
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


class HistoricalAdapter(BaseAdapter):
    """
    Adapter for loading demo data from historical_prompts.json.
    
    This enables end-to-end demo without fetching from Portkey API.
    The prompts will still go through ReplayEngine and CompositeEvaluator.
    
    Usage:
        adapter = HistoricalAdapter("docs/historical_prompts.json")
        prompts = await adapter.fetch_prompts(limit=12)
        # Returns list[CanonicalPrompt] for use with ReplayEngine
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self._data: dict | None = None
        self._references: dict[str, str] = {}  # prompt_id -> original response

    @property
    def source_name(self) -> str:
        return "historical_demo"

    async def fetch_prompts(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int | None = None,
    ) -> list[CanonicalPrompt]:
        """
        Load prompts from historical JSON file.
        
        Args:
            start_date: Ignored (all demo data returned)
            end_date: Ignored (all demo data returned)
            limit: Maximum number of prompts to return
            
        Returns:
            List of CanonicalPrompt objects ready for ReplayEngine
        """
        self._load_data()
        
        prompts_data = self._data.get("prompts", [])
        if limit:
            prompts_data = prompts_data[:limit]
        
        logger.info(
            "Loading historical prompts",
            file=str(self.file_path),
            count=len(prompts_data),
        )
        
        prompts = []
        for record in prompts_data:
            try:
                prompt = self._convert_to_canonical(record)
                prompts.append(prompt)
                
                # Store original response as reference for BERTScore
                if "original_response" in record:
                    self._references[prompt.id] = record["original_response"]["content"]
            except Exception as e:
                logger.warning(
                    "Failed to convert record",
                    record_id=record.get("id"),
                    error=str(e),
                )
        
        logger.info(
            "Loaded historical prompts",
            count=len(prompts),
            references=len(self._references),
        )
        
        return prompts

    async def count_available(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> int:
        """Count prompts in file."""
        self._load_data()
        return len(self._data.get("prompts", []))

    def _load_data(self) -> None:
        """Load and cache JSON data from file."""
        if self._data is not None:
            return
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Historical data file not found: {self.file_path}")
        
        with open(self.file_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        
        logger.debug(
            "Loaded historical data file",
            file=str(self.file_path),
            prompt_count=len(self._data.get("prompts", [])),
        )

    def _convert_to_canonical(self, record: dict[str, Any]) -> CanonicalPrompt:
        """Convert historical record to CanonicalPrompt format."""
        # Parse messages
        messages = []
        for msg in record.get("messages", []):
            role = MessageRole(msg["role"])
            messages.append(Message(role=role, content=msg["content"]))
        
        # Parse completion data from original_response
        orig = record.get("original_response", {})
        completion = CompletionData(
            text=orig.get("content", ""),
            model_id=orig.get("model", "unknown"),
            provider=orig.get("provider", "unknown"),
            input_tokens=orig.get("input_tokens", 0),
            output_tokens=orig.get("output_tokens", 0),
            total_tokens=orig.get("total_tokens", 0),
            latency_ms=orig.get("latency_ms", 0),
            cost_usd=orig.get("cost_usd", 0),
            finish_reason=orig.get("finish_reason", "stop"),
            created_at=self._parse_timestamp(record.get("created_at", "")),
        )
        
        # Parse metadata
        meta = record.get("metadata", {})
        metadata = PromptMetadata(
            task_type=self._parse_task_type(meta.get("task_type", "other")),
            complexity=self._parse_complexity(meta.get("complexity", "medium")),
            domain=meta.get("domain"),
        )
        
        return CanonicalPrompt(
            id=record["id"],
            trace_id=record.get("trace_id"),
            source=self.source_name,
            created_at=self._parse_timestamp(record.get("created_at", "")),
            messages=messages,
            completion=completion,
            metadata=metadata,
        )

    def _parse_timestamp(self, ts: str) -> datetime:
        """Parse timestamp from various formats."""
        if not ts:
            return datetime.utcnow()
        
        # Try ISO format
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            pass
        
        # Try common formats
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
        
        return datetime.utcnow()

    def _parse_task_type(self, task_type: str) -> TaskType:
        """Parse task type string to enum."""
        task_map = {
            "summarization": TaskType.SUMMARIZATION,
            "qa": TaskType.QA,
            "code_generation": TaskType.CODE_GENERATION,
            "classification": TaskType.CLASSIFICATION,
            "creative_writing": TaskType.CREATIVE_WRITING,
            "translation": TaskType.TRANSLATION,
            "extraction": TaskType.EXTRACTION,
            "reasoning": TaskType.REASONING,
            "other": TaskType.OTHER,
        }
        return task_map.get(task_type.lower(), TaskType.OTHER)

    def _parse_complexity(self, complexity: str) -> ComplexityLevel:
        """Parse complexity string to enum."""
        complexity_map = {
            "simple": ComplexityLevel.SIMPLE,
            "medium": ComplexityLevel.MEDIUM,
            "complex": ComplexityLevel.COMPLEX,
        }
        return complexity_map.get(complexity.lower(), ComplexityLevel.MEDIUM)

    def get_reference(self, prompt_id: str) -> str | None:
        """
        Get original response for a prompt (used as BERTScore reference).
        
        Args:
            prompt_id: The prompt ID to get reference for
            
        Returns:
            Original response text if available, None otherwise
        """
        return self._references.get(prompt_id)

    def get_all_references(self) -> dict[str, str]:
        """
        Get all original responses.
        
        Returns:
            Dict mapping prompt_id to original response text
        """
        return self._references.copy()

    async def close(self) -> None:
        """Clean up resources."""
        self._data = None
        self._references = {}
