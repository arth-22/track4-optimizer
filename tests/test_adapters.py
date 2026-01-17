"""Tests for data adapters."""

import pytest

from src.adapters.csv_adapter import CSVAdapter
from src.models.canonical import ComplexityLevel, TaskType


class TestCSVAdapter:
    """Tests for CSV/JSON adapter."""

    @pytest.mark.asyncio
    async def test_load_json_file(self, sample_json_data):
        """Test loading data from JSON file."""
        adapter = CSVAdapter(sample_json_data)
        prompts = await adapter.fetch_prompts()

        assert len(prompts) == 2
        assert prompts[0].id == "json-001"
        assert prompts[1].id == "json-002"

    @pytest.mark.asyncio
    async def test_count_available(self, sample_json_data):
        """Test counting records in file."""
        adapter = CSVAdapter(sample_json_data)
        count = await adapter.count_available()

        assert count == 2

    @pytest.mark.asyncio
    async def test_limit_prompts(self, sample_json_data):
        """Test limiting fetched prompts."""
        adapter = CSVAdapter(sample_json_data)
        prompts = await adapter.fetch_prompts(limit=1)

        assert len(prompts) == 1

    @pytest.mark.asyncio
    async def test_source_name(self, sample_json_data):
        """Test source name format."""
        adapter = CSVAdapter(sample_json_data)
        
        assert "file:" in adapter.source_name
        assert "test_data.json" in adapter.source_name

    @pytest.mark.asyncio
    async def test_metadata_parsing(self, sample_json_data):
        """Test metadata is correctly parsed."""
        adapter = CSVAdapter(sample_json_data)
        prompts = await adapter.fetch_prompts()

        assert prompts[0].metadata.task_type == TaskType.SUMMARIZATION
        assert prompts[0].metadata.complexity == ComplexityLevel.MEDIUM
        assert prompts[1].metadata.task_type == TaskType.TRANSLATION
        assert prompts[1].metadata.complexity == ComplexityLevel.SIMPLE

    @pytest.mark.asyncio
    async def test_provider_inference(self, sample_json_data):
        """Test provider is inferred from model ID."""
        adapter = CSVAdapter(sample_json_data)
        prompts = await adapter.fetch_prompts()

        assert prompts[0].completion.provider == "openai"
        assert prompts[1].completion.provider == "openai"
