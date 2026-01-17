"""Pytest configuration and fixtures."""

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from src.models.canonical import (
    CanonicalPrompt,
    CompletionData,
    Message,
    MessageRole,
    PromptMetadata,
    ComplexityLevel,
    TaskType,
)


@pytest.fixture
def sample_messages():
    """Sample conversation messages."""
    return [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Explain quantum computing in simple terms."),
    ]


@pytest.fixture
def sample_prompt(sample_messages):
    """Sample canonical prompt for testing."""
    return CanonicalPrompt(
        id="test-prompt-001",
        trace_id="trace-001",
        source="test",
        created_at=datetime.utcnow(),
        messages=sample_messages,
        completion=CompletionData(
            text="Quantum computing uses quantum bits or qubits...",
            model_id="gpt-4o",
            provider="openai",
            input_tokens=50,
            output_tokens=150,
            total_tokens=200,
            latency_ms=1500.0,
            cost_usd=0.001,
            finish_reason="stop",
            created_at=datetime.utcnow(),
        ),
        metadata=PromptMetadata(
            task_type=TaskType.QA,
            complexity=ComplexityLevel.MEDIUM,
        ),
    )


@pytest.fixture
def sample_prompts_batch():
    """Batch of sample prompts for testing."""
    prompts = []
    
    scenarios = [
        ("simple", "What is 2+2?", "4", ComplexityLevel.SIMPLE, TaskType.QA),
        ("medium", "Explain how neural networks learn.", "Neural networks learn through...", ComplexityLevel.MEDIUM, TaskType.QA),
        ("complex", "Write a Python function to implement quicksort with detailed comments.", "def quicksort(arr):\n    ...", ComplexityLevel.COMPLEX, TaskType.CODE_GENERATION),
    ]
    
    for i, (name, prompt_text, completion_text, complexity, task_type) in enumerate(scenarios):
        prompts.append(CanonicalPrompt(
            id=f"test-{name}-{i:03d}",
            source="test",
            created_at=datetime.utcnow(),
            messages=[
                Message(role=MessageRole.USER, content=prompt_text),
            ],
            completion=CompletionData(
                text=completion_text,
                model_id="gpt-4o",
                provider="openai",
                input_tokens=len(prompt_text) // 4,
                output_tokens=len(completion_text) // 4,
                total_tokens=(len(prompt_text) + len(completion_text)) // 4,
                latency_ms=500 + i * 500,
                cost_usd=0.0005 * (i + 1),
                created_at=datetime.utcnow(),
            ),
            metadata=PromptMetadata(
                task_type=task_type,
                complexity=complexity,
            ),
        ))
    
    return prompts


@pytest.fixture
def sample_json_data(tmp_path):
    """Create a sample JSON data file for testing."""
    data = [
        {
            "id": "json-001",
            "prompt": "Summarize this article about AI.",
            "completion": "The article discusses recent advances in AI...",
            "model": "gpt-4o",
            "input_tokens": 100,
            "output_tokens": 200,
            "cost": 0.002,
            "latency_ms": 1000,
            "task_type": "summarization",
            "complexity": "medium",
        },
        {
            "id": "json-002",
            "prompt": "Translate 'Hello world' to French.",
            "completion": "Bonjour le monde",
            "model": "gpt-4o-mini",
            "input_tokens": 20,
            "output_tokens": 10,
            "cost": 0.0001,
            "latency_ms": 300,
            "task_type": "translation",
            "complexity": "simple",
        },
    ]
    
    file_path = tmp_path / "test_data.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    
    return file_path


@pytest.fixture
def mock_portkey_api_key():
    """Mock Portkey API key for testing."""
    return os.getenv("PORTKEY_API_KEY", "test-api-key")
