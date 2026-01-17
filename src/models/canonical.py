"""Canonical data format for prompt-completion data."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message role in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """Single message in a conversation."""

    role: MessageRole
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ComplexityLevel(str, Enum):
    """Prompt complexity classification."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class TaskType(str, Enum):
    """Type of task the prompt represents."""

    SUMMARIZATION = "summarization"
    QA = "qa"
    CODE_GENERATION = "code_generation"
    CLASSIFICATION = "classification"
    CREATIVE_WRITING = "creative_writing"
    TRANSLATION = "translation"
    EXTRACTION = "extraction"
    REASONING = "reasoning"
    OTHER = "other"


class PromptMetadata(BaseModel):
    """Metadata associated with a prompt."""

    user_id: str | None = None
    task_type: TaskType = TaskType.OTHER
    complexity: ComplexityLevel = ComplexityLevel.MEDIUM
    language: str | None = None  # Programming language for code tasks
    domain: str | None = None  # e.g., "medical", "legal", "technical"
    tags: list[str] = Field(default_factory=list)
    custom: dict[str, Any] = Field(default_factory=dict)


class CompletionData(BaseModel):
    """Data about the original completion."""

    text: str
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    finish_reason: str = "stop"
    created_at: datetime


class GroundTruth(BaseModel):
    """Ground truth data for evaluation."""

    reference_output: str | None = None
    quality_score: float | None = None  # Human rating if available
    success_metric: bool | None = None  # Did it achieve the goal?
    user_feedback: str | None = None  # thumbs_up, thumbs_down, etc.


class CanonicalPrompt(BaseModel):
    """
    Canonical format for prompt-completion data.
    
    This is the normalized format that all adapters convert to.
    """

    # Identifiers
    id: str = Field(description="Unique identifier for this prompt")
    trace_id: str | None = None
    source: str = Field(description="Origin of this data (e.g., 'portkey', 'csv')")

    # Timestamps
    created_at: datetime
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    # Prompt content
    messages: list[Message] = Field(description="Conversation messages")
    system_prompt: str | None = None  # Extracted system prompt if separate

    # Original completion
    completion: CompletionData

    # Metadata
    metadata: PromptMetadata = Field(default_factory=PromptMetadata)

    # Ground truth (if available)
    ground_truth: GroundTruth | None = None
    
    # Natural language of content (auto-detected or specified)
    content_language: str = Field(
        default="en",
        description="ISO language code for prompt content (en, es, fr, de, etc.)"
    )

    @property
    def prompt_text(self) -> str:
        """Get the prompt as a single string (for simple cases)."""
        user_messages = [m for m in self.messages if m.role == MessageRole.USER]
        if user_messages:
            return user_messages[-1].content
        return ""

    @property
    def total_prompt_tokens(self) -> int:
        """Estimate total tokens in all messages."""
        # Rough estimate: 4 chars per token
        total_chars = sum(len(m.content) for m in self.messages)
        return total_chars // 4

    def to_openai_format(self) -> list[dict[str, str]]:
        """Convert messages to OpenAI API format."""
        return [{"role": m.role.value, "content": m.content} for m in self.messages]

    def to_anthropic_format(self) -> tuple[str | None, list[dict[str, str]]]:
        """Convert messages to Anthropic API format (system, messages)."""
        system = None
        messages = []
        for m in self.messages:
            if m.role == MessageRole.SYSTEM:
                system = m.content
            else:
                messages.append({"role": m.role.value, "content": m.content})
        return system, messages
