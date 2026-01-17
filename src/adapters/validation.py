"""Data validation module for ingested prompts."""

import hashlib
import re
from collections import defaultdict
from typing import Any

import structlog

from src.models.canonical import CanonicalPrompt, Message

logger = structlog.get_logger()


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class PIIScrubber:
    """
    Scrubs personally identifiable information (PII) from text.
    
    Detects and masks:
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    - IP addresses
    """
    
    PATTERNS = {
        "EMAIL": r'\b[\w.-]+@[\w.-]+\.\w+\b',
        "PHONE": r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
        "SSN": r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        "CREDIT_CARD": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        "IP_ADDRESS": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    }
    
    @classmethod
    def scrub(cls, text: str, mask: bool = True) -> tuple[str, list[str]]:
        """
        Scrub PII from text.
        
        Args:
            text: Input text to scrub
            mask: If True, replace PII with [TYPE] masks. If False, just detect.
            
        Returns:
            Tuple of (scrubbed_text, list_of_found_pii_types)
        """
        found_types = []
        scrubbed = text
        
        for pii_type, pattern in cls.PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                found_types.append(pii_type)
                if mask:
                    scrubbed = re.sub(pattern, f"[{pii_type}]", scrubbed)
        
        return scrubbed, found_types
    
    @classmethod
    def detect(cls, text: str) -> list[str]:
        """Detect PII types in text without modifying."""
        _, found = cls.scrub(text, mask=False)
        return found
    
    @classmethod
    def has_pii(cls, text: str) -> bool:
        """Check if text contains any PII."""
        return len(cls.detect(text)) > 0


class ValidationResult:
    """Result of validation check."""
    
    def __init__(self):
        self.valid_prompts: list[CanonicalPrompt] = []
        self.invalid_prompts: list[tuple[CanonicalPrompt, str]] = []
        self.duplicates_removed: int = 0
        self.outliers_flagged: int = 0
        self.warnings: list[str] = []
    
    @property
    def total_processed(self) -> int:
        return len(self.valid_prompts) + len(self.invalid_prompts)
    
    @property
    def valid_count(self) -> int:
        return len(self.valid_prompts)
    
    @property
    def invalid_count(self) -> int:
        return len(self.invalid_prompts)
    
    def summary(self) -> dict[str, Any]:
        return {
            "total_processed": self.total_processed,
            "valid": self.valid_count,
            "invalid": self.invalid_count,
            "duplicates_removed": self.duplicates_removed,
            "outliers_flagged": self.outliers_flagged,
            "warnings": self.warnings,
        }


class DataValidator:
    """
    Validates and cleans ingested prompt data.
    
    Implements:
    - Completeness checking (required fields present)
    - Format validation (valid text content)
    - Deduplication (remove exact duplicates)
    - Outlier detection (flag suspiciously long/short content)
    """
    
    # Thresholds for outlier detection
    MIN_PROMPT_LENGTH = 5  # Minimum characters
    MAX_PROMPT_LENGTH = 500_000  # ~125K tokens
    MIN_COMPLETION_LENGTH = 1
    MAX_COMPLETION_LENGTH = 200_000  # ~50K tokens
    
    # Percentile-based outlier detection
    OUTLIER_LOW_PERCENTILE = 1  # Bottom 1%
    OUTLIER_HIGH_PERCENTILE = 99  # Top 1%
    
    def __init__(
        self,
        deduplicate: bool = True,
        flag_outliers: bool = True,
        strict_mode: bool = False,
    ):
        """
        Initialize validator.
        
        Args:
            deduplicate: Remove duplicate prompt-completion pairs
            flag_outliers: Flag statistical outliers
            strict_mode: Reject prompts with warnings (otherwise just flag)
        """
        self.deduplicate = deduplicate
        self.flag_outliers = flag_outliers
        self.strict_mode = strict_mode
        self._seen_hashes: set[str] = set()
    
    def validate_batch(
        self,
        prompts: list[CanonicalPrompt],
    ) -> ValidationResult:
        """
        Validate a batch of prompts.
        
        Args:
            prompts: List of prompts to validate
            
        Returns:
            ValidationResult with valid/invalid prompts and statistics
        """
        result = ValidationResult()
        self._seen_hashes.clear()
        
        # Calculate statistics for outlier detection
        lengths = [len(p.prompt_text) for p in prompts]
        if lengths and self.flag_outliers:
            sorted_lengths = sorted(lengths)
            n = len(sorted_lengths)
            low_threshold = sorted_lengths[int(n * self.OUTLIER_LOW_PERCENTILE / 100)] if n > 100 else self.MIN_PROMPT_LENGTH
            high_threshold = sorted_lengths[int(n * self.OUTLIER_HIGH_PERCENTILE / 100)] if n > 100 else self.MAX_PROMPT_LENGTH
        else:
            low_threshold = self.MIN_PROMPT_LENGTH
            high_threshold = self.MAX_PROMPT_LENGTH
        
        for prompt in prompts:
            is_valid, error = self._validate_single(prompt, low_threshold, high_threshold)
            
            if is_valid:
                # Check for duplicates
                if self.deduplicate:
                    hash_key = self._compute_hash(prompt)
                    if hash_key in self._seen_hashes:
                        result.duplicates_removed += 1
                        continue
                    self._seen_hashes.add(hash_key)
                
                result.valid_prompts.append(prompt)
            else:
                result.invalid_prompts.append((prompt, error))
        
        logger.info(
            "Validation complete",
            **result.summary(),
        )
        
        return result
    
    def _validate_single(
        self,
        prompt: CanonicalPrompt,
        low_threshold: int,
        high_threshold: int,
    ) -> tuple[bool, str]:
        """
        Validate a single prompt.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # 1. Completeness check
        if not prompt.id:
            return False, "Missing prompt ID"
        
        if not prompt.messages or len(prompt.messages) == 0:
            return False, "No messages in prompt"
        
        if not prompt.completion:
            return False, "Missing completion data"
        
        if not prompt.completion.text:
            return False, "Empty completion text"
        
        # 2. Format validation
        for i, msg in enumerate(prompt.messages):
            if not isinstance(msg.content, str):
                return False, f"Message {i} content is not a string"
            if not msg.content.strip():
                return False, f"Message {i} has empty content"
        
        # 3. Length validation
        prompt_length = len(prompt.prompt_text)
        completion_length = len(prompt.completion.text)
        
        if prompt_length < self.MIN_PROMPT_LENGTH:
            return False, f"Prompt too short ({prompt_length} chars)"
        
        if prompt_length > self.MAX_PROMPT_LENGTH:
            return False, f"Prompt too long ({prompt_length} chars)"
        
        if completion_length < self.MIN_COMPLETION_LENGTH:
            return False, f"Completion too short ({completion_length} chars)"
        
        if completion_length > self.MAX_COMPLETION_LENGTH:
            return False, f"Completion too long ({completion_length} chars)"
        
        # 4. Outlier detection (flag but don't reject unless strict)
        if self.flag_outliers:
            if prompt_length < low_threshold or prompt_length > high_threshold:
                if self.strict_mode:
                    return False, f"Statistical outlier (length: {prompt_length})"
                # Otherwise just flag (could add to warnings)
        
        return True, ""
    
    def _compute_hash(self, prompt: CanonicalPrompt) -> str:
        """Compute hash for deduplication."""
        # Hash based on prompt content + completion
        content = "".join(m.content for m in prompt.messages)
        content += prompt.completion.text
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class SamplingStrategy:
    """
    Implements various sampling strategies for prompt selection.
    """
    
    @staticmethod
    def uniform_random(
        prompts: list[CanonicalPrompt],
        n: int,
        seed: int | None = None,
    ) -> list[CanonicalPrompt]:
        """
        Uniform random sampling.
        
        Args:
            prompts: Full list of prompts
            n: Number to sample
            seed: Random seed for reproducibility
        """
        import random
        if seed is not None:
            random.seed(seed)
        
        if n >= len(prompts):
            return prompts
        
        return random.sample(prompts, n)
    
    @staticmethod
    def stratified_by_complexity(
        prompts: list[CanonicalPrompt],
        n: int,
        seed: int | None = None,
    ) -> list[CanonicalPrompt]:
        """
        Stratified sampling by complexity level.
        
        Samples proportionally from each complexity category.
        """
        import random
        if seed is not None:
            random.seed(seed)
        
        if n >= len(prompts):
            return prompts
        
        # Group by complexity
        by_complexity: dict[str, list[CanonicalPrompt]] = defaultdict(list)
        for p in prompts:
            by_complexity[p.metadata.complexity.value].append(p)
        
        # Calculate proportions
        total = len(prompts)
        result = []
        remaining = n
        
        for complexity, group in sorted(by_complexity.items()):
            # Proportional sample size
            proportion = len(group) / total
            sample_size = min(int(n * proportion) + 1, len(group), remaining)
            
            result.extend(random.sample(group, sample_size))
            remaining -= sample_size
            
            if remaining <= 0:
                break
        
        return result[:n]
    
    @staticmethod
    def stratified_by_task_type(
        prompts: list[CanonicalPrompt],
        n: int,
        seed: int | None = None,
    ) -> list[CanonicalPrompt]:
        """
        Stratified sampling by task type.
        """
        import random
        if seed is not None:
            random.seed(seed)
        
        if n >= len(prompts):
            return prompts
        
        # Group by task type
        by_task: dict[str, list[CanonicalPrompt]] = defaultdict(list)
        for p in prompts:
            by_task[p.metadata.task_type.value].append(p)
        
        # Calculate proportions
        total = len(prompts)
        result = []
        remaining = n
        
        for task_type, group in sorted(by_task.items()):
            proportion = len(group) / total
            sample_size = min(int(n * proportion) + 1, len(group), remaining)
            
            result.extend(random.sample(group, sample_size))
            remaining -= sample_size
            
            if remaining <= 0:
                break
        
        return result[:n]
    
    @staticmethod
    def cost_focused(
        prompts: list[CanonicalPrompt],
        n: int,
        oversample_factor: float = 2.0,
        seed: int | None = None,
    ) -> list[CanonicalPrompt]:
        """
        Cost-focused sampling: oversample high-token prompts.
        
        High-token prompts drive costs, so focus analysis there.
        """
        import random
        if seed is not None:
            random.seed(seed)
        
        if n >= len(prompts):
            return prompts
        
        # Sort by token count (descending)
        sorted_prompts = sorted(
            prompts,
            key=lambda p: p.completion.total_tokens,
            reverse=True,
        )
        
        # Take more from high-token prompts
        high_cost_count = int(n * 0.6)  # 60% from top half by cost
        low_cost_count = n - high_cost_count
        
        midpoint = len(sorted_prompts) // 2
        high_cost_prompts = sorted_prompts[:midpoint]
        low_cost_prompts = sorted_prompts[midpoint:]
        
        result = []
        
        if high_cost_prompts:
            sample_size = min(high_cost_count, len(high_cost_prompts))
            result.extend(random.sample(high_cost_prompts, sample_size))
        
        if low_cost_prompts:
            sample_size = min(low_cost_count, len(low_cost_prompts))
            result.extend(random.sample(low_cost_prompts, sample_size))
        
        random.shuffle(result)
        return result[:n]
    
    @staticmethod
    def failure_focused(
        prompts: list[CanonicalPrompt],
        n: int,
        seed: int | None = None,
    ) -> list[CanonicalPrompt]:
        """
        Failure-focused sampling: prioritize prompts with low quality scores.
        
        Focus on prompts where current model may be underperforming.
        """
        import random
        if seed is not None:
            random.seed(seed)
        
        if n >= len(prompts):
            return prompts
        
        # Separate by ground truth availability
        with_feedback = [
            p for p in prompts
            if p.ground_truth and p.ground_truth.quality_score is not None
        ]
        without_feedback = [p for p in prompts if p not in with_feedback]
        
        result = []
        
        if with_feedback:
            # Sort by quality score (ascending = worst first)
            sorted_by_quality = sorted(
                with_feedback,
                key=lambda p: p.ground_truth.quality_score if p.ground_truth else 1.0,
            )
            
            # Take bottom 50% (poor performers)
            poor_count = min(int(n * 0.7), len(sorted_by_quality) // 2)
            result.extend(sorted_by_quality[:poor_count])
        
        # Fill remaining with random from without feedback
        remaining = n - len(result)
        if remaining > 0 and without_feedback:
            sample_size = min(remaining, len(without_feedback))
            result.extend(random.sample(without_feedback, sample_size))
        
        random.shuffle(result)
        return result[:n]
