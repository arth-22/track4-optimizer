"""Base adapter interface for data ingestion."""

from abc import ABC, abstractmethod
from datetime import datetime

from src.models.canonical import CanonicalPrompt


class BaseAdapter(ABC):
    """
    Abstract base class for data ingestion adapters.
    
    All adapters convert source-specific formats to the canonical format.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this data source."""
        ...

    @abstractmethod
    async def fetch_prompts(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int | None = None,
    ) -> list[CanonicalPrompt]:
        """
        Fetch prompts from the data source.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of prompts to fetch
            
        Returns:
            List of prompts in canonical format
        """
        ...

    @abstractmethod
    async def count_available(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> int:
        """
        Count available prompts without fetching them.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Count of available prompts
        """
        ...

    async def validate_connection(self) -> bool:
        """
        Validate that the data source is accessible.
        
        Returns:
            True if connection is valid
        """
        try:
            count = await self.count_available(limit=1)
            return count >= 0
        except Exception:
            return False

    def _normalize_to_canonical(self, raw_data: dict) -> CanonicalPrompt:
        """
        Convert raw data to canonical format.
        
        Subclasses should override this for source-specific normalization.
        """
        raise NotImplementedError("Subclasses must implement _normalize_to_canonical")
