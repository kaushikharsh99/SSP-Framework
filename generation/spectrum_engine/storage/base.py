"""Abstract storage interface for the Spectrum Generation Engine.

Storage backends write generated spectra to persistent storage.
They support append mode, auto-flushing, and checkpoint recovery.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Set

from ..core.types import Spectrum


class BaseStorage(ABC):
    """Abstract base class for spectrum storage backends.
    
    Storage backends are async context managers. Usage:
    
        async with JSONLStorage(config) as storage:
            await storage.write(spectrum)
    """

    @abstractmethod
    async def write(self, spectrum: Spectrum) -> None:
        """Write a single spectrum to storage.
        
        Implementations should buffer writes and flush periodically
        for performance, but must guarantee durability on flush.
        """
        pass

    @abstractmethod
    async def flush(self) -> None:
        """Force flush any buffered writes to disk."""
        pass

    @abstractmethod
    def get_completed_ids(self) -> Set[str]:
        """Return the set of prompt IDs already written to storage.
        
        Used for resume support — the scheduler skips prompts whose
        IDs appear in this set.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Flush remaining data and close the storage backend."""
        pass

    async def __aenter__(self) -> BaseStorage:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
