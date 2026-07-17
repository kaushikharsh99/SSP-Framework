"""Abstract provider interface for the Spectrum Generation Engine.

All providers (API, vLLM, future backends) implement this interface.
The scheduler interacts ONLY through this contract — it never knows
which backend is executing generation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from ..core.types import PromptRecord, ProviderInfo, SamplingConfig, Spectrum


class BaseProvider(ABC):
    """Abstract base class for all generation providers.
    
    Providers are async context managers. Usage:
    
        async with APIProvider(config) as provider:
            spectra = await provider.generate(prompts, sampling)
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (load models, create connections, etc.)."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompts: List[PromptRecord],
        sampling: SamplingConfig
    ) -> List[Spectrum]:
        """Generate spectra for a batch of prompts.
        
        Args:
            prompts: Batch of prompts to generate responses for.
            sampling: Sampling configuration to use.
            
        Returns:
            List of Spectrum objects, one per input prompt.
            
        Raises:
            ProviderError: If generation fails after retries.
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources (close connections, unload models, etc.)."""
        pass

    @abstractmethod
    def info(self) -> ProviderInfo:
        """Return metadata about this provider."""
        pass

    async def __aenter__(self) -> BaseProvider:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()
