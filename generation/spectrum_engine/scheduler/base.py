"""Abstract scheduler interface for the Spectrum Generation Engine.

The scheduler orchestrates the generation pipeline:
  Dataset → Batching → Provider → Storage → Checkpoint

It is completely backend-agnostic — it does not know whether
generation happens via API or local vLLM.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..core.types import GenerationReport, PromptRecord
from ..providers.base import BaseProvider
from ..storage.base import BaseStorage
from ..storage.checkpoint import CheckpointManager


class BaseScheduler(ABC):
    """Abstract base class for generation schedulers."""

    @abstractmethod
    async def run(
        self,
        prompts: List[PromptRecord],
        provider: BaseProvider,
        storage: BaseStorage,
        checkpoint: CheckpointManager,
        sampling: SamplingConfig
    ) -> GenerationReport:
        """Execute a full generation run.
        
        Args:
            prompts: Complete list of prompts to process.
            provider: The generation provider to use.
            storage: The storage backend to write spectra to.
            checkpoint: Checkpoint manager for resume support.
            sampling: The sampling parameters to pass to the provider.
            
        Returns:
            GenerationReport with summary statistics.
        """
        pass
