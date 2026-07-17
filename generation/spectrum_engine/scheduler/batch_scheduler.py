"""Batch scheduler implementation with checkpoint/resume support.

This is the primary scheduler for production generation runs.
It batches prompts, dispatches to the provider, handles failures,
and saves progress continuously.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import List

from ..core.config import SchedulerConfig
from ..core.types import GenerationReport, PromptRecord
from ..providers.base import BaseProvider
from ..storage.base import BaseStorage
from ..storage.checkpoint import CheckpointManager
from .base import BaseScheduler

logger = logging.getLogger("spectrum-engine.scheduler")


from ..utils.metrics import MetricsTracker
from ..core.types import SamplingConfig

class BatchScheduler(BaseScheduler):
    """Batch-oriented scheduler with checkpoint/resume and failure recovery."""

    def __init__(self, config: SchedulerConfig):
        self.config = config

    async def run(
        self,
        prompts: List[PromptRecord],
        provider: BaseProvider,
        storage: BaseStorage,
        checkpoint: CheckpointManager,
        sampling: SamplingConfig
    ) -> GenerationReport:
        """Execute batch generation with progress tracking and resume."""
        # 1. Resolve already completed work
        completed_ids = checkpoint.get_completed_ids() | storage.get_completed_ids()
        remaining_prompts = [p for p in prompts if p.id not in completed_ids]
        skipped_count = len(prompts) - len(remaining_prompts)

        # 2. Setup metrics tracker
        tracker = MetricsTracker(total_prompts=len(prompts), log_interval=self.config.progress_interval)
        for _ in range(skipped_count):
            tracker.record_skip()

        logger.info(
            f"Starting BatchScheduler. Total prompts: {len(prompts)}, "
            f"Already completed (skipped): {skipped_count}, "
            f"Remaining: {len(remaining_prompts)}"
        )

        if not remaining_prompts:
            logger.info("All prompts are already completed. Exiting execution loop.")
            return GenerationReport(
                total_prompts=len(prompts),
                completed_prompts=tracker.completed,
                skipped_prompts=tracker.skipped,
                failed_prompts=tracker.failed,
                total_responses=tracker.total_responses,
                total_tokens=tracker.total_tokens,
                elapsed_seconds=0.0,
                avg_tokens_per_second=0.0
            )

        # 3. Batch and dispatch loop
        batch_size = self.config.batch_size
        batches = [remaining_prompts[i:i + batch_size] for i in range(0, len(remaining_prompts), batch_size)]

        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} (size={len(batch)})")
            
            try:
                spectra = await provider.generate(batch, sampling)
                
                # Write spectra and update checkpoints
                for spec in spectra:
                    await storage.write(spec)
                    checkpoint.mark_completed(spec.prompt.id)
                    
                    num_responses = len(spec.responses)
                    num_tokens = sum(r.token_count for r in spec.responses)
                    tracker.record_success(num_responses, num_tokens)
                    
            except Exception as e:
                logger.error(f"Error executing batch {batch_idx + 1}: {e}")
                for _ in batch:
                    tracker.record_failure()
            
            # Periodic checkpoint saving & flushing
            if (tracker.completed + tracker.failed) % self.config.checkpoint_interval == 0:
                checkpoint.save()
                await storage.flush()

        # 4. Final save and flush
        checkpoint.save()
        await storage.flush()
        
        logger.info(tracker.summary())

        return GenerationReport(
            total_prompts=len(prompts),
            completed_prompts=tracker.completed,
            skipped_prompts=tracker.skipped,
            failed_prompts=tracker.failed,
            total_responses=tracker.total_responses,
            total_tokens=tracker.total_tokens,
            elapsed_seconds=tracker.elapsed,
            avg_tokens_per_second=tracker.tokens_per_second
        )
