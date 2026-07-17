"""Recipe execution coordinator for the Spectrum Generation Engine.
Dispatches planned GenerationJobs to backend-agnostic providers.
"""

import time
import logging
from typing import List, Dict, Any, Optional

from .job import GenerationJob
from .composer import PromptComposer
from .types import PromptRecord, ResponseRecord, SamplingConfig, Spectrum, ProviderInfo, GenerationReport
from ..providers.base import BaseProvider
from ..storage.jsonl_storage import JSONLStorage
from ..storage.checkpoint import CheckpointManager

logger = logging.getLogger("spectrum-engine.executor")


class RecipeExecutor:
    """Consumes GenerationJobs, groups by sampling parameters, and executes batches via providers."""

    def __init__(
        self,
        provider: BaseProvider,
        storage: JSONLStorage,
        checkpoint: CheckpointManager,
        max_batch_size: int = 16
    ):
        self.provider = provider
        self.storage = storage
        self.checkpoint = checkpoint
        self.max_batch_size = max_batch_size

    async def execute(self, jobs: List[GenerationJob]) -> GenerationReport:
        """Processes and runs all planned GenerationJobs with resume capability.
        
        Args:
            jobs: List of GenerationJob objects to execute.
            
        Returns:
            A GenerationReport detailing progress metrics.
        """
        # 1. Filter out completed tasks using checkpoint manager
        active_jobs = [j for j in jobs if not self.checkpoint.is_completed(j.job_id)]
        
        total_jobs = len(jobs)
        completed_jobs = total_jobs - len(active_jobs)
        failed_jobs = 0
        
        logger.info(
            f"RecipeExecutor started. Total planned: {total_jobs}, "
            f"Already completed: {completed_jobs}, Pending: {len(active_jobs)}"
        )
        
        if not active_jobs:
            return GenerationReport(
                total_prompts=total_jobs,
                completed_prompts=total_jobs,
                skipped_prompts=total_jobs,
                failed_prompts=0,
                elapsed_seconds=0.0
            )
            
        start_time = time.time()
        
        # 2. Group active jobs by sampling criteria to batch concurrent prompts safely
        grouped_jobs: Dict[tuple, List[GenerationJob]] = {}
        for job in active_jobs:
            key = (job.temperature, job.top_p, job.seed)
            grouped_jobs.setdefault(key, []).append(job)
            
        # 3. Execute grouped tasks
        for (temp, top_p, seed), job_list in grouped_jobs.items():
            logger.info(
                f"Executing sampling block (temp={temp}, top_p={top_p}, seed={seed}) "
                f"with {len(job_list)} jobs."
            )
            
            for i in range(0, len(job_list), self.max_batch_size):
                chunk = job_list[i : i + self.max_batch_size]
                
                # Compose final prompts from job configurations
                prompts_to_dispatch = [PromptComposer.compose(job) for job in chunk]
                
                sampling = SamplingConfig(
                    temperature=temp,
                    top_p=top_p,
                    seed=seed,
                    n=1  # GenerationJobs represent a single execution trajectory
                )
                
                try:
                    # Dispatch to underlying provider
                    spectra = await self.provider.generate(prompts_to_dispatch, sampling)
                    
                    # Store spectra and commit checkpoint status
                    for idx, spec in enumerate(spectra):
                        job = chunk[idx]
                        
                        # Add Job metadata to spectrum
                        spec.metadata["job_id"] = job.job_id
                        spec.metadata["persona"] = job.persona
                        spec.metadata["strategy"] = job.strategy
                        spec.metadata["style"] = job.style
                        spec.metadata["rewrite"] = job.rewrite
                        
                        await self.storage.write(spec)
                        self.checkpoint.mark_completed(job.job_id)
                        completed_jobs += 1
                        
                    self.checkpoint.save()
                    logger.info(f"Execution progress: {completed_jobs}/{total_jobs} jobs completed.")
                    
                except Exception as e:
                    logger.error(
                        f"Failed to execute batch of {len(chunk)} jobs: {e}",
                        exc_info=True
                    )
                    failed_jobs += len(chunk)
                    continue

        skipped_jobs = total_jobs - len(active_jobs)
        total_responses = completed_jobs - skipped_jobs
        
        return GenerationReport(
            total_prompts=total_jobs,
            completed_prompts=completed_jobs,
            skipped_prompts=skipped_jobs,
            failed_prompts=failed_jobs,
            total_responses=total_responses,
            elapsed_seconds=time.time() - start_time
        )
