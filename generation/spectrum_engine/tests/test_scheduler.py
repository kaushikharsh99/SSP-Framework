"""Unit tests for the BatchScheduler class.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path

from generation.spectrum_engine.core.config import SchedulerConfig, StorageConfig
from generation.spectrum_engine.core.types import (
    PromptRecord,
    ResponseRecord,
    SamplingConfig,
    Spectrum,
    ProviderInfo,
)
from generation.spectrum_engine.storage.jsonl_storage import JSONLStorage
from generation.spectrum_engine.storage.checkpoint import CheckpointManager
from generation.spectrum_engine.scheduler.batch_scheduler import BatchScheduler
from generation.spectrum_engine.providers.base import BaseProvider


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def temp_dirs():
    test_dir = tempfile.mkdtemp()
    output_dir = os.path.join(test_dir, "outputs")
    checkpoint_dir = os.path.join(test_dir, "checkpoints")
    yield output_dir, checkpoint_dir
    shutil.rmtree(test_dir, ignore_errors=True)


class MockSuccessProvider(BaseProvider):
    async def initialize(self) -> None:
        pass

    async def generate(self, prompts, sampling):
        spectra = []
        for prompt in prompts:
            responses = [
                ResponseRecord(
                    id=f"r-{prompt.id}-{i}",
                    prompt_id=prompt.id,
                    text=f"resp-{i}",
                    token_count=5,
                    finish_reason="stop"
                )
                for i in range(sampling.n)
            ]
            spectra.append(Spectrum(
                prompt=prompt,
                responses=responses,
                sampling_config=sampling,
                provider_info=self.info()
            ))
        return spectra

    async def shutdown(self) -> None:
        pass

    def info(self):
        return ProviderInfo(name="mock", backend="mock", model="mock-model")


class MockFailureProvider(BaseProvider):
    async def initialize(self) -> None:
        pass

    async def generate(self, prompts, sampling):
        raise ValueError("Simulated model failure")

    async def shutdown(self) -> None:
        pass

    def info(self):
        return ProviderInfo(name="mock-fail", backend="mock", model="mock-model")


@pytest.mark.anyio
async def test_scheduler_run_success(temp_dirs):
    output_dir, checkpoint_dir = temp_dirs

    sched_config = SchedulerConfig(
        batch_size=2,
        max_concurrent=1,
        checkpoint_interval=1,
        progress_interval=1
    )
    storage_config = StorageConfig(
        format="jsonl",
        output_dir=output_dir,
        filename_template="spectra_{timestamp}.jsonl",
        flush_interval=1,
        checkpoint_dir=checkpoint_dir
    )

    scheduler = BatchScheduler(sched_config)
    provider = MockSuccessProvider()
    storage = JSONLStorage(storage_config)
    checkpoint = CheckpointManager(checkpoint_dir, job_id="test_success_job")
    sampling = SamplingConfig(temperature=1.0, n=2)

    prompts = [
        PromptRecord.create("sys", "user1", id="p1"),
        PromptRecord.create("sys", "user2", id="p2"),
        PromptRecord.create("sys", "user3", id="p3"),
    ]

    async with storage:
        report = await scheduler.run(prompts, provider, storage, checkpoint, sampling)

    assert report.total_prompts == 3
    assert report.completed_prompts == 3
    assert report.skipped_prompts == 0
    assert report.failed_prompts == 0
    assert report.total_responses == 6
    assert report.total_tokens == 30  # 6 responses * 5 tokens each

    # Verify output file has 3 lines
    assert os.path.exists(storage._output_path)
    with open(storage._output_path, "r") as f:
        lines = f.readlines()
    assert len(lines) == 3

    # Verify checkpoint tracks completed prompt IDs
    assert checkpoint.get_completed_ids() == {"p1", "p2", "p3"}


@pytest.mark.anyio
async def test_scheduler_run_resume(temp_dirs):
    output_dir, checkpoint_dir = temp_dirs

    sched_config = SchedulerConfig(
        batch_size=1,
        max_concurrent=1,
        checkpoint_interval=1,
        progress_interval=1
    )
    storage_config = StorageConfig(
        format="jsonl",
        output_dir=output_dir,
        filename_template="spectra_{timestamp}.jsonl",
        flush_interval=1,
        checkpoint_dir=checkpoint_dir
    )

    # Pre-populate checkpoint with p1 completed
    checkpoint = CheckpointManager(checkpoint_dir, job_id="test_resume_job")
    checkpoint.mark_completed("p1")
    checkpoint.save()

    scheduler = BatchScheduler(sched_config)
    provider = MockSuccessProvider()
    storage = JSONLStorage(storage_config)
    sampling = SamplingConfig(temperature=1.0, n=1)

    prompts = [
        PromptRecord.create("sys", "user1", id="p1"),  # Should be skipped
        PromptRecord.create("sys", "user2", id="p2"),  # Should run
    ]

    async with storage:
        report = await scheduler.run(prompts, provider, storage, checkpoint, sampling)

    assert report.total_prompts == 2
    assert report.completed_prompts == 1
    assert report.skipped_prompts == 1
    assert report.failed_prompts == 0
    assert report.total_responses == 1

    # Verify checkpoint has both
    assert checkpoint.get_completed_ids() == {"p1", "p2"}


@pytest.mark.anyio
async def test_scheduler_run_failures(temp_dirs):
    output_dir, checkpoint_dir = temp_dirs

    sched_config = SchedulerConfig(
        batch_size=2,
        max_concurrent=1,
        checkpoint_interval=1,
        progress_interval=1
    )
    storage_config = StorageConfig(
        format="jsonl",
        output_dir=output_dir,
        filename_template="spectra_{timestamp}.jsonl",
        flush_interval=1,
        checkpoint_dir=checkpoint_dir
    )

    scheduler = BatchScheduler(sched_config)
    provider = MockFailureProvider()  # Always fails
    storage = JSONLStorage(storage_config)
    checkpoint = CheckpointManager(checkpoint_dir, job_id="test_fail_job")
    sampling = SamplingConfig(temperature=1.0, n=2)

    prompts = [
        PromptRecord.create("sys", "user1", id="p1"),
        PromptRecord.create("sys", "user2", id="p2"),
    ]

    async with storage:
        report = await scheduler.run(prompts, provider, storage, checkpoint, sampling)

    assert report.total_prompts == 2
    assert report.completed_prompts == 0
    assert report.failed_prompts == 2
    assert report.skipped_prompts == 0
    assert report.total_responses == 0
