"""Unit tests for the RecipeExecutor class.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path

from generation.spectrum_engine.core.executor import RecipeExecutor
from generation.spectrum_engine.core.job import GenerationJob
from generation.spectrum_engine.core.types import PromptRecord, ResponseRecord, SamplingConfig, Spectrum, ProviderInfo
from generation.spectrum_engine.providers.base import BaseProvider
from generation.spectrum_engine.storage.jsonl_storage import JSONLStorage
from generation.spectrum_engine.storage.checkpoint import CheckpointManager
from generation.spectrum_engine.core.config import StorageConfig


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def executor_env():
    """Setup temporary directories for storage and checkpoints."""
    test_dir = tempfile.mkdtemp()
    
    storage_config = StorageConfig(
        format="jsonl",
        output_dir=test_dir,
        filename_template="spectra.jsonl",
        flush_interval=1,
        checkpoint_dir=os.path.join(test_dir, ".checkpoints")
    )
    
    storage = JSONLStorage(storage_config, output_path=os.path.join(test_dir, "spectra.jsonl"))
    checkpoint = CheckpointManager(storage_config.checkpoint_dir, job_id="test_exec_job")
    
    yield test_dir, storage, checkpoint
    
    shutil.rmtree(test_dir, ignore_errors=True)


class MockExecutorProvider(BaseProvider):
    def __init__(self):
        self.generate_call_count = 0

    async def initialize(self) -> None:
        pass

    async def generate(self, prompts, sampling):
        self.generate_call_count += 1
        spectra = []
        for prompt in prompts:
            responses = [
                ResponseRecord(
                    id=f"r-{prompt.id}",
                    prompt_id=prompt.id,
                    text="mock response text",
                    token_count=5,
                    finish_reason="stop"
                )
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
        return ProviderInfo(name="mock_exec", backend="mock", model="mock-model")


@pytest.mark.anyio
async def test_recipe_executor_lifecycle(executor_env):
    test_dir, storage, checkpoint = executor_env
    provider = MockExecutorProvider()
    
    executor = RecipeExecutor(provider, storage, checkpoint, max_batch_size=2)
    
    # 1. Create a set of 3 GenerationJobs
    # Group A: temp=0.5, top_p=0.9, seed=42 (2 jobs)
    # Group B: temp=1.0, top_p=0.9, seed=42 (1 job)
    jobs = [
        GenerationJob(
            job_id="job_1",
            prompt_id="q1",
            raw_user_prompt="query 1",
            persona="analytical",
            strategy="default",
            style="standard",
            rewrite="standard",
            temperature=0.5,
            top_p=0.9,
            seed=42,
            sample_index=0,
            metadata={"meta": 1}
        ),
        GenerationJob(
            job_id="job_2",
            prompt_id="q2",
            raw_user_prompt="query 2",
            persona="analytical",
            strategy="default",
            style="standard",
            rewrite="standard",
            temperature=0.5,
            top_p=0.9,
            seed=42,
            sample_index=1,
            metadata={"meta": 2}
        ),
        GenerationJob(
            job_id="job_3",
            prompt_id="q1",
            raw_user_prompt="query 1",
            persona="teacher",
            strategy="default",
            style="standard",
            rewrite="standard",
            temperature=1.0,
            top_p=0.9,
            seed=42,
            sample_index=0,
            metadata={"meta": 3}
        )
    ]
    
    # 2. Run executor
    async with storage:
        report = await executor.execute(jobs)
        
    assert report.total_prompts == 3
    assert report.completed_prompts == 3
    assert report.failed_prompts == 0
    # Two groups: Group A (1 batch of size 2), Group B (1 batch of size 1)
    assert provider.generate_call_count == 2
    
    # 3. Check storage output exists and has 3 lines
    output_file = Path(test_dir) / "spectra.jsonl"
    assert output_file.exists()
    
    with open(output_file, "r") as f:
        lines = f.readlines()
    assert len(lines) == 3
    
    # 4. Check checkpointing works
    checkpoint.load()
    assert checkpoint.is_completed("job_1")
    assert checkpoint.is_completed("job_2")
    assert checkpoint.is_completed("job_3")
    
    # 5. Verify resume skip execution
    provider.generate_call_count = 0  # Reset counter
    async with storage:
        resume_report = await executor.execute(jobs)
        
    assert resume_report.total_prompts == 3
    assert resume_report.completed_prompts == 3  # already completed
    assert resume_report.failed_prompts == 0
    # Mock generator shouldn't be called because everything is completed
    assert provider.generate_call_count == 0
