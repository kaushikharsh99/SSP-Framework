"""Unit tests for JSONLStorage and CheckpointManager classes.
"""

import os
import tempfile
import json
import pytest
import shutil
from pathlib import Path

from generation.spectrum_engine.core.types import (
    PromptRecord,
    ResponseRecord,
    SamplingConfig,
    Spectrum,
    ProviderInfo,
)
from generation.spectrum_engine.core.config import StorageConfig
from generation.spectrum_engine.storage.jsonl_storage import JSONLStorage
from generation.spectrum_engine.storage.checkpoint import CheckpointManager


@pytest.fixture
def anyio_backend():
    """Pin the async backend to asyncio to avoid missing trio errors."""
    return "asyncio"


@pytest.fixture
def temp_dirs():
    """Fixture to handle temp dirs for storage testing."""
    test_dir = tempfile.mkdtemp()
    output_dir = os.path.join(test_dir, "outputs")
    checkpoint_dir = os.path.join(test_dir, "checkpoints")
    yield output_dir, checkpoint_dir
    # cleanup
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.mark.anyio
async def test_jsonl_storage_write_and_read(temp_dirs):
    output_dir, _ = temp_dirs
    
    config = StorageConfig(
        format="jsonl",
        output_dir=output_dir,
        filename_template="test_spectra_{timestamp}.jsonl",
        flush_interval=1,
        checkpoint_dir=""
    )

    # 1. Create storage
    storage = JSONLStorage(config)
    
    # Check that output path gets resolved correctly
    assert storage._output_path is not None
    assert "test_spectra_" in storage._output_path
    
    # 2. Write mock Spectrum objects
    prompt1 = PromptRecord.create("system1", "user1", id="p1")
    prompt2 = PromptRecord.create("system1", "user2", id="p2")
    
    sampling = SamplingConfig(temperature=0.8, n=1)
    provider = ProviderInfo(name="mock-provider", backend="mock", model="mock-model")
    
    resp1 = ResponseRecord(
        id="r1",
        prompt_id="p1",
        text="response1",
        thinking_trace="thinking1",
        extracted_answer="ans1",
        token_count=10,
        finish_reason="stop",
        latency_ms=100.0
    )
    resp2 = ResponseRecord(
        id="r2",
        prompt_id="p2",
        text="response2",
        thinking_trace="thinking2",
        extracted_answer="ans2",
        token_count=12,
        finish_reason="stop",
        latency_ms=120.0
    )

    spec1 = Spectrum(prompt=prompt1, responses=[resp1], sampling_config=sampling, provider_info=provider)
    spec2 = Spectrum(prompt=prompt2, responses=[resp2], sampling_config=sampling, provider_info=provider)

    async with storage:
        await storage.write(spec1)
        await storage.write(spec2)

    # 3. Verify file content
    assert os.path.exists(storage._output_path)
    
    with open(storage._output_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    assert len(lines) == 2
    
    data1 = json.loads(lines[0])
    assert data1["prompt"]["id"] == "p1"
    assert data1["responses"][0]["text"] == "response1"
    
    data2 = json.loads(lines[1])
    assert data2["prompt"]["id"] == "p2"
    assert data2["responses"][0]["extracted_answer"] == "ans2"

    # 4. Verify get_completed_ids
    completed_ids = storage.get_completed_ids()
    assert completed_ids == {"p1", "p2"}


def test_checkpoint_manager(temp_dirs):
    _, checkpoint_dir = temp_dirs
    
    # 1. Setup manager
    manager = CheckpointManager(checkpoint_dir, job_id="test_job")
    manager.load()
    assert manager.completed_count == 0
    
    # 2. Mark and save
    manager.mark_completed("p1")
    manager.mark_completed("p2")
    assert manager.is_completed("p1")
    assert manager.completed_count == 2
    
    manager.save()
    
    # Check that file exists on disk
    checkpoint_file = Path(checkpoint_dir) / "test_job.checkpoint.json"
    assert checkpoint_file.exists()
    
    # 3. Reload in a new manager
    new_manager = CheckpointManager(checkpoint_dir, job_id="test_job")
    new_manager.load()
    
    assert new_manager.completed_count == 2
    assert new_manager.is_completed("p1")
    assert new_manager.is_completed("p2")
    assert not new_manager.is_completed("p3")
    assert new_manager.get_completed_ids() == {"p1", "p2"}
