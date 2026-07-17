"""Unit tests for the ExperimentConfig and ExperimentRunner classes.
"""

import pytest
import os
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import patch

from generation.spectrum_engine.core.experiment import load_experiment_config, ExperimentRunner
from generation.spectrum_engine.core.config import EngineConfig, StorageConfig
from generation.spectrum_engine.core.types import PromptRecord, ResponseRecord, SamplingConfig, Spectrum, ProviderInfo
from generation.spectrum_engine.providers.base import BaseProvider


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def experiment_env():
    """Fixture creating temporary experiment config and output dirs."""
    test_dir = tempfile.mkdtemp()
    
    # 1. Create a dummy provider config
    provider_config_path = os.path.join(test_dir, "provider_config.yaml")
    with open(provider_config_path, "w", encoding="utf-8") as f:
        f.write("""
provider:
  type: api
  base_url: https://api.example.com
  model: test-model
sampling:
  temperature: 1.0
scheduler:
  batch_size: 2
storage:
  output_dir: ./outputs
  filename_template: spec.jsonl
prompts:
  system_prompt: Default system
logging:
  level: INFO
""")

    # 2. Create a dummy experiment config
    experiment_config_path = os.path.join(test_dir, "experiment.yaml")
    with open(experiment_config_path, "w", encoding="utf-8") as f:
        f.write(f"""
experiment:
  name: test_experiment
  provider_config_path: {provider_config_path}
  dataset_path: dummy.jsonl
  verifier_type: math
  output_dir: {test_dir}/outputs
  recipes:
    test_recipe:
      personas: [analytical, teacher]
      styles: [standard]
      temperatures: [0.5, 1.0]
      seeds: [42]
      samples_per_configuration: 1
""")

    yield experiment_config_path, test_dir
    shutil.rmtree(test_dir, ignore_errors=True)


class MockExperimentProvider(BaseProvider):
    async def initialize(self) -> None:
        pass

    async def generate(self, prompts, sampling):
        spectra = []
        for prompt in prompts:
            # Check which persona is used in prompt metadata
            persona = prompt.metadata.get("persona", "analytical")
            
            # Analytical answers correctly, teacher answers incorrectly
            ans = "42" if persona == "analytical" else "99"
            
            responses = [
                ResponseRecord(
                    id=f"r-{prompt.id}",
                    prompt_id=prompt.id,
                    text=f"<think>thinking</think>\n<answer>{ans}</answer>",
                    token_count=10,
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
        return ProviderInfo(name="mock", backend="mock", model="mock-model")


@pytest.mark.anyio
async def test_experiment_runner(experiment_env):
    config_path, test_dir = experiment_env
    
    # 1. Test loading experiment configuration
    exp_config = load_experiment_config(config_path)
    assert exp_config.name == "test_experiment"
    assert exp_config.verifier_type == "math"
    assert "test_recipe" in exp_config.recipes
    
    recipe = exp_config.recipes["test_recipe"]
    assert recipe.personas == ["analytical", "teacher"]
    assert recipe.temperatures == [0.5, 1.0]

    # 2. Test ExperimentRunner execution
    runner = ExperimentRunner(exp_config)
    
    # Mock the provider construction inside run_recipe
    # We patch APIProvider to return our MockExperimentProvider
    prompts = [
        PromptRecord.create("sys", "user query", id="q1", answer="42")
    ]
    
    # Patch APIProvider
    with patch("generation.spectrum_engine.core.experiment.APIProvider", return_value=MockExperimentProvider()):
        summary = await runner.run_recipe("test_recipe", prompts)
        
    # 3. Verify Leaderboard results
    assert summary["num_completed_prompts"] == 1
    assert summary["total_generated_trajectories"] == 4  # 2 personas * 1 style * 2 temps * 1 seed = 4
    
    # Verify overall correctness (since analytical succeeds and teacher fails, correctness should be 50.0%)
    assert summary["overall_trajectory_correctness"] == "50.0%"
    
    # Verify leaderboard details
    leaderboard = summary["persona_leaderboard"]
    assert len(leaderboard) == 2
    
    # Find analytical and teacher entries
    analytical_entry = next(e for e in leaderboard if e["persona"] == "analytical")
    teacher_entry = next(e for e in leaderboard if e["persona"] == "teacher")
    
    assert analytical_entry["correctness"] == "100.0%"
    assert teacher_entry["correctness"] == "0.0%"
    assert analytical_entry["avg_tokens"] == 10
    
    # Verify summary report exists on disk
    summary_file = Path(test_dir) / "outputs" / "test_experiment" / "test_recipe" / "summary.json"
    assert summary_file.exists()
    
    with open(summary_file, "r") as f:
        data = json.load(f)
    assert data["recipe"] == "test_recipe"
