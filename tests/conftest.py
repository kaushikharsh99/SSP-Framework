"""
Shared pytest fixtures and configurations.
"""

import pytest
from ssp_framework.core.config import ExperimentConfig, ModelConfig, DatasetConfig


@pytest.fixture
def base_model_config() -> ModelConfig:
    return ModelConfig(
        name="dummy-model",
        trust_remote_code=False,
        use_flash_attention_2=False,
        torch_dtype="float32",
        spectrum_dimension=16,
        signal_dimension=64,
    )


@pytest.fixture
def base_dataset_config() -> DatasetConfig:
    return DatasetConfig(
        path="dummy-path",
        split="validation",
        max_length=512,
        batch_size=2,
    )


@pytest.fixture
def experiment_config(base_model_config, base_dataset_config) -> ExperimentConfig:
    return ExperimentConfig(
        model=base_model_config,
        dataset=base_dataset_config,
        experiment_name="test_experiment",
    )
