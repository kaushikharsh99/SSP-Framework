"""
Unit tests for the configuration models.
"""

import os
import tempfile
import pytest
from omegaconf import OmegaConf
from ssp_framework.core.config import (
    ExperimentConfig,
    ModelConfig,
    DatasetConfig,
    load_config_from_file,
)


def test_default_config(experiment_config: ExperimentConfig) -> None:
    assert experiment_config.experiment_name == "test_experiment"
    assert experiment_config.model.spectrum_dimension == 16
    assert experiment_config.dataset.split == "validation"


def test_config_serialization() -> None:
    # Test that we can convert the configuration to a dictionary
    config = ExperimentConfig(
        model=ModelConfig(name="test-model"),
        dataset=DatasetConfig(path="test-path"),
    )
    
    cfg_dict = OmegaConf.to_container(OmegaConf.structured(config), resolve=True)
    assert isinstance(cfg_dict, dict)
    assert cfg_dict["model"]["name"] == "test-model"
    assert cfg_dict["dataset"]["path"] == "test-path"


def test_load_config_from_file() -> None:
    # Create a temporary yaml configuration file
    yaml_content = """
experiment_name: custom_ssp_run
model:
  name: customized-llama
  spectrum_dimension: 256
dataset:
  path: customized-dataset-path
"""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(yaml_content)
        temp_file_name = f.name

    try:
        loaded_cfg = load_config_from_file(temp_file_name)
        assert loaded_cfg.experiment_name == "custom_ssp_run"
        assert loaded_cfg.model.name == "customized-llama"
        assert loaded_cfg.model.spectrum_dimension == 256
        assert loaded_cfg.dataset.path == "customized-dataset-path"
    finally:
        os.remove(temp_file_name)
