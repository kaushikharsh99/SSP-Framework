"""Configuration loading, validation, and merging for the Spectrum Engine.

All configuration is loaded from YAML files. Environment variables
can be referenced using ${VAR_NAME} syntax in config values.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .types import SamplingConfig


@dataclass
class ProviderConfig:
    """Provider-specific configuration."""
    type: str = "api"  # 'api' or 'vllm'
    
    # --- API provider fields ---
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    max_concurrent: int = 8
    
    # --- vLLM provider fields ---
    model_path: str = ""
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    trust_remote_code: bool = False
    max_model_len: Optional[int] = None


@dataclass
class SchedulerConfig:
    """Scheduler orchestration configuration."""
    batch_size: int = 16
    max_concurrent: int = 8
    checkpoint_interval: int = 50
    progress_interval: int = 10


@dataclass
class StorageConfig:
    """Output storage configuration."""
    format: str = "jsonl"
    output_dir: str = "./outputs"
    filename_template: str = "spectra_{timestamp}.jsonl"
    flush_interval: int = 10
    checkpoint_dir: str = "./.checkpoints"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: Optional[str] = None
    structured: bool = True


@dataclass
class PromptConfig:
    """Prompt template configuration."""
    template_name: str = "default"
    system_prompt: str = "You are a helpful assistant."
    version: str = "v1"


@dataclass
class EngineConfig:
    """Top-level engine configuration aggregating all sub-configs."""
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolves ${VAR_NAME} references in config values."""
    if isinstance(value, str):
        pattern = re.compile(r'\$\{([^}]+)\}')
        def replacer(match):
            env_var = match.group(1)
            env_val = os.environ.get(env_var, "")
            if not env_val:
                import logging
                logging.getLogger("spectrum-engine").warning(
                    f"Environment variable '{env_var}' not set, using empty string."
                )
            return env_val
        return pattern.sub(replacer, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def _dict_to_dataclass(cls, data: Dict[str, Any]):
    """Converts a dictionary to a dataclass, ignoring unknown fields."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def load_config(path: str) -> EngineConfig:
    """Loads and validates an EngineConfig from a YAML file.
    
    Args:
        path: Path to the YAML configuration file.
        
    Returns:
        Fully resolved EngineConfig instance.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If required fields are missing or invalid.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    
    if not raw:
        raise ValueError(f"Config file is empty: {path}")
    
    # Resolve environment variables
    resolved = _resolve_env_vars(raw)
    
    # Build sub-configs
    provider = _dict_to_dataclass(ProviderConfig, resolved.get("provider", {}))
    sampling = _dict_to_dataclass(SamplingConfig, resolved.get("sampling", {}))
    scheduler = _dict_to_dataclass(SchedulerConfig, resolved.get("scheduler", {}))
    storage = _dict_to_dataclass(StorageConfig, resolved.get("storage", {}))
    logging_cfg = _dict_to_dataclass(LoggingConfig, resolved.get("logging", {}))
    prompts = _dict_to_dataclass(PromptConfig, resolved.get("prompts", {}))
    
    return EngineConfig(
        provider=provider,
        sampling=sampling,
        scheduler=scheduler,
        storage=storage,
        logging=logging_cfg,
        prompts=prompts
    )
