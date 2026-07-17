"""
Configuration schemas and loading utilities for the SSP Framework.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from omegaconf import OmegaConf


@dataclass
class ModelConfig:
    """Configuration for base models and SSP specific parameters."""
    name: str = "meta-llama/Meta-Llama-3-8B"
    trust_remote_code: bool = False
    use_flash_attention_2: bool = True
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    # SSP-specific placeholder config
    spectrum_dimension: int = 128
    signal_dimension: int = 4096
    ssp_layers: List[int] = field(default_factory=list)


@dataclass
class DatasetConfig:
    """Configuration for data loading and preprocessing."""
    path: str = ""
    name: Optional[str] = None
    split: str = "train"
    max_length: int = 2048
    preprocessing_num_workers: int = 4
    batch_size: int = 8
    shuffle: bool = True


@dataclass
class TrainingConfig:
    """Configuration for model training/fine-tuning (SFT)."""
    output_dir: str = "outputs"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03
    gradient_accumulation_steps: int = 1
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 10
    fp16: bool = False
    bf16: bool = True
    seed: int = 42


@dataclass
class RLConfig:
    """Configuration for Reinforcement Learning paradigms (e.g. GRPO, PPO, DPO)."""
    algorithm: str = "PPO"  # PPO, GRPO, MGPO, DPO
    learning_rate: float = 1e-6
    kl_coef: float = 0.05
    cliprange: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    rollout_batch_size: int = 512
    step_batch_size: int = 128
    reward_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation and benchmarks."""
    benchmarks: List[str] = field(default_factory=list)
    batch_size: int = 16
    metrics: List[str] = field(default_factory=list)
    limit: Optional[int] = None  # limit number of samples for fast testing


@dataclass
class InferenceConfig:
    """Configuration for generation and deployment inference."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True


@dataclass
class ExperimentConfig:
    """Unified experiment configuration for SSP Framework."""
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: Optional[TrainingConfig] = None
    rl: Optional[RLConfig] = None
    evaluation: Optional[EvaluationConfig] = None
    inference: Optional[InferenceConfig] = None
    
    # Metadata
    experiment_name: str = "ssp_experiment"
    project_name: str = "ssp-framework"
    run_name: Optional[str] = None


def load_config_from_file(config_path: str) -> ExperimentConfig:
    """Loads configuration from a YAML file and validates against the schema."""
    yaml_cfg = OmegaConf.load(config_path)
    schema = OmegaConf.structured(ExperimentConfig)
    merged = OmegaConf.merge(schema, yaml_cfg)
    return OmegaConf.to_object(merged)  # type: ignore
