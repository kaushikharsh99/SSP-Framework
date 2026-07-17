"""Core data types for the Spectrum Generation Engine.

Every component in the engine communicates through these types.
No raw strings or untyped dicts cross module boundaries.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PromptRecord:
    """A single prompt to be sent to a provider for spectrum generation.
    
    Attributes:
        id: Unique identifier for this prompt (used for checkpoint/resume).
        system_prompt: System-level instruction for the model.
        user_prompt: The actual user query or problem statement.
        metadata: Arbitrary metadata (dataset source, difficulty, domain, etc.).
    """
    id: str
    system_prompt: str
    user_prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(system_prompt: str, user_prompt: str, id: Optional[str] = None, **metadata) -> PromptRecord:
        """Factory method for creating a PromptRecord with auto-generated ID."""
        return PromptRecord(
            id=id or str(uuid.uuid4()),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata=metadata
        )


@dataclass
class SamplingConfig:
    """Generation sampling parameters passed to any provider.
    
    These map directly to OpenAI-compatible API parameters
    and vLLM SamplingParams. Providers translate as needed.
    """
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 2048
    stop_sequences: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    n: int = 1  # Number of completions per prompt


@dataclass
class ResponseRecord:
    """A single generated response (one trajectory).
    
    Attributes:
        id: Unique response identifier.
        prompt_id: ID of the PromptRecord that produced this response.
        text: Full generated text.
        thinking_trace: Extracted <think>...</think> content, if present.
        extracted_answer: Extracted final answer, if parseable.
        token_count: Number of tokens generated.
        finish_reason: Why generation stopped ('stop', 'length', 'error').
        latency_ms: Time taken for this specific response in milliseconds.
        metadata: Provider-specific metadata (logprobs, model version, etc.).
    """
    id: str
    prompt_id: str
    text: str
    thinking_trace: str = ""
    extracted_answer: str = ""
    token_count: int = 0
    finish_reason: str = ""
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderInfo:
    """Metadata describing the provider that generated a spectrum."""
    name: str           # e.g. 'openrouter', 'vllm-local'
    backend: str        # 'api' or 'vllm'
    model: str          # e.g. 'qwen/qwen3-0.6b'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Spectrum:
    """A collection of diverse reasoning trajectories for a single prompt.
    
    This is the PRIMARY output of the Spectrum Generation Engine.
    Every downstream consumer (SFT, RL, analysis) operates on Spectrum objects.
    
    Attributes:
        prompt: The input prompt that was used.
        responses: List of N generated trajectories.
        sampling_config: The sampling parameters used.
        provider_info: Which provider/model generated this.
        created_at: Unix timestamp of generation.
        metadata: Run-level metadata (job id, batch index, etc.).
    """
    prompt: PromptRecord
    responses: List[ResponseRecord]
    sampling_config: SamplingConfig
    provider_info: ProviderInfo
    created_at: float = field(default_factory=time.time)
    diversity_statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationReport:
    """Summary statistics for a completed or in-progress generation run."""
    total_prompts: int = 0
    completed_prompts: int = 0
    skipped_prompts: int = 0      # Already completed (resume)
    failed_prompts: int = 0
    total_responses: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    avg_tokens_per_second: float = 0.0
    total_retries: int = 0
    estimated_cost_usd: float = 0.0
