"""
Core data schemas for the LLM Post-Training Lab workspace.
Defines the standard DatasetEntry and Prompt structures.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetEntry:
    """Represents a standardized raw task sample from any dataset source.
    
    Attributes:
        id: Unique identifier for the sample (typically a hash or UUID string).
        prompt: Raw problem statement or user question.
        ground_truth_answer: Cleaned exact target answer string (e.g. "42", "print('hello')").
        test_cases: Optional lists of test assertions for code execution checks.
        metadata: Optional task information (difficulty, domain, source name).
    """
    id: str
    prompt: str
    ground_truth_answer: str
    test_cases: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prompt:
    """Represents the fully prepared model input prompt.
    
    Attributes:
        id: Maps directly back to the source DatasetEntry.id.
        system_prompt: System context/instructions (e.g., formatting directives).
        user_query: Formatted query body.
        formatted_prompt: Merged final string formatted according to the model's chat template.
        metadata: Additional tokenization or formatting metadata.
    """
    id: str
    system_prompt: str
    user_query: str
    formatted_prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """Represents a single candidate generation trajectory from the model.
    
    Attributes:
        id: Unique hash of the generated response.
        prompt_id: Reference back to the Prompt.id.
        text: Raw generated output string (thinking + answer).
        thinking_trace: Substring of step-by-step thinking (e.g. inside <think>).
        extracted_answer: Substring of final target answer (e.g. inside <answer>).
        token_ids: List[int] = field(default_factory=list)
        logprobs: List[float] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)
    """
    id: str
    prompt_id: str
    text: str
    thinking_trace: str = ""
    extracted_answer: str = ""
    token_ids: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Represents the outcome of correctness verification.
    
    Attributes:
        trajectory_id: Reference to the Response.id.
        is_correct: bool.
        error_message: Optional compiler error or execution warning.
        metrics: Dictionary of execution metadata.
    """
    trajectory_id: str
    is_correct: bool
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Spectrum:
    """Represents a generated spectrum of multiple candidate trajectories for a Prompt.
    
    Attributes:
        prompt: The Prompt object.
        responses: List of Response objects.
        generation_config: Dict of generation parameters used.
        metadata: Dict of custom metadata (model name, run id).
        diversity_statistics: Dict of calculated diversity metrics.
        generation_timing: Dict of timing stats (latency, throughput).
    """
    prompt: Prompt
    responses: List[Response]
    generation_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    diversity_statistics: Dict[str, Any] = field(default_factory=dict)
    generation_timing: Dict[str, Any] = field(default_factory=dict)
