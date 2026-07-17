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
