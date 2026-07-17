"""Prompt version registry for the Spectrum Generation Engine.

Maintains named prompt templates that can be loaded by the scheduler.
Supports versioning so experiments can reference specific prompt versions.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from .template import PromptTemplate

logger = logging.getLogger("spectrum-engine.prompts.registry")

# Global registry of named templates
_REGISTRY: Dict[str, PromptTemplate] = {}


def register_template(name: str, template: PromptTemplate) -> None:
    """Register a prompt template by name."""
    _REGISTRY[name] = template
    logger.debug(f"Registered prompt template: '{name}' (version {template.version})")


def get_template(name: str) -> PromptTemplate:
    """Retrieve a registered prompt template by name."""
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(f"Prompt template '{name}' not found. Available: {available}")
    return _REGISTRY[name]


def list_templates() -> Dict[str, str]:
    """List all registered templates with their versions."""
    return {name: t.version for name, t in _REGISTRY.items()}


# --- Built-in Templates ---

register_template("default", PromptTemplate(
    system_prompt="You are a helpful assistant that solves problems step by step.",
    user_template="{problem}",
    version="v1"
))

register_template("reasoning", PromptTemplate(
    system_prompt=(
        "You are a reasoning assistant. Think step by step inside <think>...</think> tags, "
        "then provide your final answer inside <answer>...</answer> tags."
    ),
    user_template="{problem}",
    version="v1"
))

register_template("math", PromptTemplate(
    system_prompt=(
        "You are a mathematics expert. Show your complete reasoning inside <think>...</think> tags. "
        "Provide your final numerical answer inside <answer>...</answer> tags."
    ),
    user_template="Solve the following problem:\n\n{problem}",
    version="v1"
))
