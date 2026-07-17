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


import os
from pathlib import Path

def load_persona_prompt(persona_name: str) -> str:
    """Loads prompt text from prompts/personas/{persona_name}.md."""
    prompts_dir = Path(__file__).parent
    path = prompts_dir / "personas" / f"{persona_name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Persona file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_style_prompt(style_name: str) -> str:
    """Loads prompt text from prompts/styles/{style_name}.md."""
    prompts_dir = Path(__file__).parent
    path = prompts_dir / "styles" / f"{style_name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Style file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_combined_system_prompt(persona_name: str, style_name: str) -> str:
    """Combines a persona and a formatting style into a single system prompt."""
    persona_content = load_persona_prompt(persona_name)
    style_content = load_style_prompt(style_name)
    return f"{persona_content}\n\nFormat instructions:\n{style_content}"
