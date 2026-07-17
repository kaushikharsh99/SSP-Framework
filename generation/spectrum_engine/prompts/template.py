"""Prompt template rendering for the Spectrum Generation Engine.

Supports system/user prompt separation and variable substitution.
Prompts are never hardcoded inside providers.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..core.types import PromptRecord

logger = logging.getLogger("spectrum-engine.prompts")


class PromptTemplate:
    """Renders prompt templates with variable substitution."""

    def __init__(self, system_prompt: str, user_template: str, version: str = "v1"):
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.version = version

    def render(self, **variables: Any) -> PromptRecord:
        """Render the template with the given variables.
        
        Args:
            **variables: Key-value pairs to substitute in the user template.
                         The template uses Python str.format() syntax: {variable_name}
        
        Returns:
            A PromptRecord ready for generation.
        """
        user_prompt = self.user_template.format(**variables)
        return PromptRecord.create(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            template_version=self.version
        )

    def render_batch(self, variable_dicts: List[Dict[str, Any]]) -> List[PromptRecord]:
        """Render the template for multiple variable sets."""
        return [self.render(**v) for v in variable_dicts]
