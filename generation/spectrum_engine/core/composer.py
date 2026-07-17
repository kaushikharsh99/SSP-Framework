"""Prompt composition engine for the Spectrum Generation Engine.
Assembles the system and user parts dynamically from active Job configurations.
"""

import os
from pathlib import Path

from .types import PromptRecord
from .job import GenerationJob
from ..prompts.registry import load_persona_prompt, load_style_prompt


def load_strategy_prompt(strategy_name: str) -> str:
    """Loads prompt text from prompts/strategies/{strategy_name}.md if it exists."""
    if strategy_name in ("default", "standard", ""):
        return ""
    prompts_dir = Path(__file__).parent.parent / "prompts"
    path = prompts_dir / "strategies" / f"{strategy_name}.md"
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_rewrite_prompt(rewrite_name: str, problem_text: str) -> str:
    """Formats the user query using rewrite templates."""
    builtins = {
        "standard": "Solve the following problem:\n\n{problem}",
        "alternative_method": "Solve the following problem using a less obvious or alternative method:\n\n{problem}",
        "beginner": "Solve the following problem and explain it in simple terms for a beginner:\n\n{problem}",
        "elegant": "Solve the following problem using an elegant mathematical shortcut:\n\n{problem}",
    }
    
    if rewrite_name in ("default", ""):
        rewrite_name = "standard"
        
    template = builtins.get(rewrite_name)
    if not template:
        prompts_dir = Path(__file__).parent.parent / "prompts"
        path = prompts_dir / "query_rewrites" / f"{rewrite_name}.md"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                template = f.read().strip()
        else:
            template = "{problem}"
            
    return template.format(problem=problem_text)


class PromptComposer:
    """Aggregates WHO (persona), HOW (strategy), WRITING FORMAT (style), and FRAMING (rewrite)."""

    @staticmethod
    def compose(job: GenerationJob) -> PromptRecord:
        """Assembles systemic instruction block and user queries into a PromptRecord.
        
        Args:
            job: Resolved GenerationJob defining the prompt elements.
            
        Returns:
            A composite PromptRecord object ready for provider dispatch.
        """
        # 1. Resolve Persona
        try:
            persona_inst = load_persona_prompt(job.persona)
        except Exception:
            persona_inst = ""

        # 2. Resolve Strategy
        strategy_inst = load_strategy_prompt(job.strategy)

        # 3. Resolve Style
        try:
            style_inst = load_style_prompt(job.style)
        except Exception:
            style_inst = ""

        # Assemble the System prompt blocks
        system_parts = []
        if persona_inst:
            system_parts.append(persona_inst)
        if strategy_inst:
            system_parts.append(f"Reasoning Strategy:\n{strategy_inst}")
        if style_inst:
            system_parts.append(f"Format instructions:\n{style_inst}")
            
        system_prompt = "\n\n".join(system_parts)
        if not system_prompt:
            system_prompt = "You are a helpful mathematical reasoning assistant."

        # Compile the formatted User query
        user_prompt = load_rewrite_prompt(job.rewrite, job.raw_user_prompt)

        return PromptRecord(
            id=job.job_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata={
                "base_prompt_id": job.prompt_id,
                **job.metadata
            }
        )
