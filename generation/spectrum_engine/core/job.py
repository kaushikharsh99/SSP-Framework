"""Generation Job planning structures for the Spectrum Generation Engine.
"""

import itertools
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .types import PromptRecord
from .recipe import RecipeConfig


@dataclass
class GenerationJob:
    """Atomic generation execution unit containing fully resolved prompt parameters."""
    job_id: str
    prompt_id: str
    raw_user_prompt: str
    persona: str
    strategy: str
    style: str
    rewrite: str
    temperature: float
    top_p: float
    seed: Optional[int]
    sample_index: int
    metadata: Dict[str, Any]


class JobPlanner:
    """Combines a base dataset and a declarative Recipe into planned GenerationJobs."""

    @staticmethod
    def plan(prompts: List[PromptRecord], recipe: RecipeConfig) -> List[GenerationJob]:
        """Expands the combinatorial Cartesian grid of parameters into atomic jobs.
        
        Args:
            prompts: Input prompt records from a dataset.
            recipe: The recipe configuration determining personas, strategies, etc.
            
        Returns:
            Flat list of GenerationJob records ready for compilation and execution.
        """
        jobs = []
        
        # Pull parameter grids, using standard defaults if a grid is left empty
        personas = recipe.personas or ["default"]
        strategies = recipe.strategies or ["default"]
        styles = recipe.styles or ["standard"]
        rewrites = recipe.query_rewrites or ["standard"]
        temperatures = recipe.temperatures or [1.0]
        top_ps = recipe.top_p or [1.0]
        seeds = recipe.seeds or [None]
        
        for prompt in prompts:
            # Perform Cartesian product expansion
            for (persona, strategy, style, rewrite, temp, top_p, seed) in itertools.product(
                personas, strategies, styles, rewrites, temperatures, top_ps, seeds
            ):
                for idx in range(recipe.samples_per_configuration):
                    # Compile a unique hash ID for tracking progress/resume state
                    job_id = f"{prompt.id}_{persona}_{strategy}_{style}_{rewrite}_t{temp}_p{top_p}_s{seed}_idx{idx}"
                    
                    metadata = {
                        "persona": persona,
                        "strategy": strategy,
                        "style": style,
                        "rewrite": rewrite,
                        "temperature": temp,
                        "top_p": top_p,
                        "seed": seed,
                        "sample_index": idx
                    }
                    
                    jobs.append(GenerationJob(
                        job_id=job_id,
                        prompt_id=prompt.id,
                        raw_user_prompt=prompt.user_prompt,
                        persona=persona,
                        strategy=strategy,
                        style=style,
                        rewrite=rewrite,
                        temperature=temp,
                        top_p=top_p,
                        seed=seed,
                        sample_index=idx,
                        metadata=metadata
                    ))
                    
        return jobs
