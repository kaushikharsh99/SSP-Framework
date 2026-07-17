"""Declarative Recipe structures for combinatorial spectrum generation.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class RecipeConfig:
    """Combines prompt layouts and sampling grids into a single declarative configuration."""
    name: str
    personas: List[str] = field(default_factory=list)
    strategies: List[str] = field(default_factory=list)
    styles: List[str] = field(default_factory=list)
    query_rewrites: List[str] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=lambda: [1.0])
    top_p: List[float] = field(default_factory=lambda: [1.0])
    seeds: List[int] = field(default_factory=lambda: [None])
    samples_per_configuration: int = 1


def load_recipe(path: str) -> RecipeConfig:
    """Loads a RecipeConfig from a YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Recipe configuration file not found at: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    recipe_data = data.get("recipe", {})
    
    return RecipeConfig(
        name=recipe_data.get("name", "unnamed_recipe"),
        personas=recipe_data.get("personas", []),
        strategies=recipe_data.get("strategies", []),
        styles=recipe_data.get("styles", []),
        query_rewrites=recipe_data.get("query_rewrites", []),
        temperatures=recipe_data.get("sampling", {}).get("temperatures", [1.0]),
        top_p=recipe_data.get("sampling", {}).get("top_p", [1.0]),
        seeds=recipe_data.get("sampling", {}).get("seeds", [None]),
        samples_per_configuration=recipe_data.get("sampling", {}).get("samples_per_configuration", 1)
    )
