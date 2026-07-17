"""Declarative Recipe structures and registry for combinatorial spectrum generation.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RecipeConfig:
    """Combines prompt layouts and sampling grids into a single declarative configuration."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    status: str = "experimental"
    tags: List[str] = field(default_factory=list)
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
        version=recipe_data.get("version", "1.0.0"),
        description=recipe_data.get("description", ""),
        author=recipe_data.get("author", ""),
        status=recipe_data.get("status", "experimental"),
        tags=recipe_data.get("tags", []),
        personas=recipe_data.get("personas", []),
        strategies=recipe_data.get("strategies", []),
        styles=recipe_data.get("styles", []),
        query_rewrites=recipe_data.get("query_rewrites", []),
        temperatures=recipe_data.get("sampling", {}).get("temperatures", [1.0]),
        top_p=recipe_data.get("sampling", {}).get("top_p", [1.0]),
        seeds=recipe_data.get("sampling", {}).get("seeds", [None]),
        samples_per_configuration=recipe_data.get("sampling", {}).get("samples_per_configuration", 1)
    )


class RecipeRegistry:
    """Registry for managing and resolving declarative Recipes by name and version."""
    
    _recipes_dir: Path = Path(__file__).parent.parent / "recipes"
    
    @classmethod
    def get(cls, name_or_spec: str) -> RecipeConfig:
        """Retrieves a recipe by name, optionally matching version (e.g. 'math_v1' or 'math_v1@1.0.0')."""
        cls.ensure_dir_exists()
        
        name = name_or_spec
        version = None
        if "@" in name_or_spec:
            name, version = name_or_spec.split("@", 1)
            
        for file in cls._recipes_dir.glob("*.yaml"):
            try:
                recipe = load_recipe(str(file))
                if recipe.name == name:
                    if version is None or recipe.version == version:
                        return recipe
            except Exception:
                continue
                
        raise KeyError(
            f"Recipe '{name_or_spec}' not found in registry folder '{cls._recipes_dir}'."
        )
        
    @classmethod
    def ensure_dir_exists(cls) -> None:
        cls._recipes_dir.mkdir(parents=True, exist_ok=True)
