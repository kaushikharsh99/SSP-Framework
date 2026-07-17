"""Unit tests for RecipeConfig, JobPlanner, and PromptComposer modules.
"""

import pytest
import os
import tempfile
import shutil

from generation.spectrum_engine.core.recipe import load_recipe, RecipeConfig
from generation.spectrum_engine.core.job import GenerationJob, JobPlanner
from generation.spectrum_engine.core.composer import PromptComposer
from generation.spectrum_engine.core.types import PromptRecord


@pytest.fixture
def recipe_yaml_path():
    """Create a temporary test recipe YAML file."""
    test_dir = tempfile.mkdtemp()
    yaml_path = os.path.join(test_dir, "test_recipe.yaml")
    
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("""
recipe:
  name: test_grid_recipe
  personas: [analytical, teacher]
  strategies: [default]
  styles: [standard]
  query_rewrites: [standard, elegant]
  sampling:
    temperatures: [0.5, 0.8]
    top_p: [0.9]
    seeds: [42]
    samples_per_configuration: 2
""")
        
    yield yaml_path
    shutil.rmtree(test_dir, ignore_errors=True)


def test_load_recipe(recipe_yaml_path):
    recipe = load_recipe(recipe_yaml_path)
    assert recipe.name == "test_grid_recipe"
    assert recipe.personas == ["analytical", "teacher"]
    assert recipe.temperatures == [0.5, 0.8]
    assert recipe.top_p == [0.9]
    assert recipe.seeds == [42]
    assert recipe.samples_per_configuration == 2


def test_job_planner_expansion(recipe_yaml_path):
    recipe = load_recipe(recipe_yaml_path)
    
    prompts = [
        PromptRecord.create("sys", "user problem 1", id="q1"),
        PromptRecord.create("sys", "user problem 2", id="q2")
    ]
    
    # 2 prompts
    # x 2 personas (analytical, teacher)
    # x 1 strategy (default)
    # x 1 style (standard)
    # x 2 rewrites (standard, elegant)
    # x 2 temperatures (0.5, 0.8)
    # x 1 top_p (0.9)
    # x 1 seed (42)
    # x 2 samples = 32 jobs total
    jobs = JobPlanner.plan(prompts, recipe)
    assert len(jobs) == 32
    
    # Check first job attributes
    first_job = jobs[0]
    assert first_job.prompt_id == "q1"
    assert first_job.persona == "analytical"
    assert first_job.rewrite == "standard"
    assert first_job.temperature == 0.5
    assert first_job.sample_index == 0


def test_prompt_composer(recipe_yaml_path):
    recipe = load_recipe(recipe_yaml_path)
    
    prompts = [
        PromptRecord.create("sys", "Calculate 5 + 5.", id="q1")
    ]
    
    jobs = JobPlanner.plan(prompts, recipe)
    
    # Filter a job to compose
    test_job = next(j for j in jobs if j.persona == "analytical" and j.rewrite == "elegant")
    
    composed_prompt = PromptComposer.compose(test_job)
    
    # Validate composed prompt record
    assert composed_prompt.id == test_job.job_id
    # Since rewrite is 'elegant', user prompt must be rewritten using the elegant template
    assert "shortcut" in composed_prompt.user_prompt
    assert "5 + 5" in composed_prompt.user_prompt
    
    # Since persona is 'analytical', the system prompt should have analytical instruction block
    assert "rigor" in composed_prompt.system_prompt.lower()
