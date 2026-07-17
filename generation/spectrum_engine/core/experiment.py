"""Experiment Runner for optimizing reasoning prompts and generation settings.
Enables running multiple ablation recipes systematically.
"""

import time
import os
import sys
import json
import logging
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from .types import PromptRecord, ResponseRecord, SamplingConfig, Spectrum, ProviderInfo, GenerationReport
from .config import load_config, EngineConfig
from ..providers.api_provider import APIProvider
from ..providers.vllm_provider import VLLMProvider
from ..storage.jsonl_storage import JSONLStorage
from ..storage.checkpoint import CheckpointManager
from ..prompts.registry import build_combined_system_prompt
from .diversity import LexicalDiversityCalculator

# Resolve the datasets folder dynamically to import local verification modules
_current_dir = os.path.dirname(os.path.abspath(__file__))
_datasets_dir = os.path.abspath(os.path.join(_current_dir, "../../../datasets"))
if _datasets_dir not in sys.path:
    sys.path.insert(0, _datasets_dir)

try:
    from verifiers import VerifierRegistry
except ImportError:
    VerifierRegistry = None

logger = logging.getLogger("spectrum-engine.experiment")


@dataclass
class RecipeConfig:
    """Configuration for a single generation parameter recipe."""
    personas: List[str]
    styles: List[str]
    temperatures: List[float]
    seeds: List[int]
    samples_per_configuration: int = 1


@dataclass
class ExperimentConfig:
    """Configuration for an end-to-end ablation experiment."""
    name: str
    provider_config_path: str
    dataset_path: str
    output_dir: str = "./outputs/experiments"
    recipes: Dict[str, RecipeConfig] = field(default_factory=dict)
    verifier_type: str = "math"  # 'math' or 'code'


def load_experiment_config(path: str) -> ExperimentConfig:
    """Loads and parses an ExperimentConfig from a YAML file."""
    import yaml
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Experiment config file not found: {path}")
        
    config_dir = os.path.dirname(os.path.abspath(path))
    
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
        
    exp_data = raw.get("experiment", {})
    recipes_data = exp_data.get("recipes", {})
    
    # Resolve relative paths relative to config file directory
    provider_config_path = exp_data.get("provider_config_path", "")
    if provider_config_path and not os.path.isabs(provider_config_path):
        provider_config_path = os.path.abspath(os.path.join(config_dir, provider_config_path))
        
    dataset_path = exp_data.get("dataset_path", "")
    if dataset_path and not os.path.isabs(dataset_path):
        dataset_path = os.path.abspath(os.path.join(config_dir, dataset_path))
        
    output_dir = exp_data.get("output_dir", "./outputs/experiments")
    if output_dir and not os.path.isabs(output_dir):
        output_dir = os.path.abspath(os.path.join(config_dir, output_dir))
    
    recipes = {}
    for name, r_data in recipes_data.items():
        recipes[name] = RecipeConfig(
            personas=r_data.get("personas", []),
            styles=r_data.get("styles", []),
            temperatures=r_data.get("temperatures", [1.0]),
            seeds=r_data.get("seeds", [None]),
            samples_per_configuration=r_data.get("samples_per_configuration", 1)
        )
        
    return ExperimentConfig(
        name=exp_data.get("name", "experiment"),
        provider_config_path=provider_config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        recipes=recipes,
        verifier_type=exp_data.get("verifier_type", "math")
    )


class ExperimentRunner:
    """Executes multi-recipe experiments and evaluates results."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.base_engine_config = load_config(self.config.provider_config_path)
        self.diversity_calculator = LexicalDiversityCalculator()

    async def run_recipe(self, recipe_name: str, prompts: List[PromptRecord]) -> Dict[str, Any]:
        """Runs generation and verification for a single recipe, writing outputs to disk."""
        recipe = self.config.recipes.get(recipe_name)
        if not recipe:
            raise ValueError(f"Recipe '{recipe_name}' not defined in experiment configuration.")

        recipe_dir = os.path.join(self.config.output_dir, self.config.name, recipe_name)
        os.makedirs(recipe_dir, exist_ok=True)
        
        # Setup specific storage and checkpoints
        from ..core.config import StorageConfig
        storage_config = StorageConfig(
            format="jsonl",
            output_dir=recipe_dir,
            filename_template="spectra.jsonl",
            flush_interval=1,
            checkpoint_dir=os.path.join(recipe_dir, ".checkpoints")
        )
        
        checkpoint = CheckpointManager(storage_config.checkpoint_dir, job_id=recipe_name)
        checkpoint.load()

        storage = JSONLStorage(storage_config, output_path=os.path.join(recipe_dir, "spectra.jsonl"))
        
        # Instantiate provider
        if self.base_engine_config.provider.type == "api":
            provider = APIProvider(self.base_engine_config.provider)
        else:
            provider = VLLMProvider(self.base_engine_config.provider)

        # Load verification module if available
        verifier = None
        if VerifierRegistry:
            try:
                verifier = VerifierRegistry.get(self.config.verifier_type)
                logger.info(f"Verifier '{self.config.verifier_type}' loaded successfully for evaluation.")
            except Exception as e:
                logger.warning(f"Failed to load verifier registry: {e}. Correctness checks will be bypassed.")

        # Compute all Cartesian combinations of prompts, personas, styles, temps, seeds
        combos = list(itertools.product(recipe.personas, recipe.styles, recipe.temperatures, recipe.seeds))
        logger.info(f"Starting recipe '{recipe_name}'. Configurations: {len(combos)} per prompt.")

        all_spectra: List[Spectrum] = []

        async with provider, storage:
            for prompt_idx, base_prompt in enumerate(prompts):
                if checkpoint.is_completed(base_prompt.id):
                    logger.info(f"Skipping completed prompt ID: {base_prompt.id}")
                    continue
                
                logger.info(f"Processing prompt {prompt_idx + 1}/{len(prompts)}: {base_prompt.id}")
                all_responses: List[ResponseRecord] = []
                
                # Execute combinations sequentially or concurrently. To respect API rate limits, we execute combos sequentially.
                for persona, style, temp, seed in combos:
                    logger.debug(f"Executing combo: persona={persona}, style={style}, temp={temp}, seed={seed}")
                    
                    system_prompt = build_combined_system_prompt(persona, style)
                    
                    config_prompt = PromptRecord(
                        id=f"{base_prompt.id}_{persona}_{style}_t{temp}_s{seed}",
                        system_prompt=system_prompt,
                        user_prompt=base_prompt.user_prompt,
                        metadata={
                            **base_prompt.metadata,
                            "persona": persona,
                            "style": style,
                            "temperature": temp,
                            "seed": seed
                        }
                    )
                    
                    config_sampling = SamplingConfig(
                        temperature=temp,
                        seed=seed,
                        n=recipe.samples_per_configuration,
                        top_p=self.base_engine_config.sampling.top_p,
                        top_k=self.base_engine_config.sampling.top_k,
                        min_p=self.base_engine_config.sampling.min_p,
                        max_tokens=self.base_engine_config.sampling.max_tokens,
                        stop_sequences=self.base_engine_config.sampling.stop_sequences,
                        repetition_penalty=self.base_engine_config.sampling.repetition_penalty
                    )
                    
                    try:
                        config_spectra = await provider.generate([config_prompt], config_sampling)
                        if config_spectra:
                            responses = config_spectra[0].responses
                            
                            # Attach config metadata and verify responses
                            for resp in responses:
                                resp.metadata["persona"] = persona
                                resp.metadata["style"] = style
                                resp.metadata["temperature"] = temp
                                resp.metadata["seed"] = seed
                                
                                if verifier:
                                    ground_truth_str = base_prompt.metadata.get("answer") or base_prompt.metadata.get("ground_truth") or ""
                                    test_cases = base_prompt.metadata.get("test_cases") or []
                                    
                                    try:
                                        from entry import DatasetEntry as RealDatasetEntry
                                        entry = RealDatasetEntry(
                                            id=base_prompt.id,
                                            prompt=base_prompt.user_prompt,
                                            ground_truth_answer=ground_truth_str,
                                            test_cases=test_cases
                                        )
                                    except ImportError:
                                        class DummyEntry:
                                            def __init__(self, ans, tests):
                                                self.ground_truth_answer = ans
                                                self.test_cases = tests
                                        entry = DummyEntry(ground_truth_str, test_cases)
                                        
                                    verify_res = verifier.verify(resp, entry)
                                    resp.metadata["is_correct"] = verify_res.is_correct
                                    resp.metadata["verify_error"] = verify_res.error_message
                                
                            all_responses.extend(responses)
                    except Exception as e:
                        logger.error(f"Failed combo {persona}/{style}/t={temp}/s={seed}: {e}")

                if all_responses:
                    # Construct single unified Spectrum representing the diverse responses for this prompt
                    spectrum = Spectrum(
                        prompt=base_prompt,
                        responses=all_responses,
                        sampling_config=self.base_engine_config.sampling,
                        provider_info=provider.info(),
                        created_at=time.time(),
                        metadata={
                            "experiment": self.config.name,
                            "recipe": recipe_name
                        }
                    )
                    
                    # Compute diversity stats across all responses
                    spectrum.diversity_statistics = self.diversity_calculator.calculate(all_responses)
                    
                    # Store spectrum and update checkpoint
                    await storage.write(spectrum)
                    checkpoint.mark_completed(base_prompt.id)
                    checkpoint.save()
                    all_spectra.append(spectrum)

        # Compute summary stats and persona leaderboard
        summary = self._compute_recipe_summary(recipe_name, all_spectra)
        
        summary_path = os.path.join(recipe_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Summary report for recipe '{recipe_name}' written to: {summary_path}")
        return summary

    def _compute_recipe_summary(self, recipe_name: str, spectra: List[Spectrum]) -> Dict[str, Any]:
        """Compute aggregate stats and leaderboard ranking of personas."""
        if not spectra:
            return {"recipe": recipe_name, "num_completed_prompts": 0}

        total_trajectories = 0
        correct_trajectories = 0
        tokens_by_persona = {}
        correct_by_persona = {}
        total_by_persona = {}
        
        for spec in spectra:
            for resp in spec.responses:
                total_trajectories += 1
                is_correct = resp.metadata.get("is_correct")
                if is_correct is True:
                    correct_trajectories += 1
                    
                persona = resp.metadata.get("persona", "unknown")
                tokens_by_persona.setdefault(persona, []).append(resp.token_count)
                
                total_by_persona[persona] = total_by_persona.get(persona, 0) + 1
                if is_correct is True:
                    correct_by_persona[persona] = correct_by_persona.get(persona, 0) + 1

        leaderboard = []
        for persona in total_by_persona:
            tot = total_by_persona[persona]
            corr = correct_by_persona.get(persona, 0)
            correctness_rate = corr / tot if tot > 0 else 0.0
            avg_tokens = sum(tokens_by_persona[persona]) / len(tokens_by_persona[persona]) if tokens_by_persona[persona] else 0.0
            
            leaderboard.append({
                "persona": persona,
                "correctness": f"{correctness_rate:.1%}",
                "correct_count": corr,
                "total_count": tot,
                "avg_tokens": int(avg_tokens)
            })

        diversity_scores = [s.diversity_statistics.get("diversity_score", 0.0) for s in spectra if s.diversity_statistics]
        avg_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0

        return {
            "recipe": recipe_name,
            "experiment": self.config.name,
            "num_completed_prompts": len(spectra),
            "total_generated_trajectories": total_trajectories,
            "overall_trajectory_correctness": f"{correct_trajectories / total_trajectories:.1%}" if total_trajectories > 0 else "0.0%",
            "average_lexical_diversity": f"{avg_diversity:.3f}",
            "persona_leaderboard": leaderboard
        }
