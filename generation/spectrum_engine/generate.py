from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import json
import hashlib
from typing import List

# Allow running this file directly: python generate.py
# by ensuring the parent directories are on sys.path
_engine_dir = os.path.dirname(os.path.abspath(__file__))
_generation_dir = os.path.dirname(_engine_dir)
_project_root = os.path.dirname(_generation_dir)
for _p in (_project_root, _generation_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generation.spectrum_engine.core.config import load_config
from generation.spectrum_engine.utils.logging import setup_logging
from generation.spectrum_engine.providers.api_provider import APIProvider
from generation.spectrum_engine.providers.vllm_provider import VLLMProvider
from generation.spectrum_engine.storage.jsonl_storage import JSONLStorage
from generation.spectrum_engine.storage.checkpoint import CheckpointManager
from generation.spectrum_engine.scheduler.batch_scheduler import BatchScheduler
from generation.spectrum_engine.core.types import PromptRecord

logger = logging.getLogger("spectrum-engine.cli")


def load_dataset_prompts(path: str, system_prompt: str) -> List[PromptRecord]:
    """Helper to load JSON/JSONL datasets and convert entries to PromptRecord objects."""
    prompts = []
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []

        # 1. Try parsing as single JSON list of items
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    user_prompt = item.get("problem") or item.get("query") or item.get("user_prompt") or ""
                    prompt_id = str(item.get("id") or item.get("task_id") or f"prompt-{idx}")
                    metadata = {k: v for k, v in item.items() if k not in ("problem", "query", "user_prompt", "id", "task_id")}
                    prompts.append(PromptRecord(
                        id=prompt_id,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        metadata=metadata
                    ))
                return prompts
        except json.JSONDecodeError:
            pass  # Fallback to JSONL format

        # 2. Parse line-by-line JSONL format
        lines = content.split("\n")
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                user_prompt = item.get("problem") or item.get("query") or item.get("user_prompt") or ""
                prompt_id = str(item.get("id") or item.get("task_id") or f"prompt-{idx}")
                metadata = {k: v for k, v in item.items() if k not in ("problem", "query", "user_prompt", "id", "task_id")}
                prompts.append(PromptRecord(
                    id=prompt_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    metadata=metadata
                ))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {idx + 1} in dataset: {e}")

    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spectrum Generation Engine — Generate diverse reasoning trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python generate.py --provider api --config configs/api_openrouter.yaml --dataset problems.jsonl
  python generate.py --provider vllm --config configs/vllm_local.yaml --dataset math.json
  python generate.py --experiment configs/experiment_ablation.yaml
  python generate.py --experiment configs/experiment_ablation.yaml --dataset problems_subset.jsonl
"""
    )
    parser.add_argument(
        "--provider", type=str, required=False, choices=["api", "vllm"],
        help="Generation provider to use (required unless running an experiment)."
    )
    parser.add_argument(
        "--config", type=str, required=False,
        help="Path to YAML configuration file (required unless running an experiment)."
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Path to experiment configuration YAML file."
    )
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="Resume from the last checkpoint."
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to input dataset (JSON/JSONL). Overrides config if provided."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL file path (optional)."
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Validate config and print plan without generating."
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    
    # Check if running an experiment
    if args.experiment:
        from generation.spectrum_engine.core.experiment import load_experiment_config, ExperimentRunner
        
        # Load experiment config
        try:
            exp_config = load_experiment_config(args.experiment)
        except Exception as e:
            print(f"Error loading experiment configuration: {e}")
            sys.exit(1)
            
        # Initialize logging using referenced provider config
        provider_config = load_config(exp_config.provider_config_path)
        setup_logging(provider_config.logging)
        
        logger.info(f"Running Experiment: {exp_config.name}")
        
        # Resolve dataset path
        dataset_path = args.dataset or exp_config.dataset_path
        if not dataset_path:
            logger.error("Dataset path must be specified either in the experiment config or via --dataset.")
            sys.exit(1)
            
        # Load prompts
        try:
            prompts = load_dataset_prompts(dataset_path, provider_config.prompts.system_prompt)
            logger.info(f"Loaded {len(prompts)} prompts from dataset: {dataset_path}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            sys.exit(1)
            
        if args.dry_run:
            logger.info("Dry run complete — configuration and dataset validated successfully.")
            logger.info(f"Recipes to run: {list(exp_config.recipes.keys())}")
            return
            
        # Run each recipe
        runner = ExperimentRunner(exp_config)
        for recipe_name in exp_config.recipes:
            logger.info(f"Starting execution of recipe: {recipe_name}")
            try:
                recipe_summary = await runner.run_recipe(recipe_name, prompts)
                logger.info(f"Recipe '{recipe_name}' completed. Summary:")
                logger.info(f"  Overall Correctness: {recipe_summary.get('overall_trajectory_correctness')}")
                logger.info(f"  Average Diversity: {recipe_summary.get('average_lexical_diversity')}")
                
                # Print leaderboard
                logger.info("  Persona Leaderboard:")
                for entry in recipe_summary.get("persona_leaderboard", []):
                    logger.info(
                        f"    - {entry['persona']}: Correctness={entry['correctness']} "
                        f"({entry['correct_count']}/{entry['total_count']}), "
                        f"Avg Tokens={entry['avg_tokens']}"
                    )
            except Exception as e:
                logger.error(f"Failed to execute recipe '{recipe_name}': {e}", exc_info=True)
                
        logger.info("All experiment recipes executed.")
        return

    # Standard run path
    if not args.provider or not args.config:
        print("Error: --provider and --config are required when not running an experiment.")
        sys.exit(1)

    # 1. Load configuration
    config = load_config(args.config)
    
    # 2. Setup logging
    setup_logging(config.logging)
    
    # 3. Handle dry run mode
    if args.dry_run:
        logger.info("Dry run mode — printing configuration and exiting.")
        logger.info(f"Provider: {args.provider}")
        logger.info(f"Model: {config.provider.model or config.provider.model_path}")
        logger.info(f"Sampling: temp={config.sampling.temperature}, n={config.sampling.n}")
        logger.info(f"Output Directory: {config.storage.output_dir}")
        if args.dataset:
            try:
                prompts = load_dataset_prompts(args.dataset, config.prompts.system_prompt)
                logger.info(f"Dataset successfully loaded. Number of prompts: {len(prompts)}")
            except Exception as e:
                logger.error(f"Failed to parse dataset in dry-run mode: {e}")
        return

    # 4. Require input dataset for execution runs
    if not args.dataset:
        logger.error("Dataset parameter is required for execution. Please specify --dataset <path>")
        sys.exit(1)

    # 5. Load prompts from dataset
    try:
        prompts = load_dataset_prompts(args.dataset, config.prompts.system_prompt)
        logger.info(f"Loaded {len(prompts)} prompts from dataset: {args.dataset}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # 6. Instantiate pipeline components
    if args.provider == "api":
        provider = APIProvider(config.provider)
    else:
        provider = VLLMProvider(config.provider)

    storage = JSONLStorage(config.storage, output_path=args.output)
    
    # Deduce a job ID from the model name/path to isolate checkpoint files
    model_name = config.provider.model or config.provider.model_path or "default"
    job_hash = hashlib.md5(model_name.encode("utf-8")).hexdigest()[:8]
    job_id = f"{args.provider}_{job_hash}"
    
    checkpoint = CheckpointManager(config.storage.checkpoint_dir, job_id=job_id)
    scheduler = BatchScheduler(config.scheduler)

    # 7. Execute the generation loop
    logger.info("Spectrum Generation Engine starting...")
    if args.resume:
        checkpoint.load()

    async with provider, storage:
        report = await scheduler.run(
            prompts=prompts,
            provider=provider,
            storage=storage,
            checkpoint=checkpoint,
            sampling=config.sampling
        )

    logger.info(f"Generation run completed.\n{report.total_prompts} total prompts, {report.completed_prompts} successfully generated.")


if __name__ == "__main__":
    asyncio.run(main())
