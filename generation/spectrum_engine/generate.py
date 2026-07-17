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
  python generate.py --provider api --config configs/api_openrouter.yaml --dataset problems.jsonl --resume
"""
    )
    parser.add_argument(
        "--provider", type=str, required=True, choices=["api", "vllm"],
        help="Generation provider to use."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file."
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
