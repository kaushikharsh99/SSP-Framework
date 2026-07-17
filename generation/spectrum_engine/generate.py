#!/usr/bin/env python3
"""CLI entry point for the Spectrum Generation Engine.

Usage:
    python generate.py --provider api --config configs/api_openrouter.yaml
    python generate.py --provider vllm --config configs/vllm_local.yaml

No code changes are required when switching providers.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

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

logger = logging.getLogger("spectrum-engine.cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spectrum Generation Engine — Generate diverse reasoning trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python generate.py --provider api --config configs/api_openrouter.yaml
  python generate.py --provider vllm --config configs/vllm_local.yaml
  python generate.py --provider api --config configs/api_openrouter.yaml --resume
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
        help="Path to input dataset (JSONL). Overrides config if provided."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path. Overrides config if provided."
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Validate config and print plan without generating."
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config.logging)
    
    if args.dry_run:
        logger.info("Dry run mode — printing configuration and exiting.")
        logger.info(f"Provider: {args.provider}")
        logger.info(f"Model: {config.provider.model or config.provider.model_path}")
        logger.info(f"Sampling: temp={config.sampling.temperature}, n={config.sampling.n}")
        logger.info(f"Output: {config.storage.output_dir}")
        return
    
    logger.info("Spectrum Generation Engine starting...")
    logger.info(f"Provider: {args.provider}")
    
    # Implementation will wire up: Provider → Scheduler → Storage → Run
    logger.warning("Full pipeline not yet implemented. Use --dry-run to validate config.")


if __name__ == "__main__":
    asyncio.run(main())
