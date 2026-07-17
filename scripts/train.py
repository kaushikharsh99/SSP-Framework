#!/usr/bin/env python3
"""
Training launch script for the SSP Framework (Supervised SFT & RL).
"""

import argparse
from ssp_framework.utils import setup_logger

logger = setup_logger("training-launcher")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch training/alignment runs in the SSP Framework."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the Hydra or YAML configuration file.",
    )
    parser.add_argument(
        "--rl",
        action="store_true",
        help="Flag to launch Reinforcement Learning instead of Supervised SFT.",
    )
    args = parser.parse_args()

    logger.info(f"Loading experiment configuration from: {args.config}")
    if args.rl:
        logger.info("Setting up RL alignment run...")
        logger.warning("RL training pipeline is not yet implemented.")
    else:
        logger.info("Setting up Supervised Fine-Tuning (SFT) run...")
        logger.warning("SFT training pipeline is not yet implemented.")
        
    # TODO: Initialize model, dataset, optimizer, and run training loop.


if __name__ == "__main__":
    main()
