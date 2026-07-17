#!/usr/bin/env python3
"""
Evaluation launch script for checking model capability benchmarks.
"""

import argparse
from ssp_framework.utils import setup_logger

logger = setup_logger("evaluation-launcher")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate models against benchmarks in the SSP Framework."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint directory or Hugging Face hub id.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["gsm8k", "math"],
        help="List of benchmarks to evaluate.",
    )
    args = parser.parse_args()

    logger.info(f"Starting evaluation of model: {args.model_path}")
    logger.info(f"Benchmarks target list: {args.benchmarks}")
    logger.warning("Evaluation pipeline is not yet implemented.")
    # TODO: Load model, build datasets, compute metrics.


if __name__ == "__main__":
    main()
