#!/usr/bin/env python3
"""
Script to convert checkpoints between different formats or base models.
"""

import argparse
from ssp_framework.utils import setup_logger

logger = setup_logger("convert-checkpoint")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert model checkpoint keys and formats."
    )
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="Path to source checkpoint.",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        required=True,
        help="Path to save converted checkpoint.",
    )
    args = parser.parse_args()

    logger.info(f"Targeting checkpoint conversion from {args.source_path} to {args.target_path}")
    logger.warning("Checkpoint conversion utility is not yet implemented.")
    # TODO: Implement parameter mapping and weights transpose/renaming logic.


if __name__ == "__main__":
    main()
