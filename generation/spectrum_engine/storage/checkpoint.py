"""Checkpoint manager for resume support.

Tracks which prompt IDs have been successfully generated and stored.
Enables the scheduler to skip completed work after interruptions.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Set

logger = logging.getLogger("spectrum-engine.storage.checkpoint")


class CheckpointManager:
    """Tracks completed prompt IDs for generation resume."""

    def __init__(self, checkpoint_dir: str, job_id: str = "default"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.job_id = job_id
        self._completed: Set[str] = set()
        self._checkpoint_path = self.checkpoint_dir / f"{job_id}.checkpoint.json"

    def load(self) -> None:
        """Load checkpoint state from disk."""
        if self._checkpoint_path.exists():
            with open(self._checkpoint_path, "r") as f:
                data = json.load(f)
                self._completed = set(data.get("completed_ids", []))
            logger.info(f"Loaded checkpoint: {len(self._completed)} completed prompts.")
        else:
            logger.info("No existing checkpoint found. Starting fresh.")

    def save(self) -> None:
        """Persist checkpoint state to disk."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(self._checkpoint_path, "w") as f:
            json.dump({"completed_ids": list(self._completed), "job_id": self.job_id}, f)
        logger.debug(f"Checkpoint saved: {len(self._completed)} completed prompts.")

    def mark_completed(self, prompt_id: str) -> None:
        """Mark a prompt as successfully completed."""
        self._completed.add(prompt_id)

    def is_completed(self, prompt_id: str) -> bool:
        """Check if a prompt has already been completed."""
        return prompt_id in self._completed

    def get_completed_ids(self) -> Set[str]:
        """Return all completed prompt IDs."""
        return self._completed.copy()

    @property
    def completed_count(self) -> int:
        return len(self._completed)
