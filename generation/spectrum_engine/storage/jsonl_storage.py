"""JSONL storage backend for the Spectrum Generation Engine.

Writes one Spectrum per line in JSON Lines format.
Supports append mode, periodic flushing, and resume via ID scanning.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Set

from ..core.config import StorageConfig
from ..core.types import Spectrum
from .base import BaseStorage

logger = logging.getLogger("spectrum-engine.storage.jsonl")


class JSONLStorage(BaseStorage):
    """Append-mode JSONL storage with auto-flush and resume support."""

    def __init__(self, config: StorageConfig, output_path: Optional[str] = None):
        self.config = config
        self._file = None
        self._write_count = 0
        
        if output_path:
            self._output_path = output_path
        else:
            out_dir = Path(self.config.output_dir)
            timestamp = int(time.time())
            filename = self.config.filename_template.format(timestamp=timestamp)
            self._output_path = str(out_dir / filename)

    async def _open_file(self) -> None:
        """Helper to lazily open the target output file."""
        if not self._file:
            path = Path(self._output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._output_path, "a", encoding="utf-8")
            logger.info(f"Opened JSONL storage file: {self._output_path}")

    async def write(self, spectrum: Spectrum) -> None:
        """Serialize and append a spectrum to the JSONL file."""
        await self._open_file()
        
        serialized = dataclasses.asdict(spectrum)
        self._file.write(json.dumps(serialized) + "\n")
        self._write_count += 1

        # Periodic flush based on configured interval
        if self._write_count % self.config.flush_interval == 0:
            await self.flush()

    async def flush(self) -> None:
        """Force flush the file buffer to disk."""
        if self._file:
            self._file.flush()
            try:
                os.fsync(self._file.fileno())
            except Exception as e:
                logger.warning(f"Failed to fsync JSONL file: {e}")

    def get_completed_ids(self) -> Set[str]:
        """Scan existing JSONL target file and all other JSONL files in output directory for completed prompt IDs."""
        completed_ids = set()
        
        # 1. Scan current output file if it exists
        if self._output_path and os.path.exists(self._output_path):
            completed_ids.update(self._scan_file_for_ids(self._output_path))
            
        # 2. Scan other JSONL files in the output directory
        out_dir = Path(self.config.output_dir)
        if out_dir.exists():
            for filepath in out_dir.glob("*.jsonl"):
                full_path = str(filepath.resolve())
                target_resolved = str(Path(self._output_path).resolve() if self._output_path else "")
                if full_path != target_resolved:
                    completed_ids.update(self._scan_file_for_ids(full_path))
                    
        return completed_ids

    def _scan_file_for_ids(self, filepath: str) -> Set[str]:
        """Helper scanner to extract prompt IDs from a single JSONL file."""
        ids = set()
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "prompt" in data and "id" in data["prompt"]:
                            ids.add(data["prompt"]["id"])
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Error scanning file {filepath} for completed IDs: {e}")
        return ids

    async def close(self) -> None:
        """Flush and close the file handle."""
        if self._file:
            await self.flush()
            self._file.close()
            self._file = None
            logger.info(f"Closed JSONL storage file: {self._output_path}")
