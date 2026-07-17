"""
Dataset loading utilities for JSON, JSONL, and Hugging Face sources.
Standardizes external formats using the DatasetRegistry.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from entry import DatasetEntry
from registry import DatasetRegistry

logger = logging.getLogger("datasets-loader")


class DatasetLoader:
    """Loads and normalizes raw dataset records from file paths or Hugging Face hubs."""
    
    @staticmethod
    def load_from_json(file_path: str, dataset_name: str) -> List[DatasetEntry]:
        """Loads entries from a standard JSON file containing a list of objects.
        
        Args:
            file_path: Absolute or relative file path.
            dataset_name: Name matching registry key.
        """
        logger.info(f"Loading JSON dataset '{dataset_name}' from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        if not isinstance(raw_data, list):
            raise ValueError(f"JSON file must contain a list of objects. Got: {type(raw_data)}")
            
        return DatasetLoader._preprocess_batch(raw_data, dataset_name)

    @staticmethod
    def load_from_jsonl(file_path: str, dataset_name: str) -> List[DatasetEntry]:
        """Loads entries from a line-separated JSONL file.
        
        Args:
            file_path: Absolute or relative file path.
            dataset_name: Name matching registry key.
        """
        logger.info(f"Loading JSONL dataset '{dataset_name}' from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        raw_data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON line {line_idx + 1}: {e}")
                    
        return DatasetLoader._preprocess_batch(raw_data, dataset_name)

    @staticmethod
    def load_from_hf(
        path: str, 
        dataset_name: str, 
        name: Optional[str] = None, 
        split: str = "train",
        **kwargs: Any
    ) -> List[DatasetEntry]:
        """Loads entries from Hugging Face Datasets Hub.
        
        Args:
            path: HF dataset repository path (e.g. 'gsm8k').
            dataset_name: Registry name mapping to preprocess function (e.g. 'gsm8k').
            name: Optional sub-dataset configuration name.
            split: Target dataset split (e.g. 'train', 'test').
            kwargs: Extra parameters passed to HF load_dataset.
        """
        logger.info(f"Loading HF dataset '{dataset_name}' from hub path '{path}' split '{split}'")
        try:
            from datasets import load_dataset as hf_load_dataset
        except ImportError as e:
            logger.error("Hugging Face 'datasets' library is not installed. Run 'pip install datasets'.")
            raise e

        # Load HF dataset
        hf_dataset = hf_load_dataset(path, name=name, split=split, **kwargs)
        
        # Convert HF Dataset records to list of dicts
        raw_data = [dict(record) for record in hf_dataset]
        return DatasetLoader._preprocess_batch(raw_data, dataset_name)

    @staticmethod
    def _preprocess_batch(raw_data: List[Dict[str, Any]], dataset_name: str) -> List[DatasetEntry]:
        """Applies the registered preprocessing function to a batch of raw records."""
        config = DatasetRegistry.get(dataset_name)
        preprocess_fn = config["preprocess_fn"]
        
        entries: List[DatasetEntry] = []
        duplicate_ids = set()
        malformed_count = 0
        
        for idx, raw_record in enumerate(raw_data):
            try:
                entry = preprocess_fn(raw_record, idx)
                
                # Check for key field validations
                if not entry.id or not entry.prompt or not entry.ground_truth_answer:
                    logger.warning(f"Sample at index {idx} failed core validation (empty ID/prompt/answer). Skipping.")
                    malformed_count += 1
                    continue
                    
                if entry.id in duplicate_ids:
                    logger.warning(f"Duplicate entry ID detected: {entry.id}. Skipping.")
                    continue
                    
                duplicate_ids.add(entry.id)
                entries.append(entry)
            except Exception as e:
                logger.warning(f"Error preprocessing sample at index {idx}: {e}. Skipping.")
                malformed_count += 1
                
        logger.info(
            f"Dataset Ingestion complete for '{dataset_name}'. "
            f"Loaded: {len(entries)} samples. "
            f"Skipped (malformed/errors): {malformed_count}. "
            f"Skipped (duplicates): {len(raw_data) - len(entries) - malformed_count}."
        )
        return entries
