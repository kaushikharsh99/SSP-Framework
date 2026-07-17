"""
Modular preprocessing steps for cleaning, validating, and normalizing dataset entries.
Each step is designed to be independently callable.
"""

import logging
import re
from typing import List, Set
from entry import DatasetEntry

logger = logging.getLogger("datasets-preprocessing")


def clean_text_whitespace(text: str) -> str:
    """Standardizes spaces, removes leading/trailing lines, and cleans tabs."""
    if not text:
        return ""
    # Standardize newline variations
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace sequences of spaces/tabs but keep lines intact
    text = re.sub(r"[ \t]+", " ", text)
    # Remove leading/trailing line whitespace and filter empty lines
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines).strip()


def validate_entry(entry: DatasetEntry) -> bool:
    """Performs integrity checks on a DatasetEntry record.
    
    Returns:
        True if the entry is valid, False otherwise.
    """
    if not entry.id or not isinstance(entry.id, str) or len(entry.id.strip()) == 0:
        logger.debug(f"Entry validation failed: Missing or invalid 'id'.")
        return False
        
    if not entry.prompt or not isinstance(entry.prompt, str) or len(entry.prompt.strip()) == 0:
        logger.debug(f"Entry ID {entry.id} validation failed: Empty or invalid 'prompt'.")
        return False
        
    if entry.ground_truth_answer is None or not isinstance(entry.ground_truth_answer, str):
        logger.debug(f"Entry ID {entry.id} validation failed: Missing 'ground_truth_answer'.")
        return False
        
    return True


def deduplicate_entries(entries: List[DatasetEntry]) -> List[DatasetEntry]:
    """Filters duplicate entries by ID to ensure unique records.
    
    Logs duplicates found.
    """
    seen_ids: Set[str] = set()
    deduplicated: List[DatasetEntry] = []
    
    for entry in entries:
        if entry.id in seen_ids:
            logger.warning(f"Removing duplicate sample ID: {entry.id}")
            continue
        seen_ids.add(entry.id)
        deduplicated.append(entry)
        
    return deduplicated


def normalize_math_string(text: str) -> str:
    """Helper to clean mathematical strings (e.g. stripping LaTeX artifacts).
    
    Converts LaTeX wrappers like $...$ or \\(...\\) to standard values.
    """
    if not text:
        return ""
    
    # Strip dollar signs
    text = text.replace("$", "").strip()
    
    # Strip LaTeX parentheses indicators
    text = text.replace(r"\(", "").replace(r"\)", "").strip()
    text = text.replace(r"\[", "").replace(r"\]", "").strip()
    
    # Standardize LaTeX structures (e.g., fractional representations or percentage signs)
    text = text.replace(r"\percent", "%").replace(r"\%", "%")
    
    return text.strip()


class IngestionPipeline:
    """A pipeline that sequences multiple independently callable preprocessing stages."""
    
    def __init__(self, apply_whitespace_clean: bool = True, apply_math_norm: bool = False):
        self.apply_whitespace_clean = apply_whitespace_clean
        self.apply_math_norm = apply_math_norm

    def process(self, entries: List[DatasetEntry]) -> List[DatasetEntry]:
        """Runs the sequential processing pipeline over a list of DatasetEntry objects."""
        processed_list: List[DatasetEntry] = []
        
        for entry in entries:
            # 1. Base Validation
            if not validate_entry(entry):
                continue
                
            # 2. Text Whitespace cleaning
            cleaned_prompt = clean_text_whitespace(entry.prompt)
            cleaned_answer = clean_text_whitespace(entry.ground_truth_answer)
            
            if self.apply_math_norm:
                cleaned_answer = normalize_math_string(cleaned_answer)
                
            # 3. Create updated entry
            updated_entry = DatasetEntry(
                id=entry.id,
                prompt=cleaned_prompt,
                ground_truth_answer=cleaned_answer,
                test_cases=entry.test_cases,
                metadata=entry.metadata
            )
            processed_list.append(updated_entry)
            
        # 4. Deduplicate
        return deduplicate_entries(processed_list)
