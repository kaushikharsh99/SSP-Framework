"""
Dataset registry to map dataset names to task types and preprocessing functions.
Allows flexible registration of new sources without altering ingestion logic.
"""

import logging
from typing import Callable, Dict, Any

from entry import DatasetEntry

logger = logging.getLogger("datasets-registry")


class DatasetRegistry:
    """Registry to keep track of active datasets and their preprocessors."""
    
    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, task_type: str) -> Callable:
        """Decorator to register a preprocessing function for a specific dataset.
        
        Args:
            name: The unique string name of the dataset.
            task_type: The domain class (e.g. 'math', 'code', 'reasoning').
        """
        def decorator(preprocess_fn: Callable[[Dict[str, Any], int], DatasetEntry]) -> Callable:
            cls._registry[name] = {
                "name": name,
                "task_type": task_type,
                "preprocess_fn": preprocess_fn,
            }
            logger.debug(f"Registered dataset '{name}' under task type '{task_type}'")
            return preprocess_fn
        return decorator

    @classmethod
    def get(cls, name: str) -> Dict[str, Any]:
        """Retrieves a registered dataset configuration by name.
        
        Args:
            name: String name of the registered dataset.
            
        Raises:
            KeyError if name is not found in the registry.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Dataset '{name}' is not registered. Registered options: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_registered(cls) -> Dict[str, str]:
        """Lists all registered datasets and their corresponding task types."""
        return {name: info["task_type"] for name, info in cls._registry.items()}


# --- Starter Registry Implementations ---

@DatasetRegistry.register(name="gsm8k", task_type="math")
def preprocess_gsm8k(raw_entry: Dict[str, Any], index: int) -> DatasetEntry:
    """Preprocess GSM8K raw records into DatasetEntry.
    
    Expected raw keys: 'question' or 'prompt', 'answer' or 'ground_truth'.
    """
    prompt = raw_entry.get("question", raw_entry.get("prompt", "")).strip()
    answer = raw_entry.get("answer", raw_entry.get("ground_truth", "")).strip()
    
    # GSM8K ground truth answers often look like "blah blah #### 42"
    # We can keep the full steps in metadata, but exact answer is usually after ####
    extracted_exact = answer
    if "####" in answer:
        extracted_exact = answer.split("####")[-1].strip()
        
    return DatasetEntry(
        id=raw_entry.get("id", f"gsm8k-{index}"),
        prompt=prompt,
        ground_truth_answer=extracted_exact,
        metadata={
            "full_answer_steps": answer,
            "index": index,
            "dataset": "gsm8k"
        }
    )


@DatasetRegistry.register(name="mbpp", task_type="code")
def preprocess_mbpp(raw_entry: Dict[str, Any], index: int) -> DatasetEntry:
    """Preprocess MBPP coding records into DatasetEntry.
    
    Expected raw keys: 'text' (prompt), 'code' (ground_truth), 'test_list' (test cases).
    """
    prompt = raw_entry.get("text", raw_entry.get("prompt", "")).strip()
    code = raw_entry.get("code", raw_entry.get("ground_truth", "")).strip()
    
    test_cases_raw = raw_entry.get("test_list", raw_entry.get("test_cases", []))
    test_cases = []
    for tc in test_cases_raw:
        if isinstance(tc, str):
            test_cases.append({"assertion": tc})
        elif isinstance(tc, dict):
            test_cases.append(tc)

    return DatasetEntry(
        id=raw_entry.get("id", raw_entry.get("task_id", f"mbpp-{index}")),
        prompt=prompt,
        ground_truth_answer=code,
        test_cases=test_cases,
        metadata={
            "index": index,
            "dataset": "mbpp"
        }
    )
