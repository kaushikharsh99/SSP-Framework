"""
Unit tests for the datasets module: registry, loader, preprocessing, prompt builder, and tokenizer wrapper.
"""

import os
import sys
import json
import tempfile
import pytest
import torch

# Add the datasets directory to the Python path to avoid name collisions with site-packages datasets
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets")))

from entry import DatasetEntry, Prompt
from registry import DatasetRegistry
from loader import DatasetLoader
from preprocessing import clean_text_whitespace, validate_entry, deduplicate_entries, normalize_math_string, IngestionPipeline
from prompt_builder import PromptBuilder
from tokenizer import TokenizerWrapper


# --- Mock Tokenizer for Offline Testing ---
class MockTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.vocab_size = 10
        
    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            text = [text]
        input_ids = [[2, 3, 4] for _ in text]
        attention_mask = [[1, 1, 1] for _ in text]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask)
        }

    def decode(self, token_ids, **kwargs):
        return "mock decoded text"

    def batch_decode(self, sequences, **kwargs):
        return ["mock decoded text" for _ in sequences]

    def apply_chat_template(self, messages, **kwargs):
        # Mock chat template format
        formatted = ""
        for msg in messages:
            formatted += f"[{msg['role']}]: {msg['content']}\n"
        return formatted.strip()


# --- Tests ---

def test_dataset_registry():
    # Verify that gsm8k and mbpp are registered
    registered = DatasetRegistry.list_registered()
    assert "gsm8k" in registered
    assert "mbpp" in registered
    assert registered["gsm8k"] == "math"
    assert registered["mbpp"] == "code"

    # Verify retrieval
    config = DatasetRegistry.get("gsm8k")
    assert config["name"] == "gsm8k"
    assert callable(config["preprocess_fn"])


def test_preprocessing_utilities():
    # 1. Whitespace clean
    dirty_text = "  hello   world \n\n new   line   "
    assert clean_text_whitespace(dirty_text) == "hello world\nnew line"

    # 2. Math normalization
    latex_math = "$x^2 + y = 3$"
    assert normalize_math_string(latex_math) == "x^2 + y = 3"
    
    latex_paren = r"\( \frac{a}{b} \)"
    assert normalize_math_string(latex_paren) == r"\frac{a}{b}"

    # 3. Entry Validation
    valid_entry = DatasetEntry(id="1", prompt="What is 2+2?", ground_truth_answer="4")
    assert validate_entry(valid_entry) is True

    invalid_entry = DatasetEntry(id="", prompt="What is 2+2?", ground_truth_answer="4")
    assert validate_entry(invalid_entry) is False


def test_dataset_loader_json_jsonl():
    # Prepare dummy data
    raw_data = [
        {"question": "What is 2+2?", "answer": "#### 4", "id": "q1"},
        {"question": "What is 3+3?", "answer": "#### 6", "id": "q2"},
        {"question": "Duplicate ID Test", "answer": "#### 6", "id": "q2"},  # Duplicate
        {"question": "", "answer": "#### 5", "id": "q3"},                   # Invalid prompt
    ]

    # Test JSON Loading
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(raw_data, f)
        temp_json = f.name

    try:
        entries = DatasetLoader.load_from_json(temp_json, "gsm8k")
        # Should load 2 valid entries (skips duplicate q2 and empty prompt q3)
        assert len(entries) == 2
        assert entries[0].id == "q1"
        assert entries[0].prompt == "What is 2+2?"
        assert entries[0].ground_truth_answer == "4"
        assert entries[1].id == "q2"
        assert entries[1].ground_truth_answer == "6"
    finally:
        os.remove(temp_json)

    # Test JSONL Loading
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        for item in raw_data:
            f.write(json.dumps(item) + "\n")
        temp_jsonl = f.name

    try:
        entries = DatasetLoader.load_from_jsonl(temp_jsonl, "gsm8k")
        assert len(entries) == 2
    finally:
        os.remove(temp_jsonl)


def test_ingestion_pipeline():
    entries = [
        DatasetEntry(id="e1", prompt="  hello   world  ", ground_truth_answer=" $42$ "),
        DatasetEntry(id="e2", prompt="  test prompt  ", ground_truth_answer="  code  "),
        DatasetEntry(id="e1", prompt="duplicate", ground_truth_answer="dup"),
    ]
    
    pipeline = IngestionPipeline(apply_whitespace_clean=True, apply_math_norm=True)
    processed = pipeline.process(entries)
    
    # Should validate, clean whitespace, normalize math, and deduplicate
    assert len(processed) == 2
    assert processed[0].id == "e1"
    assert processed[0].prompt == "hello world"
    assert processed[0].ground_truth_answer == "42"
    assert processed[1].prompt == "test prompt"


def test_prompt_builder():
    entry = DatasetEntry(id="test-1", prompt="2+2", ground_truth_answer="4")
    builder = PromptBuilder(
        system_prompt="Solve math problems.",
        user_template="Problem: {prompt}"
    )

    # 1. Test fallback without tokenizer
    prompt_fallback = builder.build(entry)
    assert prompt_fallback.id == "test-1"
    assert prompt_fallback.system_prompt == "Solve math problems."
    assert prompt_fallback.user_query == "Problem: 2+2"
    assert "system\nSolve math problems." in prompt_fallback.formatted_prompt
    assert "user\nProblem: 2+2" in prompt_fallback.formatted_prompt

    # 2. Test template with mock tokenizer
    mock_tok = MockTokenizer()
    prompt_tok = builder.build(entry, tokenizer=mock_tok)
    assert "[system]: Solve math problems." in prompt_tok.formatted_prompt
    assert "[user]: Problem: 2+2" in prompt_tok.formatted_prompt


def test_tokenizer_wrapper():
    mock_tok = MockTokenizer()
    wrapper = TokenizerWrapper(tokenizer_or_path=mock_tok, max_length=128)

    assert wrapper.pad_token_id == 0
    assert wrapper.eos_token_id == 1
    assert wrapper.vocab_size == 10

    # Test encode
    encoded = wrapper.encode("hello world")
    assert "input_ids" in encoded
    assert "attention_mask" in encoded
    assert isinstance(encoded["input_ids"], torch.Tensor)

    # Test decode
    decoded = wrapper.decode([2, 3, 4])
    assert decoded == "mock decoded text"
