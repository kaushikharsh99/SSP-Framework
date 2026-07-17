"""
Unit tests for the Spectrum Generation Engine and diversity metrics.
"""

import os
import sys
import json
import tempfile
import pytest
import torch

# Add directories to the Python path to avoid name collisions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../models")))

from entry import DatasetEntry, Prompt, Response, Spectrum
from diversity import LexicalDiversityCalculator
from generator import HFSpectrumGenerator


# --- Mock Classes for Testing ---

class MockTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.vocab_size = 100

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            text = [text]
        input_ids = [[1, 2, 3, 4] for _ in text]
        attention_mask = [[1, 1, 1, 1] for _ in text]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask)
        }

    def encode(self, text, **kwargs):
        return self(text, **kwargs)

    def decode(self, token_ids, **kwargs):
        # Simulate generating tags
        return "<think>Let me count.</think> <answer>42</answer>"

    def batch_decode(self, sequences, **kwargs):
        return ["<think>Let me count.</think> <answer>42</answer>" for _ in sequences]

    def apply_chat_template(self, messages, **kwargs):
        return "mock formatted template"


class MockHFModel:
    def __init__(self):
        self.device = torch.device("cpu")

    def generate(self, input_ids, **kwargs):
        # Replicated input: [N, SeqLen]
        batch_size = input_ids.shape[0]
        gen_len = 5
        # Simulate generation outputs by appending integers
        gen_tokens = torch.randint(10, 50, (batch_size, gen_len))
        return torch.cat([input_ids, gen_tokens], dim=-1)


# --- Tests ---

def test_lexical_diversity_calculator():
    calculator = LexicalDiversityCalculator()

    # Create dummy responses
    responses = [
        Response(id="r1", prompt_id="p1", text="hello world", extracted_answer="42", token_ids=[1, 2]),
        Response(id="r2", prompt_id="p1", text="hello world", extracted_answer="42", token_ids=[1, 2]),
        Response(id="r3", prompt_id="p1", text="different text", extracted_answer="100", token_ids=[3, 4, 5]),
        Response(id="r4", prompt_id="p1", text="hello distinct world", extracted_answer="42", token_ids=[1, 6, 2])
    ]

    stats = calculator.calculate(responses)

    assert stats["num_total_responses"] == 4
    assert stats["num_unique_responses"] == 3
    assert stats["num_unique_answers"] == 2
    assert stats["duplicate_response_count"] == 1
    assert stats["token_count_min"] == 2
    assert stats["token_count_max"] == 3
    assert "avg_pairwise_lexical_similarity" in stats
    assert "diversity_score" in stats
    
    # Verify diversity score calculations
    assert 0.0 <= stats["diversity_score"] <= 1.0


def test_generator_single_prompt():
    mock_model = MockHFModel()
    mock_tokenizer = MockTokenizer()
    
    generator = HFSpectrumGenerator(
        model_name_or_path=mock_model,
        tokenizer_or_path=mock_tokenizer
    )

    prompt = Prompt(id="p1", system_prompt="Solve", user_query="2+2", formatted_prompt="Solve 2+2")
    
    # Generate a spectrum of size N=4
    spectrum = generator.generate_spectrum(prompt, num_trajectories=4)

    assert isinstance(spectrum, Spectrum)
    assert spectrum.prompt.id == "p1"
    assert len(spectrum.responses) == 4
    assert spectrum.diversity_statistics["num_total_responses"] == 4
    assert spectrum.responses[0].extracted_answer == "42"
    assert spectrum.responses[0].thinking_trace == "Let me count."
    assert "latency_seconds" in spectrum.generation_timing


def test_generator_deterministic_mode():
    mock_model = MockHFModel()
    mock_tokenizer = MockTokenizer()

    generator = HFSpectrumGenerator(
        model_name_or_path=mock_model,
        tokenizer_or_path=mock_tokenizer
    )

    prompt = Prompt(id="p1", system_prompt="Solve", user_query="2+2", formatted_prompt="Solve 2+2")
    
    # When temperature=0, it should configure do_sample=False
    spectrum = generator.generate_spectrum(prompt, num_trajectories=2, temperature=0.0)

    assert spectrum.generation_config["do_sample"] is False
    assert "temperature" not in spectrum.generation_config


def test_generator_jsonl_save():
    mock_model = MockHFModel()
    mock_tokenizer = MockTokenizer()

    generator = HFSpectrumGenerator(
        model_name_or_path=mock_model,
        tokenizer_or_path=mock_tokenizer
    )

    prompt = Prompt(id="p2", system_prompt="Solve", user_query="3+3", formatted_prompt="Solve 3+3")
    
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        temp_jsonl = f.name

    try:
        # Generate and save
        generator.generate_spectrum(prompt, num_trajectories=2, save_path=temp_jsonl)
        
        # Read back and parse
        with open(temp_jsonl, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["prompt"]["id"] == "p2"
        assert len(data["responses"]) == 2
        assert "diversity_statistics" in data
    finally:
        os.remove(temp_jsonl)
