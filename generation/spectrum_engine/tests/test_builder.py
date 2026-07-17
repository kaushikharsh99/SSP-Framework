"""Unit tests for the SpectrumBuilder module.
"""

import pytest
from generation.spectrum_engine.core.builder import SpectrumBuilder
from generation.spectrum_engine.core.types import PromptRecord, ResponseRecord, SamplingConfig, ProviderInfo
from generation.spectrum_engine.core.diversity import LexicalDiversityCalculator


def test_spectrum_builder_assembly():
    prompt = PromptRecord.create("sys", "Answer is 10 + 10.", id="q-1", answer="20")
    
    responses = [
        ResponseRecord(
            id="r-1",
            prompt_id="q-1",
            text="<think>Add</think>\n<answer>20</answer>",
            token_count=5,
            finish_reason="stop"
        ),
        ResponseRecord(
            id="r-2",
            prompt_id="q-1",
            text="<think>Add</think>\n<answer>30</answer>",
            token_count=5,
            finish_reason="stop"
        )
    ]
    
    sampling = SamplingConfig(temperature=0.8)
    provider = ProviderInfo(name="test_prov", backend="test", model="test-model")
    div_calculator = LexicalDiversityCalculator()
    
    spectrum = SpectrumBuilder.build(
        prompt=prompt,
        responses=responses,
        verifier_type="math",
        sampling_config=sampling,
        provider_info=provider,
        recipe_name="math_v1",
        diversity_calculator=div_calculator
    )
    
    # 1. Assert verified fields
    assert len(spectrum.responses) == 2
    assert spectrum.responses[0].metadata["is_correct"] is True
    assert spectrum.responses[1].metadata["is_correct"] is False
    
    # 2. Assert consolidated metadata
    assert spectrum.metadata["recipe_name"] == "math_v1"
    assert spectrum.metadata["total_candidates"] == 2
    assert spectrum.metadata["correct_candidates"] == 1
    assert spectrum.metadata["incorrect_candidates"] == 1
    assert spectrum.metadata["correctness_ratio"] == 0.5
    
    # 3. Assert diversity statistics
    assert spectrum.diversity_statistics is not None
    assert "diversity_score" in spectrum.diversity_statistics
    assert spectrum.diversity_statistics["num_total_responses"] == 2
