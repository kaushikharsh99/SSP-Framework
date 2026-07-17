"""Unit tests for the ResponseVerifier integration.
"""

import pytest
from generation.spectrum_engine.core.verifier import ResponseVerifier
from generation.spectrum_engine.core.types import ResponseRecord


def test_response_verifier_correct_math():
    resp = ResponseRecord(
        id="r-1",
        prompt_id="q-1",
        text="<think>Adding 10 + 10.</think>\nTherefore, <answer>20</answer>.",
        token_count=10,
        finish_reason="stop"
    )
    
    metrics = ResponseVerifier.verify(
        response=resp,
        verifier_type="math",
        ground_truth="20"
    )
    
    assert metrics["is_correct"] is True
    assert metrics["extracted_answer"] == "20"
    assert metrics["reasoning_length_chars"] == len("Adding 10 + 10.")
    assert metrics["error_message"] is None
    
    # Check mutations in response
    assert resp.thinking_trace == "Adding 10 + 10."
    assert resp.extracted_answer == "20"
    assert resp.metadata["is_correct"] is True
    assert resp.metadata["extracted_answer"] == "20"


def test_response_verifier_incorrect_math():
    resp = ResponseRecord(
        id="r-2",
        prompt_id="q-1",
        text="<think>Adding 10 + 10.</think>\nTherefore, <answer>30</answer>.",
        token_count=10,
        finish_reason="stop"
    )
    
    metrics = ResponseVerifier.verify(
        response=resp,
        verifier_type="math",
        ground_truth="20"
    )
    
    assert metrics["is_correct"] is False
    assert metrics["extracted_answer"] == "30"
    assert metrics["error_message"] is not None
    
    # Check mutations in response
    assert resp.metadata["is_correct"] is False
    assert "Answer mismatch" in resp.metadata["verify_error"] or "Fallback" in resp.metadata["verify_error"]
