"""Unit tests for ResponseParser extraction capabilities.
"""

import pytest
from generation.spectrum_engine.core.parser import ResponseParser, ParsedResponse


def test_parser_xml_thinking_and_answer():
    text = "<think>We need to sum 2 + 2.</think>\nSome intermediate text.\n<answer>4</answer>"
    res = ResponseParser.parse(text)
    assert res.thinking == "We need to sum 2 + 2."
    assert res.answer == "4"
    assert res.reasoning_length_chars == len("We need to sum 2 + 2.")
    assert res.metadata["has_xml_thinking"] is True
    assert res.metadata["has_xml_answer"] is True


def test_parser_latex_boxed():
    text = "<think>Let x be the result.</think>\nTherefore, the answer is \\boxed{42}."
    res = ResponseParser.parse(text)
    assert res.thinking == "Let x be the result."
    assert res.answer == "42"


def test_parser_nested_latex_boxed():
    text = "The value is \\boxed{\\frac{1}{2}}."
    res = ResponseParser.parse(text)
    assert res.answer == "\\frac{1}{2}"


def test_parser_post_thinking_fallback():
    text = "<think>Thinking trace here.</think>\nSo the final total is 9.5."
    res = ResponseParser.parse(text)
    assert res.thinking == "Thinking trace here."
    assert res.answer == "9.5"


def test_parser_global_fallback_number():
    text = "Calculating compound interest...\nTotal: 1000"
    res = ResponseParser.parse(text)
    assert res.thinking == ""
    assert res.answer == "1000"


def test_parser_global_fallback_text():
    text = "Calculating value...\nDone. The result is positive."
    res = ResponseParser.parse(text)
    assert res.thinking == ""
    assert res.answer == "Done. The result is positive."


def test_parser_empty_input():
    res = ResponseParser.parse("")
    assert res.thinking == ""
    assert res.answer == ""
    assert res.reasoning_length_chars == 0
