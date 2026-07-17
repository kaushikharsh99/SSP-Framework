"""
Unit tests for the verifiers module: MathVerifier and CodeVerifier.
"""

import os
import sys
import pytest

# Add the datasets directory to the Python path to avoid name collisions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets")))

from entry import DatasetEntry, Response
from verifiers import MathVerifier, CodeVerifier, VerifierRegistry


def test_math_verifier_extraction():
    # Test XML extraction
    assert MathVerifier.extract_answer("The result is <answer>100</answer> yards.") == "100"
    
    # Test boxed LaTeX extraction
    assert MathVerifier.extract_answer("Use \\boxed{\\frac{1}{2}} to solve.") == "\\frac{1}{2}"
    
    # Test fallback extraction (last float/int)
    assert MathVerifier.extract_answer("Result: -3.5") == "-3.5"
    assert MathVerifier.extract_answer("Equations: 2x = 4 / 2") == "2"


def test_math_verifier_correctness():
    verifier = MathVerifier()

    # 1. Symbolic expression equivalence (requires SymPy)
    entry_sym = DatasetEntry(id="m1", prompt="Solve", ground_truth_answer="x + y")
    
    res_sym_correct = Response(id="r1", prompt_id="m1", text="<answer>y + x</answer>")
    result1 = verifier.verify(res_sym_correct, entry_sym)
    assert result1.is_correct is True

    res_sym_incorrect = Response(id="r2", prompt_id="m1", text="<answer>y - x</answer>")
    result2 = verifier.verify(res_sym_incorrect, entry_sym)
    assert result2.is_correct is False

    # 2. Fractions and arithmetic evaluation
    entry_arith = DatasetEntry(id="m2", prompt="Solve", ground_truth_answer="1/2 * x")
    res_arith_correct = Response(id="r3", prompt_id="m2", text="<answer>0.5 * x</answer>")
    assert verifier.verify(res_arith_correct, entry_arith).is_correct is True

    # 3. Numeric float evaluation
    entry_num = DatasetEntry(id="m3", prompt="Solve", ground_truth_answer="3.14159")
    res_num_correct = Response(id="r4", prompt_id="m3", text="<answer>3.141590</answer>")
    assert verifier.verify(res_num_correct, entry_num).is_correct is True

    # 4. String fallback
    entry_str = DatasetEntry(id="m4", prompt="Solve", ground_truth_answer="yes")
    res_str_correct = Response(id="r5", prompt_id="m4", text="Yes")
    assert verifier.verify(res_str_correct, entry_str).is_correct is True


def test_code_verifier_extraction():
    code_text = """
Some thinking steps here.
```python
def add(a, b):
    return a + b
```
Some other text.
"""
    assert CodeVerifier.extract_code(code_text) == "def add(a, b):\n    return a + b"


def test_code_verifier_execution():
    # Setup verifier with short timeout for test efficiency
    verifier = CodeVerifier(timeout_seconds=0.5)

    # Base correct submission
    entry = DatasetEntry(
        id="c1", 
        prompt="Write function to add two numbers.", 
        ground_truth_answer="def add(a, b): return a + b",
        test_cases=[{"assertion": "assert add(2, 3) == 5"}, {"assertion": "assert add(-1, 1) == 0"}]
    )

    # 1. Correct code execution
    res_correct = Response(
        id="rc1", 
        prompt_id="c1", 
        text="```python\ndef add(a, b):\n    return a + b\n```"
    )
    result_correct = verifier.verify(res_correct, entry)
    assert result_correct.is_correct is True
    assert result_correct.error_message is None

    # 2. Code execution with failing assertion
    res_failing = Response(
        id="rc2", 
        prompt_id="c1", 
        text="```python\ndef add(a, b):\n    return a + b + 1  # Bug!\n```"
    )
    result_failing = verifier.verify(res_failing, entry)
    assert result_failing.is_correct is False
    assert "AssertionError" in result_failing.error_message

    # 3. Code execution with syntax error
    res_syntax = Response(
        id="rc3", 
        prompt_id="c1", 
        text="```python\ndef add(a, b):\n    return a + b # Missing indentation\n  return 0\n```"
    )
    result_syntax = verifier.verify(res_syntax, entry)
    assert result_syntax.is_correct is False
    assert "IndentationError" in result_syntax.error_message or "TabError" in result_syntax.error_message


def test_code_verifier_timeout():
    verifier = CodeVerifier(timeout_seconds=0.5)
    entry = DatasetEntry(
        id="c2",
        prompt="Write a loop.",
        ground_truth_answer="pass"
    )
    # Loop that never ends
    res_loop = Response(
        id="rc_loop",
        prompt_id="c2",
        text="```python\nimport time\nwhile True:\n    pass\n```"
    )
    result = verifier.verify(res_loop, entry)
    assert result.is_correct is False
    assert "Timeout expired" in result.error_message


def test_code_verifier_security():
    verifier = CodeVerifier()
    entry = DatasetEntry(id="c3", prompt="Write file.", ground_truth_answer="pass")

    res_unsafe = Response(
        id="rc_unsafe",
        prompt_id="c3",
        text="```python\nimport os\nos.system('rm -rf /')\n```"
    )
    result = verifier.verify(res_unsafe, entry)
    assert result.is_correct is False
    assert "Security rejection" in result.error_message


def test_verifier_registry():
    math_verifier = VerifierRegistry.get("math")
    code_verifier = VerifierRegistry.get("code")
    
    assert isinstance(math_verifier, MathVerifier)
    assert isinstance(code_verifier, CodeVerifier)
    
    with pytest.raises(KeyError):
        VerifierRegistry.get("invalid_type")
