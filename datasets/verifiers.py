"""
Verifier implementations for validating model responses.
Provides BaseVerifier, MathVerifier (symbolic matching), and CodeVerifier (subprocess sandbox testing).
"""

import logging
import re
import os
import sys
import tempfile
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

# Try importing SymPy for symbolic math verification
try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

from entry import DatasetEntry, Response, VerificationResult

logger = logging.getLogger("datasets-verifiers")


class BaseVerifier(ABC):
    """Abstract base class defining the verification contract."""

    @abstractmethod
    def verify(
        self, 
        response: Response, 
        ground_truth: DatasetEntry
    ) -> VerificationResult:
        """Evaluates whether the candidate response matches the target ground truth.
        
        Args:
            response: Model-generated response trajectory containing text and extracted answer.
            ground_truth: Expected source DatasetEntry record.
        """
        pass


class MathVerifier(BaseVerifier):
    """Verifies mathematical answers using SymPy symbolic evaluation and numerical fallbacks."""

    def __init__(self, fallback_exact: bool = True):
        self.fallback_exact = fallback_exact
        if not HAS_SYMPY:
            logger.warning("SymPy is not installed. MathVerifier will fall back to string/numeric exact checks.")

    def verify(
        self, 
        response: Response, 
        ground_truth: DatasetEntry
    ) -> VerificationResult:
        candidate_str = self.extract_answer(response.text)
        target_str = ground_truth.ground_truth_answer.strip()
        
        # Log extraction details
        logger.debug(f"Math verification - Extracted: '{candidate_str}' | Target: '{target_str}'")

        # 1. Edge Case: Empty Candidate
        if not candidate_str:
            return VerificationResult(
                trajectory_id=response.id,
                is_correct=False,
                error_message="Could not extract mathematical answer from text."
            )

        # 2. Symbolic Match Check (if SymPy is available)
        if HAS_SYMPY:
            try:
                # Parse expressions
                expr_cand = parse_expr(candidate_str)
                expr_targ = parse_expr(target_str)
                
                # Check equivalence by simplifying difference
                diff = sympy.simplify(expr_cand - expr_targ)
                if diff == 0:
                    return VerificationResult(trajectory_id=response.id, is_correct=True)
            except Exception as e:
                logger.debug(f"SymPy symbolic parsing failed for candidate '{candidate_str}' vs target '{target_str}': {e}")

        # 3. Numeric Float Equivalence Check
        try:
            val_cand = float(candidate_str)
            val_targ = float(target_str)
            if abs(val_cand - val_targ) < 1e-5:
                return VerificationResult(trajectory_id=response.id, is_correct=True)
        except ValueError:
            pass

        # 4. Fallback: Direct String Equivalence Check
        is_match = candidate_str.lower() == target_str.lower()
        
        return VerificationResult(
            trajectory_id=response.id,
            is_correct=is_match,
            error_message=None if is_match else f"Answer mismatch. Candidate: {candidate_str}, Target: {target_str}"
        )

    @staticmethod
    def extract_answer(text: str) -> str:
        """Extracts the math answer substring from output text.
        
        Looks for:
        - XML tags: <answer>...</answer>
        - LaTeX box: \\boxed{...}
        - Fallback: last numeric string or equation sequence.
        """
        if not text:
            return ""
            
        # 1. XML tags check
        xml_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if xml_match:
            return xml_match.group(1).strip()

        # 2. LaTeX boxed block check (with matching nested braces logic)
        idx = text.find(r"\boxed{")
        if idx != -1:
            start_idx = idx + len(r"\boxed{")
            depth = 1
            current_idx = start_idx
            while current_idx < len(text) and depth > 0:
                char = text[current_idx]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                current_idx += 1
            if depth == 0:
                return text[start_idx : current_idx - 1].strip()

        # 3. Fallback: Search backward for numbers or equations at the end of the text
        # Match floating point numbers, fractions, or negative numbers
        matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+/\d+|[-+]?\d+", text)
        if matches:
            return matches[-1].strip()

        return text.strip()


class CodeVerifier(BaseVerifier):
    """Executes generated Python functions in a subprocess sandbox against test assertions."""

    def __init__(self, timeout_seconds: float = 2.0, reject_unsafe_keywords: bool = True):
        self.timeout_seconds = timeout_seconds
        self.reject_unsafe_keywords = reject_unsafe_keywords
        # List of critical keywords that warrant direct rejection to safeguard personal workspace
        self.unsafe_patterns = [
            r"import\s+os", r"import\s+sys", r"import\s+shutil", r"import\s+subprocess",
            r"from\s+os", r"from\s+sys", r"from\s+subprocess",
            r"__import__", r"eval\(", r"exec\(", r"open\("
        ]

    def verify(
        self, 
        response: Response, 
        ground_truth: DatasetEntry
    ) -> VerificationResult:
        code_block = self.extract_code(response.text)
        
        if not code_block:
            return VerificationResult(
                trajectory_id=response.id,
                is_correct=False,
                error_message="Could not extract Python code block from response."
            )

        # 1. Security Check
        if self.reject_unsafe_keywords:
            for pattern in self.unsafe_patterns:
                if re.search(pattern, code_block):
                    return VerificationResult(
                        trajectory_id=response.id,
                        is_correct=False,
                        error_message=f"Security rejection: unsafe pattern '{pattern}' detected in generated code."
                    )

        # 2. Extract unit tests from ground_truth
        test_cases = ground_truth.test_cases or []
        assertions = []
        for tc in test_cases:
            assertion = tc.get("assertion", "")
            if assertion:
                assertions.append(assertion)

        # Assemble full execution script
        exec_script = code_block + "\n\n"
        exec_script += "# --- Automated Test Assertions ---\n"
        if not assertions:
            # If no assertions are explicitly defined, we check if the code runs without syntax/execution errors
            exec_script += "pass\n"
        else:
            for idx, assertion in enumerate(assertions):
                exec_script += f"try:\n    {assertion}\nexcept AssertionError as ae:\n    raise AssertionError(f'Test case {idx} failed: {assertion}') from ae\n"

        # 3. Subprocess Execution Sandbox
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(exec_script)
            temp_script_path = f.name

        try:
            # Execute python script in an isolated subprocess
            res = subprocess.run(
                [sys.executable, temp_script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            
            if res.returncode == 0:
                return VerificationResult(trajectory_id=response.id, is_correct=True)
            else:
                error_log = res.stderr.strip() or res.stdout.strip()
                # Clean temp path from logs to keep trace clean
                error_log = error_log.replace(temp_script_path, "sandbox.py")
                return VerificationResult(
                    trajectory_id=response.id,
                    is_correct=False,
                    error_message=f"Execution error:\n{error_log}"
                )
                
        except subprocess.TimeoutExpired:
            return VerificationResult(
                trajectory_id=response.id,
                is_correct=False,
                error_message=f"Timeout expired: execution exceeded limit of {self.timeout_seconds}s."
            )
        except Exception as e:
            return VerificationResult(
                trajectory_id=response.id,
                is_correct=False,
                error_message=f"System error executing sandbox: {e}"
            )
        finally:
            if os.path.exists(temp_script_path):
                os.remove(temp_script_path)

    @staticmethod
    def extract_code(text: str) -> str:
        """Extracts python code blocks from markdown tags or XML answers."""
        if not text:
            return ""

        # 1. Look for markdown python block
        md_match = re.search(r"```python(.*?)```", text, re.DOTALL)
        if md_match:
            return md_match.group(1).strip()

        # 2. Look for general markdown code block
        general_match = re.search(r"```(.*?)```", text, re.DOTALL)
        if general_match:
            return general_match.group(1).strip()
            
        # 3. Look for XML answer tags
        xml_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if xml_match:
            return xml_match.group(1).strip()

        return text.strip()


class VerifierRegistry:
    """Registry to map task types to specialized Verifier instances."""
    
    _verifiers: Dict[str, BaseVerifier] = {
        "math": MathVerifier(),
        "code": CodeVerifier()
    }

    @classmethod
    def register(cls, task_type: str, verifier: BaseVerifier) -> None:
        """Registers a custom verifier mapping for a task type."""
        cls._verifiers[task_type] = verifier

    @classmethod
    def get(cls, task_type: str) -> BaseVerifier:
        """Retrieves the registered verifier for the task type."""
        if task_type not in cls._verifiers:
            raise KeyError(
                f"No verifier registered for task type '{task_type}'. Registered: {list(cls._verifiers.keys())}"
            )
        return cls._verifiers[task_type]
