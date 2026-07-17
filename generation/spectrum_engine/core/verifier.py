"""Verifier integration wrapper for the Spectrum Generation Engine.
Binds canonical parser extraction outputs to task-specific verifiers.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List

from .parser import ResponseParser
from .types import ResponseRecord

logger = logging.getLogger("spectrum-engine.verifier")

# Dynamically resolve datasets folder to import local verification modules
_current_dir = os.path.dirname(os.path.abspath(__file__))
_datasets_dir = os.path.abspath(os.path.join(_current_dir, "../../../datasets"))
if _datasets_dir not in sys.path:
    sys.path.insert(0, _datasets_dir)

try:
    from verifiers import VerifierRegistry
except ImportError:
    VerifierRegistry = None

try:
    from entry import DatasetEntry
except ImportError:
    DatasetEntry = None


class ResponseVerifier:
    """Orchestrates parsing raw responses, running evaluations, and compiling metrics."""

    @staticmethod
    def verify(
        response: ResponseRecord,
        verifier_type: str,
        ground_truth: str,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Parses and verifies a response, mutating ResponseRecord fields in-place.
        
        Args:
            response: ResponseRecord containing raw generated text.
            verifier_type: The type of verification ('math' or 'code').
            ground_truth: Correct solution target string.
            test_cases: Optional test assertions list for code runs.
            
        Returns:
            Dictionary containing verification status and parsing metrics.
        """
        # 1. Parse raw text into structured parts
        parsed = ResponseParser.parse(response.text)
        
        # Populate ResponseRecord fields
        response.thinking_trace = parsed.thinking
        response.extracted_answer = parsed.answer

        is_correct = False
        error_message = None

        # 2. Evaluate answer correctness using VerifierRegistry
        if VerifierRegistry:
            try:
                verifier = VerifierRegistry.get(verifier_type)
                
                if DatasetEntry:
                    entry = DatasetEntry(
                        id=response.prompt_id,
                        prompt="",
                        ground_truth_answer=ground_truth,
                        test_cases=test_cases or []
                    )
                else:
                    class DummyEntry:
                        def __init__(self, ans, tests):
                            self.ground_truth_answer = ans
                            self.test_cases = tests
                    entry = DummyEntry(ground_truth, test_cases or [])
                    
                verify_res = verifier.verify(response, entry)
                is_correct = verify_res.is_correct
                error_message = verify_res.error_message
            except Exception as e:
                logger.error(f"Verification failed with exception: {e}", exc_info=True)
                error_message = f"Verifier exception: {str(e)}"
        else:
            # Fallback direct string evaluation
            is_correct = parsed.answer.strip().lower() == ground_truth.strip().lower()
            if not is_correct:
                error_message = f"Fallback exact string mismatch. Candidate: '{parsed.answer}', Target: '{ground_truth}'"

        # Update metadata records
        response.metadata["is_correct"] = is_correct
        response.metadata["verify_error"] = error_message
        response.metadata["extracted_answer"] = parsed.answer
        response.metadata["reasoning_length_chars"] = parsed.reasoning_length_chars

        return {
            "is_correct": is_correct,
            "error_message": error_message,
            "extracted_answer": parsed.answer,
            "reasoning_length_chars": parsed.reasoning_length_chars
        }
