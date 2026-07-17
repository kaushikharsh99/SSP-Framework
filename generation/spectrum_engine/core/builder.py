"""Spectrum builder for consolidating generated reasoning trajectories.
Converts collections of responses into unified verified Spectrums.
"""

import time
import logging
from typing import List, Dict, Any, Optional

from .types import PromptRecord, ResponseRecord, SamplingConfig, Spectrum, ProviderInfo
from .verifier import ResponseVerifier
from .diversity import LexicalDiversityCalculator

logger = logging.getLogger("spectrum-engine.builder")


class SpectrumBuilder:
    """Consolidates discrete trajectory generations into unified, verified Spectrum records."""

    @staticmethod
    def build(
        prompt: PromptRecord,
        responses: List[ResponseRecord],
        verifier_type: str,
        sampling_config: SamplingConfig,
        provider_info: ProviderInfo,
        recipe_name: str,
        diversity_calculator: Optional[LexicalDiversityCalculator] = None
    ) -> Spectrum:
        """Assembles a verified and analyzed Spectrum from candidate responses.
        
        Args:
            prompt: Base PromptRecord question.
            responses: Candidate model responses generated for this task.
            verifier_type: Problem domain verifier type ('math' or 'code').
            sampling_config: Config options utilized during sampling.
            provider_info: Provider/model origin metadata.
            recipe_name: Reference tag for the active generation recipe.
            diversity_calculator: Optional LexicalDiversityCalculator instance.
            
        Returns:
            Spectrum artifact containing evaluated candidates and metrics.
        """
        ground_truth = prompt.metadata.get("answer") or prompt.metadata.get("ground_truth") or ""
        test_cases = prompt.metadata.get("test_cases") or []

        # 1. Parse and Verify each response trajectory
        for resp in responses:
            ResponseVerifier.verify(
                response=resp,
                verifier_type=verifier_type,
                ground_truth=ground_truth,
                test_cases=test_cases
            )

        # 2. Compile candidate statistics
        num_correct = sum(1 for r in responses if r.metadata.get("is_correct") is True)
        
        metadata = {
            "recipe_name": recipe_name,
            "total_candidates": len(responses),
            "correct_candidates": num_correct,
            "incorrect_candidates": len(responses) - num_correct,
            "correctness_ratio": num_correct / len(responses) if responses else 0.0
        }

        # 3. Calculate diversity metrics if calculator is available
        diversity_statistics = {}
        if diversity_calculator and responses:
            diversity_statistics = diversity_calculator.calculate(responses)

        return Spectrum(
            prompt=prompt,
            responses=responses,
            sampling_config=sampling_config,
            provider_info=provider_info,
            created_at=time.time(),
            diversity_statistics=diversity_statistics,
            metadata=metadata
        )
