"""
Diversity metrics calculator for analyzing reasoning trajectories in a Spectrum.
Decouples diversity measurements from model generation logic.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set

import sys
import os
# Add root path to PYTHONPATH to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets")))
from entry import Response

logger = logging.getLogger("models-diversity")


class BaseDiversityCalculator(ABC):
    """Abstract interface defining the contract for calculating Spectrum diversity."""

    @abstractmethod
    def calculate(self, responses: List[Response]) -> Dict[str, Any]:
        """Calculates diversity metrics for a set of candidate responses.
        
        Args:
            responses: List of Response objects generated for a single prompt.
            
        Returns:
            Dictionary containing metrics (e.g. unique response ratio, Jaccard distance).
        """
        pass


class LexicalDiversityCalculator(BaseDiversityCalculator):
    """Computes string-based lexical diversity and overlap statistics."""

    def calculate(self, responses: List[Response]) -> Dict[str, Any]:
        if not responses:
            return {}

        total_count = len(responses)
        raw_texts = [r.text for r in responses]
        extracted_answers = [r.extracted_answer for r in responses]

        # 1. Uniqueness Counts
        unique_responses: Set[str] = set(raw_texts)
        unique_answers: Set[str] = set(extracted_answers)

        num_unique_responses = len(unique_responses)
        num_unique_answers = len(unique_answers)
        duplicate_count = total_count - num_unique_responses

        # 2. Length Statistics (Character lengths)
        lengths = [len(t) for t in raw_texts]
        avg_char_length = sum(lengths) / total_count
        min_char_length = min(lengths)
        max_char_length = max(lengths)

        # 3. Token-Level Length Statistics (if token_ids are populated)
        token_lengths = [len(r.token_ids) for r in responses if r.token_ids]
        if token_lengths:
            avg_token_count = sum(token_lengths) / len(token_lengths)
            min_token_count = min(token_lengths)
            max_token_count = max(token_lengths)
        else:
            avg_token_count = 0.0
            min_token_count = 0
            max_token_count = 0

        # 4. Pairwise Lexical Jaccard Similarity
        avg_pairwise_jaccard = self._compute_avg_pairwise_jaccard(raw_texts)
        
        # 5. Semantic Similarity (Placeholder, currently delegates to lexical Jaccard)
        avg_pairwise_semantic = self._compute_avg_pairwise_semantic(raw_texts, avg_pairwise_jaccard)

        # 6. Combined Diversity Score
        # Diversity score increases as Jaccard similarity decreases, and scales with the ratio of unique answers
        unique_answer_ratio = num_unique_answers / total_count
        diversity_score = unique_answer_ratio * (1.0 - avg_pairwise_jaccard)

        return {
            "num_total_responses": total_count,
            "num_unique_responses": num_unique_responses,
            "num_unique_answers": num_unique_answers,
            "duplicate_response_count": duplicate_count,
            "char_length_avg": avg_char_length,
            "char_length_min": min_char_length,
            "char_length_max": max_char_length,
            "token_count_avg": avg_token_count,
            "token_count_min": min_token_count,
            "token_count_max": max_token_count,
            "avg_pairwise_lexical_similarity": avg_pairwise_jaccard,
            "avg_pairwise_semantic_similarity": avg_pairwise_semantic,
            "diversity_score": diversity_score
        }

    def _compute_avg_pairwise_jaccard(self, texts: List[str]) -> float:
        """Computes average word-level Jaccard similarity across all unique pairs."""
        if len(texts) < 2:
            return 0.0

        # Convert texts to word token sets
        word_sets = [set(t.split()) for t in texts]
        
        total_similarity = 0.0
        comparisons = 0

        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                set1 = word_sets[i]
                set2 = word_sets[j]
                
                if not set1 or not set2:
                    similarity = 0.0
                else:
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    similarity = intersection / union if union > 0 else 0.0
                
                total_similarity += similarity
                comparisons += 1

        return total_similarity / comparisons if comparisons > 0 else 0.0

    def _compute_avg_pairwise_semantic(self, texts: List[str], lexical_score: float) -> float:
        """Interface placeholder for embedding-based semantic similarity.
        
        Subclasses can override this to run Sentence-Transformers or API embeddings.
        Currently defaults to a mock computation proportional to the lexical Jaccard score.
        """
        # Placeholder calculation
        return lexical_score
