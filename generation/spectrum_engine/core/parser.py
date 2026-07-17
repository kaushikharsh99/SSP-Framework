"""Canonical response parser for extracting structured reasoning elements from LLM outputs.
Decouples raw text responses into thinking traces and final answers.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ParsedResponse:
    """Structured fields extracted from a raw model generation string."""
    thinking: str
    answer: str
    reasoning_length_chars: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResponseParser:
    """Extracts thinking traces and clean target answers using XML tags, LaTeX, and numeric fallbacks."""

    @staticmethod
    def parse(text: str) -> ParsedResponse:
        """Parses raw text into structured thinking trace and final answer fields.
        
        Looks for:
        1. XML tags check: <think>...</think> and <answer>...</answer>
        2. LaTeX check: \boxed{...}
        3. Fallbacks: Last number/lines after thinking trace.
        
        Args:
            text: Raw generated output string from a model.
            
        Returns:
            ParsedResponse object containing clean separated fields.
        """
        if not text:
            return ParsedResponse(thinking="", answer="", reasoning_length_chars=0)

        # 1. Extract thinking trace: <think>...</think>
        thinking = ""
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        if think_match:
            thinking = think_match.group(1).strip()

        # 2. Extract final answer
        answer = ""
        
        # Check XML tags <answer>...</answer>
        xml_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if xml_match:
            answer = xml_match.group(1).strip()
        else:
            # Check LaTeX \boxed{...} with support for nested braces
            boxed_idx = text.find(r"\boxed{")
            if boxed_idx != -1:
                start_idx = boxed_idx + len(r"\boxed{")
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
                    answer = text[start_idx : current_idx - 1].strip()

        # Fallback 1: Extract from text following thinking trace if answer not found yet
        if not answer and think_match:
            post_think_text = text[think_match.end():].strip()
            if post_think_text:
                # Look for numbers/fractions at the end of post-thinking block
                num_matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+/\d+|[-+]?\d+", post_think_text)
                if num_matches:
                    answer = num_matches[-1].strip()
                else:
                    # Grab the last non-empty line
                    lines = [l.strip() for l in post_think_text.split("\n") if l.strip()]
                    if lines:
                        answer = lines[-1]

        # Fallback 2: General global regex search for numbers or last lines in full text
        if not answer:
            num_matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+/\d+|[-+]?\d+", text)
            if num_matches:
                answer = num_matches[-1].strip()
            else:
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                if lines:
                    answer = lines[-1]

        return ParsedResponse(
            thinking=thinking,
            answer=answer,
            reasoning_length_chars=len(thinking),
            metadata={
                "has_xml_thinking": think_match is not None,
                "has_xml_answer": xml_match is not None
            }
        )
