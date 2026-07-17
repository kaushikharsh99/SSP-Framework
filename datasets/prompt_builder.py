"""
Prompt Builder module to format DatasetEntry objects into Prompt schema objects.
Supports flexible system prompts and user templates, with Hugging Face Chat Template integration.
"""

import logging
from typing import Any, Optional

from entry import DatasetEntry, Prompt

logger = logging.getLogger("datasets-prompt-builder")


class PromptBuilder:
    """Formats raw user prompts and queries using structured templates.
    
    Decoupled from specific datasets or tokenizer chat configurations.
    """

    def __init__(self, system_prompt: str, user_template: str = "{prompt}"):
        """Initializes PromptBuilder with system prompts and user query wrappers.
        
        Args:
            system_prompt: Guideline prompt (e.g. telling model to use specific XML tags).
            user_template: User prompt wrapper. Must contain '{prompt}' pattern.
        """
        self.system_prompt = system_prompt
        self.user_template = user_template
        if "{prompt}" not in self.user_template:
            logger.warning("User template does not contain '{prompt}' placeholder. Query might be dropped.")

    def build(self, entry: DatasetEntry, tokenizer: Optional[Any] = None) -> Prompt:
        """Transforms a DatasetEntry into a structured Prompt object.
        
        Args:
            entry: Ingested DatasetEntry.
            tokenizer: Optional wrapper or Hugging Face tokenizer supporting apply_chat_template.
        """
        user_query = self.user_template.format(prompt=entry.prompt)
        
        # Build the final prompt sequence
        # Check if tokenizer has apply_chat_template capability
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_query}
                ]
                # vLLM or Hugging Face AutoTokenizer apply_chat_template
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.debug(f"Tokenizer chat template formatting failed: {e}. Falling back to default format.")
                formatted_prompt = self._default_concat_format(user_query)
        else:
            formatted_prompt = self._default_concat_format(user_query)

        return Prompt(
            id=entry.id,
            system_prompt=self.system_prompt,
            user_query=user_query,
            formatted_prompt=formatted_prompt,
            metadata={
                **entry.metadata,
                "system_prompt_length": len(self.system_prompt),
                "user_query_length": len(user_query)
            }
        )

    def _default_concat_format(self, user_query: str) -> str:
        """Simplistic markdown fallback formatting if tokenizer is absent."""
        return (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
