"""
Tokenization wrapper layer to abstract model-specific encoding and decoding.
Prevents direct calls to Hugging Face tokenizers outside the datasets module.
"""

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("datasets-tokenizer")


class TokenizerWrapper:
    """Wrapper encapsulating Hugging Face or custom tokenization operations."""

    def __init__(
        self, 
        tokenizer_or_path: Union[str, Any], 
        max_length: int = 2048, 
        padding_side: str = "left", 
        trust_remote_code: bool = False,
        **kwargs: Any
    ):
        """Initializes the TokenizerWrapper.
        
        Args:
            tokenizer_or_path: Hugging Face model path/identifier string, 
                               or an instantiated custom tokenizer object.
            max_length: Default truncation length limit.
            padding_side: Padding alignment ('left' or 'right').
            trust_remote_code: Hugging Face specific remote code flag.
            kwargs: Extra parameters passed to HF from_pretrained.
        """
        self.max_length = max_length
        
        if isinstance(tokenizer_or_path, str):
            from transformers import AutoTokenizer
            logger.info(f"Loading AutoTokenizer from: {tokenizer_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_or_path,
                padding_side=padding_side,
                trust_remote_code=trust_remote_code,
                **kwargs
            )
        else:
            logger.info("Initializing TokenizerWrapper with custom pre-instantiated tokenizer.")
            self.tokenizer = tokenizer_or_path

        # Setup standard padding tokens if not defined (e.g., Llama-based architectures)
        if hasattr(self.tokenizer, "pad_token") and self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.debug("pad_token was unset; matched to eos_token.")

    def encode(
        self, 
        text: Union[str, List[str]], 
        max_length: Optional[int] = None, 
        truncation: bool = True, 
        padding: Union[bool, str] = True,
        return_tensors: Optional[str] = "pt",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Encodes strings or batches of strings into token sequences.
        
        Args:
            text: Single string or List of string prompts.
            max_length: Overrides default max length constraint.
            truncation: Toggles truncation of long sequences.
            padding: Configures padding type.
            return_tensors: Target return format (e.g. 'pt' for PyTorch, 'np' for NumPy).
            kwargs: Additional arguments passed to the underlying encoder call.
        """
        target_limit = max_length if max_length is not None else self.max_length
        
        # Safe execution of call
        return self.tokenizer(
            text,
            max_length=target_limit,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors,
            **kwargs
        )

    def decode(self, token_ids: Any, skip_special_tokens: bool = True, **kwargs: Any) -> str:
        """Decodes token sequences back into strings.
        
        Args:
            token_ids: Iterable of integers or PyTorch/NumPy array.
            skip_special_tokens: If True, filters out padding/EOS/BOS markers.
        """
        return self.tokenizer.decode(
            token_ids, 
            skip_special_tokens=skip_special_tokens, 
            **kwargs
        )

    def batch_decode(self, sequences: Any, skip_special_tokens: bool = True, **kwargs: Any) -> List[str]:
        """Decodes a batch of token sequences back into strings.
        
        Args:
            sequences: Multi-dimensional array of token sequences.
            skip_special_tokens: If True, filters out padding/EOS/BOS markers.
        """
        return self.tokenizer.batch_decode(
            sequences, 
            skip_special_tokens=skip_special_tokens, 
            **kwargs
        )

    def apply_chat_template(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Applies chat formatting templates if supported by the tokenizer config."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        raise AttributeError("Underlying tokenizer does not support 'apply_chat_template'.")

    @property
    def pad_token_id(self) -> Optional[int]:
        return getattr(self.tokenizer, "pad_token_id", None)

    @property
    def eos_token_id(self) -> Optional[int]:
        return getattr(self.tokenizer, "eos_token_id", None)

    @property
    def vocab_size(self) -> int:
        return getattr(self.tokenizer, "vocab_size", 0)
