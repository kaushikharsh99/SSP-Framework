"""
Spectrum Generation Engine implementation.
Defines BaseSpectrumGenerator and HFSpectrumGenerator for generating reasoning trajectories.
Supports saving spectra to JSONL and fallback CPU modes.
"""

import time
import os
import sys
import json
import logging
import re
import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch

# Add paths to PYTHONPATH to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from entry import DatasetEntry, Prompt, Response, Spectrum
from tokenizer import TokenizerWrapper
from diversity import LexicalDiversityCalculator

logger = logging.getLogger("models-generator")


class BaseSpectrumGenerator(ABC):
    """Abstract interface defining the contract for generating reasoning Spectra."""

    @abstractmethod
    def generate_spectrum(
        self, 
        prompt: Prompt, 
        num_trajectories: int = 4, 
        save_path: Optional[str] = None,
        **kwargs: Any
    ) -> Spectrum:
        """Generates a diversity Spectrum for a single prompt.
        
        Args:
            prompt: Formatted Prompt object.
            num_trajectories: Number of independent trajectories to generate.
            save_path: Optional path to append the generated Spectrum as JSONL.
        """
        pass

    @abstractmethod
    def batch_generate_spectra(
        self, 
        prompts: List[Prompt], 
        num_trajectories: int = 4, 
        save_path: Optional[str] = None,
        **kwargs: Any
    ) -> List[Spectrum]:
        """Generates diversity Spectra for a batch of prompts.
        
        Args:
            prompts: List of formatted Prompt objects.
            num_trajectories: Number of trajectories per prompt.
            save_path: Optional path to append the generated Spectra as JSONL.
        """
        pass


class HFSpectrumGenerator(BaseSpectrumGenerator):
    """Hugging Face Causal Language Model backend for generating solution spectra."""

    def __init__(
        self,
        model_name_or_path: Union[str, Any],
        tokenizer_or_path: Optional[Union[str, Any]] = None,
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        max_length: int = 2048,
        **model_kwargs: Any
    ):
        """Initializes model, tokenizer wrapper, and diversity metrics calculator."""
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.diversity_calculator = LexicalDiversityCalculator()

        # Resolve dtype
        if torch_dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif torch_dtype == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        # 1. Load Tokenizer
        tokenizer_target = tokenizer_or_path if tokenizer_or_path is not None else model_name_or_path
        self.tokenizer = TokenizerWrapper(
            tokenizer_or_path=tokenizer_target,
            max_length=max_length,
            trust_remote_code=trust_remote_code
        )

        # 2. Resolve Device & Load Model
        self.device = device
        if isinstance(model_name_or_path, str):
            from transformers import AutoModelForCausalLM
            
            # Autodetect CUDA
            if device == "auto":
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                resolved_device = device
                
            logger.info(f"Loading HF model '{model_name_or_path}' on device: {resolved_device} ({self.dtype})")
            
            # Safe load with fallback to CPU if OOM occurs or CUDA is missing
            try:
                if resolved_device == "cuda":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,
                        torch_dtype=self.dtype,
                        device_map="auto",
                        trust_remote_code=trust_remote_code,
                        **model_kwargs
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,
                        torch_dtype=torch.float32,  # Fallback to fp32 on CPU
                        device_map=None,
                        trust_remote_code=trust_remote_code,
                        **model_kwargs
                    ).to("cpu")
            except Exception as e:
                logger.error(f"Failed to load model {model_name_or_path}: {e}")
                raise e
        else:
            # Mock or pre-loaded model passed directly for unit tests
            self.model = model_name_or_path
            logger.info("Generator initialized with pre-loaded or mock model wrapper.")

    def generate_spectrum(
        self, 
        prompt: Prompt, 
        num_trajectories: int = 4, 
        save_path: Optional[str] = None,
        **kwargs: Any
    ) -> Spectrum:
        """Generates a diversity Spectrum for a single prompt."""
        spectra = self.batch_generate_spectra(
            prompts=[prompt],
            num_trajectories=num_trajectories,
            save_path=save_path,
            **kwargs
        )
        return spectra[0]

    def batch_generate_spectra(
        self, 
        prompts: List[Prompt], 
        num_trajectories: int = 4, 
        save_path: Optional[str] = None,
        **kwargs: Any
    ) -> List[Spectrum]:
        """Generates diversity Spectra for a batch of prompts."""
        spectra: List[Spectrum] = []
        
        # In case we pass an empty list
        if not prompts:
            return []

        # Merge defaults with custom generation kwargs
        gen_config = self._build_generation_config(kwargs)

        for prompt_idx, prompt in enumerate(prompts):
            start_time = time.time()
            logger.info(f"Generating spectrum size N={num_trajectories} for prompt {prompt_idx + 1}/{len(prompts)}")

            try:
                responses = self._generate_trajectories(prompt, num_trajectories, gen_config)
            except Exception as e:
                logger.error(f"Error during trajectory generation for prompt '{prompt.id}': {e}. Recovering with empty responses.")
                responses = []

            # Compute statistics and timing metrics
            latency = time.time() - start_time
            total_tokens = sum(len(r.token_ids) for r in responses)
            throughput = total_tokens / latency if latency > 0 else 0.0

            timing = {
                "latency_seconds": latency,
                "total_tokens_generated": total_tokens,
                "tokens_per_second": throughput
            }

            # Compute diversity metrics
            diversity_stats = self.diversity_calculator.calculate(responses)

            # Create Spectrum object
            spectrum = Spectrum(
                prompt=prompt,
                responses=responses,
                generation_config=gen_config,
                metadata={
                    "model": str(self.model_name_or_path),
                    "timestamp": time.time()
                },
                diversity_statistics=diversity_stats,
                generation_timing=timing
            )

            # Log spectrum details
            score = diversity_stats.get("diversity_score")
            score_str = f"{score:.3f}" if score is not None else "N/A"
            logger.info(
                f"Generated Spectrum. Size: {len(responses)}, "
                f"Unique Answers: {diversity_stats.get('num_unique_answers')}, "
                f"Diversity Score: {score_str}"
            )

            # Save to disk as JSONL if requested
            if save_path:
                self._save_spectrum_to_disk(spectrum, save_path)

            spectra.append(spectrum)

        return spectra

    def _generate_trajectories(
        self, 
        prompt: Prompt, 
        num_trajectories: int, 
        gen_config: Dict[str, Any]
    ) -> List[Response]:
        """Generates trajectories by replicating the input prompt N times."""
        # 1. Encode prompt
        encoded = self.tokenizer.encode(
            prompt.formatted_prompt, 
            padding=True, 
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Handle device placement (mock models might not have .device attribute)
        if hasattr(self.model, "device"):
            input_ids = input_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)

        # Replicate batch dimension N times
        replicated_input_ids = input_ids.repeat(num_trajectories, 1)
        replicated_attention_mask = attention_mask.repeat(num_trajectories, 1)

        prompt_len = input_ids.shape[1]

        # Extract stop tokens/sequences (use .get to preserve across batch iterations)
        stop_sequences = gen_config.get("stop_sequences", [])

        # Filter out custom keys that HF model.generate() doesn't understand
        hf_gen_config = {k: v for k, v in gen_config.items() if k != "stop_sequences"}

        # 2. Run model generation
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=replicated_input_ids,
                attention_mask=replicated_attention_mask,
                **hf_gen_config
            )

        responses: List[Response] = []
        
        # Decode and parse responses
        for idx in range(num_trajectories):
            output_seq = outputs[idx]
            
            # Separate prompt tokens from generated tokens
            gen_tokens = output_seq[prompt_len:]
            gen_token_ids = gen_tokens.tolist()

            # Decode text
            raw_text = self.tokenizer.decode(gen_token_ids, skip_special_tokens=True)

            # Extract thinking trace and final answer from raw text (before truncation)
            thinking_trace = self._extract_thinking_trace(raw_text)
            extracted_answer = self._extract_answer_value(raw_text)

            # Apply stop sequence truncation for the stored text field
            clean_text = self._truncate_at_stop_sequences(raw_text, stop_sequences)

            response_obj = Response(
                id=f"{prompt.id}-traj-{idx}-{hash(clean_text)}",
                prompt_id=prompt.id,
                text=clean_text,
                thinking_trace=thinking_trace,
                extracted_answer=extracted_answer,
                token_ids=gen_token_ids,
                metadata={
                    "replicated_index": idx,
                    "raw_text_length": len(raw_text),
                    "is_truncated_by_stop": len(clean_text) < len(raw_text)
                }
            )
            responses.append(response_obj)

        return responses

    def _build_generation_config(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merges default configurations with generation overrides."""
        defaults = {
            "max_new_tokens": 512,
            "min_new_tokens": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "stop_sequences": ["<|im_end|>", "</answer>"]
        }

        # If temperature is 0, disable sampling to prevent runtime exceptions in transformers
        if overrides.get("temperature") == 0.0 or overrides.get("do_sample") is False:
            overrides["do_sample"] = False
            overrides["temperature"] = None
            overrides["top_p"] = None
            overrides["top_k"] = None

        config = {**defaults, **overrides}
        
        # Clean config of None values
        return {k: v for k, v in config.items() if v is not None}

    def _truncate_at_stop_sequences(self, text: str, stop_sequences: List[str]) -> str:
        """Truncates the generated output at the first occurrence of any stop sequence."""
        if not text or not stop_sequences:
            return text
            
        earliest_idx = len(text)
        found = False

        for seq in stop_sequences:
            idx = text.find(seq)
            if idx != -1 and idx < earliest_idx:
                earliest_idx = idx
                found = True

        return text[:earliest_idx].strip() if found else text

    def _extract_thinking_trace(self, text: str) -> str:
        """Extracts content inside <think>...</think> tags if present."""
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_answer_value(self, text: str) -> str:
        """Extracts final output answer (looks for <answer> or fallback boxed)."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        # Fallback boxed
        boxed_match = re.search(r"\\boxed\{(.*?)\}", text)
        if boxed_match:
            return boxed_match.group(1).strip()
            
        return text.strip()

    def _save_spectrum_to_disk(self, spectrum: Spectrum, save_path: str) -> None:
        """Saves a single Spectrum instance to a JSONL file."""
        try:
            # Ensure output directory exists
            out_dir = os.path.dirname(save_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

            serialized = dataclasses.asdict(spectrum)
            
            with open(save_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(serialized) + "\n")
            logger.debug(f"Saved Spectrum for prompt '{spectrum.prompt.id}' to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save spectrum to disk: {e}")
