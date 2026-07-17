import logging
import time
import uuid
import re
from typing import Any, Dict, List, Optional

from ..core.config import ProviderConfig
from ..core.types import (
    PromptRecord,
    ProviderInfo,
    ResponseRecord,
    SamplingConfig,
    Spectrum,
)
from .base import BaseProvider

logger = logging.getLogger("spectrum-engine.providers.vllm")


class VLLMProvider(BaseProvider):
    """Provider that generates via local vLLM engine."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._llm = None  # Will hold vllm.LLM instance

    async def initialize(self) -> None:
        """Load the model into vLLM (lazy import)."""
        logger.info(
            f"Initializing vLLM provider: {self.config.model_path} "
            f"(TP={self.config.tensor_parallel_size}, "
            f"GPU mem={self.config.gpu_memory_utilization})"
        )
        try:
            from vllm import LLM
        except ImportError as e:
            logger.error("vLLM package not installed. Cannot initialize VLLMProvider.")
            raise e

        # Initialize offline vLLM engine
        self._llm = LLM(
            model=self.config.model_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            dtype=self.config.dtype,
            trust_remote_code=self.config.trust_remote_code,
            max_model_len=self.config.max_model_len
        )

    async def generate(
        self,
        prompts: List[PromptRecord],
        sampling: SamplingConfig
    ) -> List[Spectrum]:
        """Generate spectra using vLLM's offline batch generation."""
        if not self._llm:
            await self.initialize()

        from vllm import SamplingParams

        # 1. Translate SamplingConfig -> vLLM SamplingParams
        vllm_stop = sampling.stop_sequences if sampling.stop_sequences else None
        
        sampling_params = SamplingParams(
            n=sampling.n,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            min_p=sampling.min_p,
            repetition_penalty=sampling.repetition_penalty,
            max_tokens=sampling.max_tokens,
            stop=vllm_stop,
            seed=sampling.seed
        )

        # 2. Format prompts using tokenizer chat template (if available)
        tokenizer = self._llm.get_tokenizer()
        formatted_prompts = []
        for prompt in prompts:
            messages = []
            if prompt.system_prompt:
                messages.append({"role": "system", "content": prompt.system_prompt})
            messages.append({"role": "user", "content": prompt.user_prompt})

            try:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                formatted = f"{prompt.system_prompt}\n\n{prompt.user_prompt}"
            formatted_prompts.append(formatted)

        # 3. Dispatch batch to vLLM
        start_time = time.time()
        outputs = self._llm.generate(formatted_prompts, sampling_params)
        batch_latency_ms = (time.time() - start_time) * 1000.0

        # 4. Parse output trajectories
        spectra = []
        for prompt, output in zip(prompts, outputs):
            responses = []
            
            for i, completion in enumerate(output.outputs):
                text = completion.text
                token_count = len(completion.token_ids)
                finish_reason = completion.finish_reason

                # Extract thinking trace
                thinking_trace = ""
                think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
                if think_match:
                    thinking_trace = think_match.group(1).strip()

                # Extract answer value
                extracted_answer = text.strip()
                ans_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
                if ans_match:
                    extracted_answer = ans_match.group(1).strip()
                else:
                    boxed_match = re.search(r"\\boxed\{(.*?)\}", text)
                    if boxed_match:
                        extracted_answer = boxed_match.group(1).strip()

                responses.append(ResponseRecord(
                    id=f"resp-{uuid.uuid4()}",
                    prompt_id=prompt.id,
                    text=text,
                    thinking_trace=thinking_trace,
                    extracted_answer=extracted_answer,
                    token_count=token_count,
                    finish_reason=finish_reason,
                    latency_ms=batch_latency_ms / len(output.outputs),
                    metadata={"sample_index": i}
                ))

            spectra.append(Spectrum(
                prompt=prompt,
                responses=responses,
                sampling_config=sampling,
                provider_info=self.info(),
                created_at=time.time()
            ))

        return spectra

    async def shutdown(self) -> None:
        """Unload the model and free memory."""
        if self._llm:
            logger.info("Shutting down vLLM provider.")
            # Explicitly delete the LLM engine to free CUDA memory
            del self._llm
            self._llm = None

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="vllm-local",
            backend="vllm",
            model=self.config.model_path or self.config.model,
            metadata={
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "dtype": self.config.dtype,
            }
        )
