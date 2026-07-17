"""OpenAI-compatible API provider for the Spectrum Generation Engine.

Works with any OpenAI-compatible endpoint:
  - OpenRouter
  - Together AI
  - Fireworks AI
  - Groq
  - Local vLLM/TGI servers
  - Any future compatible provider

Nothing is hardcoded to any specific provider.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from ..core.config import ProviderConfig
from ..core.exceptions import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from ..core.types import (
    PromptRecord,
    ProviderInfo,
    ResponseRecord,
    SamplingConfig,
    Spectrum,
)
from .base import BaseProvider

logger = logging.getLogger("spectrum-engine.providers.api")


import httpx
import re

from ..core.config import ProviderConfig
from ..core.exceptions import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from ..core.types import (
    PromptRecord,
    ProviderInfo,
    ResponseRecord,
    SamplingConfig,
    Spectrum,
)
from .base import BaseProvider
from ..utils.retry import retry_async

logger = logging.getLogger("spectrum-engine.providers.api")


class APIProvider(BaseProvider):
    """Provider that generates via OpenAI-compatible HTTP APIs."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._client = None  # Will hold the async HTTP client

    async def initialize(self) -> None:
        """Create the async HTTP client."""
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        # Add headers for common aggregators (like OpenRouter)
        if "openrouter.ai" in self.config.base_url.lower():
            headers["HTTP-Referer"] = "https://github.com/kaushikharsh99/SSP-Framework"
            headers["X-Title"] = "Spectrum Generation Engine"
            
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(self.config.timeout),
            follow_redirects=True
        )
        logger.info(f"Initialized API provider: {self.config.base_url} / {self.config.model}")

    async def generate(
        self,
        prompts: List[PromptRecord],
        sampling: SamplingConfig
    ) -> List[Spectrum]:
        """Generate spectra via API calls.
        
        Each prompt results in `sampling.n` independent API calls.
        Handles retries, rate limiting, and timeouts internally.
        """
        if not self._client:
            await self.initialize()

        sem = asyncio.Semaphore(self.config.max_concurrent)
        
        # Create concurrent tasks for all prompts and all trajectories
        tasks = []
        for prompt in prompts:
            for _ in range(sampling.n):
                tasks.append(self._generate_single_trajectory(prompt, sampling, sem))
                
        # Gather all responses
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Group responses by prompt ID
        responses_by_prompt: Dict[str, List[ResponseRecord]] = {p.id: [] for p in prompts}
        
        errors = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Trajectory generation failed with exception: {res}")
                errors.append(res)
                continue
            if isinstance(res, ResponseRecord):
                responses_by_prompt[res.prompt_id].append(res)
                
        # Build Spectra objects
        spectra = []
        for prompt in prompts:
            prompt_responses = responses_by_prompt[prompt.id]
            if not prompt_responses:
                if errors:
                    raise errors[0]
                raise ProviderError(f"All generation trajectories failed for prompt '{prompt.id}'")
                
            spectra.append(Spectrum(
                prompt=prompt,
                responses=prompt_responses,
                sampling_config=sampling,
                provider_info=self.info(),
                created_at=time.time()
            ))
            
        return spectra

    async def _generate_single_trajectory(
        self,
        prompt: PromptRecord,
        sampling: SamplingConfig,
        sem: asyncio.Semaphore
    ) -> ResponseRecord:
        """Helper to call HTTP API and return a single ResponseRecord."""
        url = self.config.base_url
        if not url.endswith("/chat/completions") and not url.endswith("/completions"):
            url = f"{url.rstrip('/')}/chat/completions"
            
        messages = []
        if prompt.system_prompt:
            messages.append({"role": "system", "content": prompt.system_prompt})
        messages.append({"role": "user", "content": prompt.user_prompt})
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "n": 1,
        }
        
        if sampling.top_k > 0:
            payload["top_k"] = sampling.top_k
        if sampling.min_p > 0.0:
            payload["min_p"] = sampling.min_p
        if sampling.repetition_penalty != 1.0:
            payload["repetition_penalty"] = sampling.repetition_penalty
        if sampling.stop_sequences:
            payload["stop"] = sampling.stop_sequences
        if sampling.seed is not None:
            payload["seed"] = sampling.seed

        async def _make_request():
            async with sem:
                start_time = time.time()
                try:
                    response = await self._client.post(url, json=payload)
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    status_code = e.response.status_code
                    if status_code in (401, 403):
                        raise ProviderAuthError(f"API Authentication failed: {e.response.text}")
                    elif status_code == 429:
                        retry_after = 0.0
                        if "retry-after" in e.response.headers:
                            try:
                                retry_after = float(e.response.headers["retry-after"])
                            except ValueError:
                                pass
                        raise ProviderRateLimitError(f"API Rate limit exceeded: {e.response.text}", retry_after=retry_after)
                    else:
                        raise ProviderError(f"API HTTP error {status_code}: {e.response.text}")
                except httpx.TimeoutException as e:
                    raise ProviderTimeoutError(f"API request timed out: {e}")
                except httpx.RequestError as e:
                    raise ProviderConnectionError(f"API connection error: {e}")
                    
                latency_ms = (time.time() - start_time) * 1000.0
                return response.json(), latency_ms

        res_json, latency_ms = await retry_async(
            _make_request,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_delay,
            retryable_exceptions=(ProviderRateLimitError, ProviderTimeoutError, ProviderConnectionError)
        )
        
        choices = res_json.get("choices", [])
        if not choices:
            raise ProviderError(f"Empty choices returned from API: {res_json}")
            
        choice = choices[0]
        text = choice.get("message", {}).get("content", "")
        finish_reason = choice.get("finish_reason", "")
        
        usage = res_json.get("usage", {})
        token_count = usage.get("completion_tokens", 0)
        
        # Extract thinking and answer values
        thinking_trace = ""
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if think_match:
            thinking_trace = think_match.group(1).strip()
            
        extracted_answer = text.strip()
        ans_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if ans_match:
            extracted_answer = ans_match.group(1).strip()
        else:
            boxed_match = re.search(r"\\boxed\{(.*?)\}", text)
            if boxed_match:
                extracted_answer = boxed_match.group(1).strip()
                
        return ResponseRecord(
            id=f"resp-{uuid.uuid4()}",
            prompt_id=prompt.id,
            text=text,
            thinking_trace=thinking_trace,
            extracted_answer=extracted_answer,
            token_count=token_count,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            metadata={"model": res_json.get("model", self.config.model)}
        )

    async def shutdown(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("Shutting down API provider.")

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="api",
            backend="api",
            model=self.config.model,
            metadata={
                "base_url": self.config.base_url,
                "max_retries": self.config.max_retries,
                "timeout": self.config.timeout,
            }
        )
