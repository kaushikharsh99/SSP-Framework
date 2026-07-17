"""Unit tests for the VLLMProvider class.
"""

import sys
from unittest.mock import MagicMock, patch

# Create mock classes for vLLM structures before loading the module
class MockCompletionOutput:
    def __init__(self, text, token_ids, finish_reason="stop"):
        self.text = text
        self.token_ids = token_ids
        self.finish_reason = finish_reason

class MockRequestOutput:
    def __init__(self, outputs):
        self.outputs = outputs

# Set up mock vllm module in sys.modules so imports do not fail on CPU-only envs
mock_vllm = MagicMock()
mock_vllm.SamplingParams = MagicMock()
sys.modules["vllm"] = mock_vllm

import pytest
from generation.spectrum_engine.core.config import ProviderConfig
from generation.spectrum_engine.core.types import PromptRecord, SamplingConfig
from generation.spectrum_engine.providers.vllm_provider import VLLMProvider


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def vllm_config():
    return ProviderConfig(
        type="vllm",
        model_path="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        dtype="auto",
        trust_remote_code=False,
        max_model_len=1024
    )


@pytest.mark.anyio
async def test_vllm_provider_generate(vllm_config):
    # Mock LLM instance and tokenizer
    mock_llm_instance = MagicMock()
    mock_vllm.LLM.return_value = mock_llm_instance
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<mock_chat>user_prompt</mock_chat>"
    mock_llm_instance.get_tokenizer.return_value = mock_tokenizer

    # Mock output returned by generate
    mock_completion_1 = MockCompletionOutput(
        text="<think>thinking</think>\n<answer>10</answer>",
        token_ids=[1, 2, 3]
    )
    mock_completion_2 = MockCompletionOutput(
        text="<think>thinking</think>\n<answer>10</answer>",
        token_ids=[1, 2, 3]
    )
    
    mock_request_output = MockRequestOutput([mock_completion_1, mock_completion_2])
    mock_llm_instance.generate.return_value = [mock_request_output]

    # Instantiate provider
    provider = VLLMProvider(vllm_config)
    await provider.initialize()

    # Generate
    prompt = PromptRecord.create("sys", "problem", id="p1")
    sampling = SamplingConfig(temperature=0.8, n=2)

    spectra = await provider.generate([prompt], sampling)

    # Verify
    assert mock_vllm.LLM.call_count == 1
    assert mock_llm_instance.generate.call_count == 1
    
    assert len(spectra) == 1
    spec = spectra[0]
    assert len(spec.responses) == 2
    assert spec.responses[0].text == "<think>thinking</think>\n<answer>10</answer>"
    assert spec.responses[0].thinking_trace == "thinking"
    assert spec.responses[0].extracted_answer == "10"
    assert spec.responses[0].token_count == 3
    assert spec.responses[0].finish_reason == "stop"

    await provider.shutdown()
