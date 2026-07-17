"""Unit tests for the APIProvider class.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from generation.spectrum_engine.core.config import ProviderConfig
from generation.spectrum_engine.core.exceptions import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderConnectionError,
    ProviderError
)
from generation.spectrum_engine.core.types import PromptRecord, SamplingConfig
from generation.spectrum_engine.providers.api_provider import APIProvider


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def api_config():
    return ProviderConfig(
        type="api",
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="test-model",
        timeout=10,
        max_retries=2,
        retry_delay=0.01,
        max_concurrent=4
    )


@pytest.mark.anyio
async def test_api_provider_success(api_config):
    provider = APIProvider(api_config)
    await provider.initialize()

    # Mock response
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "<think>solving...</think>\n<answer>42</answer>"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "completion_tokens": 15,
            "prompt_tokens": 5
        },
        "model": "returned-model"
    }

    # Mock the post method of the client
    provider._client.post = AsyncMock(return_value=mock_response)

    prompt = PromptRecord.create("sys", "user", id="p1")
    sampling = SamplingConfig(temperature=0.7, n=2)  # Should spawn 2 calls

    spectra = await provider.generate([prompt], sampling)
    
    assert len(spectra) == 1
    spec = spectra[0]
    assert len(spec.responses) == 2
    assert spec.responses[0].text == "<think>solving...</think>\n<answer>42</answer>"
    assert spec.responses[0].thinking_trace == "solving..."
    assert spec.responses[0].extracted_answer == "42"
    assert spec.responses[0].token_count == 15
    assert spec.responses[0].finish_reason == "stop"
    
    # Assert correct number of mock calls
    assert provider._client.post.call_count == 2
    
    await provider.shutdown()


@pytest.mark.anyio
async def test_api_provider_auth_error(api_config):
    provider = APIProvider(api_config)
    await provider.initialize()

    # Mock response throwing HTTPStatusError for 401
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    
    # raise_for_status throws HTTPStatusError
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        message="Auth failure",
        request=MagicMock(spec=httpx.Request),
        response=mock_response
    )

    provider._client.post = AsyncMock(return_value=mock_response)

    prompt = PromptRecord.create("sys", "user", id="p1")
    sampling = SamplingConfig(n=1)

    # Auth error should not be retried, failing immediately
    with pytest.raises(ProviderAuthError):
        await provider.generate([prompt], sampling)
        
    await provider.shutdown()


@pytest.mark.anyio
async def test_api_provider_rate_limit_retry(api_config):
    provider = APIProvider(api_config)
    await provider.initialize()

    # Create one mock failing with 429, then succeeding
    mock_response_fail = MagicMock(spec=httpx.Response)
    mock_response_fail.status_code = 429
    mock_response_fail.headers = {"retry-after": "0.01"}
    mock_response_fail.text = "Too Many Requests"
    mock_response_fail.raise_for_status.side_effect = httpx.HTTPStatusError(
        message="Rate limit",
        request=MagicMock(spec=httpx.Request),
        response=mock_response_fail
    )

    mock_response_success = MagicMock(spec=httpx.Response)
    mock_response_success.status_code = 200
    mock_response_success.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Success!"}}],
        "usage": {"completion_tokens": 5}
    }

    # Side effect: first call raises status error, second succeeds
    provider._client.post = AsyncMock(side_effect=[mock_response_fail, mock_response_success])

    prompt = PromptRecord.create("sys", "user", id="p1")
    sampling = SamplingConfig(n=1)

    spectra = await provider.generate([prompt], sampling)
    
    assert len(spectra) == 1
    assert spectra[0].responses[0].text == "Success!"
    assert provider._client.post.call_count == 2
    
    await provider.shutdown()
