"""Retry utilities with exponential backoff."""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Callable, Optional, Tuple, Type

logger = logging.getLogger("spectrum-engine.utils.retry")


async def retry_async(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Any:
    """Execute an async function with exponential backoff retry.
    
    Args:
        fn: Async callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay cap.
        backoff_factor: Multiplier applied to delay after each retry.
        jitter: If True, add random jitter to prevent thundering herd.
        retryable_exceptions: Tuple of exception types that should trigger a retry.
        
    Returns:
        The return value of fn() on success.
        
    Raises:
        The last exception encountered after all retries are exhausted.
    """
    last_exception = None
    delay = base_delay

    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except retryable_exceptions as e:
            last_exception = e
            if attempt == max_retries:
                logger.error(f"All {max_retries} retries exhausted. Last error: {e}")
                raise
            
            actual_delay = min(delay, max_delay)
            if jitter:
                actual_delay *= (0.5 + random.random())
            
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                f"Retrying in {actual_delay:.1f}s..."
            )
            await asyncio.sleep(actual_delay)
            delay *= backoff_factor

    raise last_exception  # Should never reach here
