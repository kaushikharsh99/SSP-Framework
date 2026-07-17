"""Custom exception hierarchy for the Spectrum Generation Engine.

All engine exceptions inherit from SpectrumEngineError so callers
can catch broadly or narrowly as needed.
"""


class SpectrumEngineError(Exception):
    """Base exception for all Spectrum Engine errors."""
    pass


# --- Provider Errors ---

class ProviderError(SpectrumEngineError):
    """Base exception for provider-related failures."""
    pass


class ProviderConnectionError(ProviderError):
    """Failed to connect to the provider endpoint."""
    pass


class ProviderRateLimitError(ProviderError):
    """Provider rate limit exceeded. Caller should back off and retry."""
    def __init__(self, message: str = "Rate limit exceeded", retry_after: float = 0.0):
        super().__init__(message)
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    """Provider request timed out."""
    pass


class ProviderAuthError(ProviderError):
    """Authentication failed (invalid API key, expired token, etc.)."""
    pass


# --- Config Errors ---

class ConfigError(SpectrumEngineError):
    """Configuration loading or validation error."""
    pass


# --- Storage Errors ---

class StorageError(SpectrumEngineError):
    """Storage read/write failure."""
    pass


class CheckpointError(StorageError):
    """Checkpoint save/load failure."""
    pass


# --- Scheduler Errors ---

class SchedulerError(SpectrumEngineError):
    """Scheduler orchestration failure."""
    pass
