"""Live metrics tracker for generation runs."""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("spectrum-engine.metrics")


class MetricsTracker:
    """Tracks live generation metrics and logs periodic summaries."""

    def __init__(self, total_prompts: int = 0, log_interval: int = 10):
        self.total_prompts = total_prompts
        self.log_interval = log_interval
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.total_tokens = 0
        self.total_responses = 0
        self.total_retries = 0
        self._start_time = time.time()
        self._last_log = 0

    def record_success(self, num_responses: int, num_tokens: int) -> None:
        """Record a successful spectrum generation."""
        self.completed += 1
        self.total_responses += num_responses
        self.total_tokens += num_tokens
        self._maybe_log()

    def record_failure(self) -> None:
        """Record a failed generation attempt."""
        self.failed += 1

    def record_skip(self) -> None:
        """Record a skipped prompt (already completed)."""
        self.skipped += 1

    def record_retry(self) -> None:
        """Record a retry attempt."""
        self.total_retries += 1

    @property
    def elapsed(self) -> float:
        return time.time() - self._start_time

    @property
    def tokens_per_second(self) -> float:
        return self.total_tokens / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def prompts_per_minute(self) -> float:
        return (self.completed * 60) / self.elapsed if self.elapsed > 0 else 0.0

    def _maybe_log(self) -> None:
        """Log progress at configured intervals."""
        if self.completed - self._last_log >= self.log_interval:
            self._last_log = self.completed
            pct = (self.completed / self.total_prompts * 100) if self.total_prompts > 0 else 0
            logger.info(
                f"Progress: {self.completed}/{self.total_prompts} ({pct:.1f}%) | "
                f"Tokens/s: {self.tokens_per_second:.0f} | "
                f"Prompts/min: {self.prompts_per_minute:.1f} | "
                f"Failed: {self.failed} | Retries: {self.total_retries}"
            )

    def summary(self) -> str:
        """Return a human-readable summary string."""
        return (
            f"\n{'='*60}\n"
            f"Generation Complete\n"
            f"{'='*60}\n"
            f"  Completed:    {self.completed}/{self.total_prompts}\n"
            f"  Skipped:      {self.skipped}\n"
            f"  Failed:       {self.failed}\n"
            f"  Responses:    {self.total_responses}\n"
            f"  Tokens:       {self.total_tokens:,}\n"
            f"  Elapsed:      {self.elapsed:.1f}s\n"
            f"  Tokens/sec:   {self.tokens_per_second:.0f}\n"
            f"  Retries:      {self.total_retries}\n"
            f"{'='*60}"
        )
