"""Per-request Tavily circuit breaker.

Tavily can hard-throttle the account (HTTP 432/433) for many minutes. Without a
breaker, the pipeline keeps cycling through every relaxation query at every
geo tier — each retrying 5 × ~14 s — so a single failing request burns minutes
of wall time and credits before surfacing.

Scoped to one pipeline run via :func:`tavily_circuit_breaker_scope`. After
``failure_threshold`` complete retry exhaustions, the breaker trips and every
subsequent call returns ``None`` immediately (no HTTP cost, no latency).
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TavilyCircuitBreaker:
    failure_threshold: int = 2
    consecutive_failures: int = 0
    tripped: bool = False

    def is_open(self) -> bool:
        return self.tripped

    def record_failure(self, *, reason: str = "retry_exhausted") -> None:
        if self.tripped:
            return
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_threshold:
            self.tripped = True
            logger.warning(
                "tavily_circuit_breaker_tripped",
                extra={
                    "stage": "tavily",
                    "failures": self.consecutive_failures,
                    "threshold": self.failure_threshold,
                    "reason": reason,
                },
            )

    def record_success(self) -> None:
        if self.consecutive_failures or self.tripped:
            self.consecutive_failures = 0
            self.tripped = False


#: Default no-op breaker used outside pipeline scope (tests, ad-hoc scripts).
_NOOP_BREAKER = TavilyCircuitBreaker(failure_threshold=10**9)

_current: ContextVar[TavilyCircuitBreaker] = ContextVar(
    "tavily_circuit_breaker",
    default=_NOOP_BREAKER,
)


def get_breaker() -> TavilyCircuitBreaker:
    return _current.get()


@contextlib.contextmanager
def tavily_circuit_breaker_scope(
    failure_threshold: int = 2,
) -> Iterator[TavilyCircuitBreaker]:
    """Bind a fresh breaker to the current async context for one pipeline run."""
    breaker = TavilyCircuitBreaker(failure_threshold=max(1, int(failure_threshold)))
    token = _current.set(breaker)
    try:
        yield breaker
    finally:
        _current.reset(token)
