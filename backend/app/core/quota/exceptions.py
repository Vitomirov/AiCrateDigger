"""Quota domain errors (mapped to HTTP responses in ``app.main``)."""

from __future__ import annotations

from app.core.quota.kinds import QuotaKind


class QuotaExceededError(Exception):
    """Raised when a global daily cap would be exceeded."""

    def __init__(
        self,
        kind: QuotaKind,
        *,
        limit: int,
        current: int,
        reason: str = "daily_cap",
    ) -> None:
        self.kind = kind
        self.limit = limit
        self.current = current
        self.reason = reason
        super().__init__(f"{kind.value} daily quota exceeded ({current}/{limit})")


class QuotaUnavailableError(Exception):
    """Raised when quota enforcement is enabled but Redis is unreachable."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)
