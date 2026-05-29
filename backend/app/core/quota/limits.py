"""Map quota kinds to ``Settings`` limits (0 = unlimited for that bucket)."""

from __future__ import annotations

from app.core.config import Settings
from app.core.quota.kinds import QuotaKind


def daily_limit_for_kind(settings: Settings, kind: QuotaKind) -> int:
    """Return the configured daily maximum; ``0`` disables enforcement for this kind."""
    return {
        QuotaKind.PARSE: settings.global_daily_quota_parse_max,
        QuotaKind.TAVILY: settings.global_daily_quota_tavily_max,
        QuotaKind.OPENAI_EXTRACT: settings.global_daily_quota_openai_extract_max,
    }[kind]
