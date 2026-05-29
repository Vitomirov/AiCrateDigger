"""Reusable guard scopes for provider call sites."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from app.core.quota.kinds import QuotaKind
from app.core.quota.service import assert_quota_available, record_quota_usage


@asynccontextmanager
async def openai_extract_quota_scope() -> AsyncIterator[None]:
    """Assert extract quota, yield for the OpenAI call, record on success."""
    await assert_quota_available(QuotaKind.OPENAI_EXTRACT)
    try:
        yield
    except Exception:
        raise
    else:
        await record_quota_usage(QuotaKind.OPENAI_EXTRACT)
