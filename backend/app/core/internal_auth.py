"""Shared secret between the Next.js BFF and this API."""

from __future__ import annotations

import hmac
import logging

from fastapi import HTTPException, Request

from app.core.config import get_settings

logger = logging.getLogger(__name__)

INTERNAL_API_SECRET_HEADER = "X-Internal-Api-Secret"


def is_internal_request(request: Request) -> bool:
    """Return whether the request presents the configured BFF secret."""
    settings = get_settings()
    expected = (settings.internal_api_secret or "").strip()
    if not expected:
        return False
    provided = (request.headers.get(INTERNAL_API_SECRET_HEADER) or "").strip()
    if not provided:
        return False
    return hmac.compare_digest(
        provided.encode("utf-8"),
        expected.encode("utf-8"),
    )


async def require_internal_api_secret(request: Request) -> None:
    """FastAPI dependency: reject paid routes when a secret is configured but missing/wrong."""
    settings = get_settings()
    expected = (settings.internal_api_secret or "").strip()
    if not expected:
        logger.debug(
            "internal_auth_bypass",
            extra={"stage": "internal_auth", "status": "bypass", "reason": "secret_unset"},
        )
        return

    if is_internal_request(request):
        return

    logger.info(
        "internal_auth_rejected",
        extra={"stage": "internal_auth", "status": "blocked"},
    )
    raise HTTPException(status_code=401, detail="Unauthorized")
