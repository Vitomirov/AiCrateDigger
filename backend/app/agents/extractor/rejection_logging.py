"""Structured INFO logs for extractor rejections."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

logger = logging.getLogger("app.agents.extractor")


def log_extractor_reject(
    url: str,
    reason: str,
    *,
    artist_match: float | None = None,
    detail: str | None = None,
) -> None:
    domain = urlparse(url).netloc.lower() if url else ""
    extras: dict = {
        "stage": "extractor",
        "status": "fail",
        "reason": reason,
        "url": url,
        "domain": domain,
    }
    if artist_match is not None:
        extras["artist_match"] = round(artist_match, 2)
    if detail:
        extras["detail"] = detail
    logger.info("extractor_reject", extra=extras)
