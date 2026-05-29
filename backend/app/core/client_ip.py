"""Client IP resolution for rate limiting (trust XFF only from the BFF)."""

from __future__ import annotations

from fastapi import Request

from app.core.internal_auth import is_internal_request


def _direct_peer_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _ip_from_forwarded_headers(request: Request) -> str | None:
    forwarded = request.headers.get("X-Forwarded-For") or request.headers.get("x-forwarded-for")
    if not forwarded:
        return None
    first = forwarded.split(",")[0].strip()
    return first or None


def resolve_client_ip(request: Request) -> str:
    """Rate-limit key IP.

    Honors ``X-Forwarded-For`` only when the BFF shared secret is valid so direct
    callers cannot spoof their way around per-IP limits.
    """
    if is_internal_request(request):
        return _ip_from_forwarded_headers(request) or _direct_peer_ip(request)
    return _direct_peer_ip(request)
