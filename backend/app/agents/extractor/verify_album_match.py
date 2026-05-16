"""LLM-based release verifier for high-stakes local-shop results.

The Local-First Strike applies a +500-point bonus to indie ``local_shop``
results whose store sits in the resolved city. That bonus must NEVER fire for
a row whose title names a different release — otherwise a Bucharest indie
selling "Spacekid" would crowd out an actual "Andrew Red Hand" listing on
HHV. This module runs deterministic snippet checks PLUS a batched LLM gate.

Contract:
- Deterministic (``temperature=0``, JSON-only).
- Single HTTP call per invocation (skipped when every row fails deterministic evidence).
- Failure-tolerant: returns an empty dict on missing keys / decoding errors so callers
  keep rows as "unsure" except where deterministic rejects apply.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

from app.config import get_settings
from app.domain.listing_schema import Listing
from app.agents.extractor.evidence_alignment import evidence_blob_matches_target_release
from app.services.tavily_service import normalize_url

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AlbumMatchVerdict:
    url: str
    verdict: str  # "confirmed" | "reject" | "unsure"
    reason: str


_VERIFY_SYSTEM_PROMPT = """
You are AiCrateDigger's RELEASE VERIFIER.

INPUT bundles `target_artist`, `target_album`, plus `listings` where each row has:
``url``, ``snippet_title``, ``snippet_excerpt``.

CRITICAL SOURCE-OF-TRUTH:
- Decide using ONLY tokens present inside ``snippet_title`` + ``snippet_excerpt``.
- If those fields describe a DIFFERENT album/artist/comp than the TARGET, you MUST return
  `"reject"` even if the PDP URL could hypothetically correspond to anything else offline.
- Never treat the shopper's hoped-for release name as factual unless BOTH target artist and
  target album wording (or unmistakable synonyms printed in THAT snippet excerpt) explicitly
  appear describing the SAME product listing.

DECISION RULES:

- `"confirmed"` — the snippet plainly names the SAME release as the TARGET: required album cues
  (title / unmistakable synonym) PLUS artist cues WHEN a target artist is provided. Casual
  format noise (LP, gatefold, 180g, year) ignored.
- `"reject"` — the snippet names another release/composer/collection/non-music SKU, contradicts the
  target album, OR when a target artist exists the snippet never references that artist but instead
  references someone else.
- `"unsure"` — the snippet is too sparse (only store nav, price, or generic "vinyl" text) to align
  with any specific release. Use rarely; default to reject when the snippet clearly describes a
  different record.

OUTPUT — strict JSON only (no prose, no markdown):
{
  "verdicts": [
    {"url": "<exact url from input>", "verdict": "confirmed|reject|unsure", "reason": "<short>"}
  ]
}

`reason` must be ≤ 80 chars. There MUST be exactly one verdict per survivor listing.
""".strip()


def _listing_evidence_blob_lc(lst: Listing) -> str:
    parts: list[str] = []
    sn = getattr(lst, "source_snippet", None)
    if isinstance(sn, str) and sn.strip():
        parts.append(sn.strip())
    if lst.title:
        parts.append(str(lst.title))
    return " ".join(parts).strip().lower()


async def verify_album_match(
    listings: list[Listing],
    *,
    artist: str | None,
    album_title: str,
) -> dict[str, AlbumMatchVerdict]:
    """Return ``{url -> AlbumMatchVerdict}``. Deterministic rejects win over any LLM output."""
    if not listings or not (album_title or "").strip():
        return {}

    out: dict[str, AlbumMatchVerdict] = {}
    survivors: list[Listing] = []

    for lst in listings:
        url = str(getattr(lst, "url", "") or "").strip()
        if not url:
            continue
        blob_lc = _listing_evidence_blob_lc(lst)
        if not evidence_blob_matches_target_release(
            blob_lc,
            artist=artist,
            album=album_title,
        ):
            out[url] = AlbumMatchVerdict(
                url=url,
                verdict="reject",
                reason="deterministic_snippet_misses_target",
            )
            nk = normalize_url(url)
            if nk and nk != url:
                out[nk] = out[url]
            continue
        survivors.append(lst)

    if not survivors:
        return out

    settings = get_settings()
    if not settings.openai_api_key:
        for lst in survivors:
            u = str(lst.url)
            out[u] = AlbumMatchVerdict(url=u, verdict="unsure", reason="no_openai_key")
            nk = normalize_url(u)
            if nk and nk != u:
                out[nk] = out[u]
        return out

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    payload = {
        "target_artist": (artist or "").strip(),
        "target_album": album_title.strip(),
        "listings": [
            {
                "url": str(getattr(lst, "url", "") or ""),
                "snippet_title": (getattr(lst, "title", "") or "")[:280],
                "snippet_excerpt": (
                    str(
                        getattr(lst, "source_snippet", None)
                        or (getattr(lst, "title", None) or "")
                    )[:600]
                ),
            }
            for lst in survivors
        ],
    }

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _VERIFY_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "verify_album_match_llm_failed",
            extra={"stage": "verify_album_match", "reason": str(exc)},
        )
        for lst in survivors:
            u = str(lst.url)
            out[u] = AlbumMatchVerdict(url=u, verdict="unsure", reason="llm_error")
            nk = normalize_url(u)
            if nk and nk != u:
                out[nk] = out[u]
        return out

    raw = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(
            "verify_album_match_json_decode_error",
            extra={
                "stage": "verify_album_match",
                "reason": str(exc),
                "raw_head": raw[:200],
            },
        )
        for lst in survivors:
            u = str(lst.url)
            out[u] = AlbumMatchVerdict(url=u, verdict="unsure", reason="json_decode")
            nk = normalize_url(u)
            if nk and nk != u:
                out[nk] = out[u]
        return out

    for row in data.get("verdicts", []) or []:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "").strip()
        verdict = str(row.get("verdict") or "").strip().lower()
        if not url or verdict not in {"confirmed", "reject", "unsure"}:
            continue
        av = AlbumMatchVerdict(
            url=url,
            verdict=verdict,
            reason=str(row.get("reason") or "")[:120],
        )
        out[url] = av
        nk = normalize_url(url)
        if nk and nk != url:
            out[nk] = av
    return out
