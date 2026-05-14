"""LLM-based release verifier for high-stakes local-shop results.

The Local-First Strike applies a +500-point bonus to indie ``local_shop``
results whose store sits in the resolved city. That bonus must NEVER fire for
a row whose title names a different release — otherwise a Bucharest indie
selling "Spacekid" would crowd out an actual "Andrew Red Hand" listing on
HHV. This module runs a single batched LLM call to label each candidate as
``confirmed`` | ``reject`` | ``unsure`` so the pipeline can drop the rejects
and gate the bonus.

Contract:
- Deterministic (``temperature=0``, JSON-only).
- Single HTTP call regardless of input size.
- Failure-tolerant: returns an empty dict on any error so the caller falls
  back to "unsure" semantics (do not break, do not bonus).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

from app.config import get_settings
from app.domain.listing_schema import Listing

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AlbumMatchVerdict:
    url: str
    verdict: str  # "confirmed" | "reject" | "unsure"
    reason: str


_VERIFY_SYSTEM_PROMPT = """
You are AiCrateDigger's RELEASE VERIFIER.

INPUT: target artist + target album, plus a small list of candidate listings
(URL + title). Decide whether each listing names the SAME release.

DECISION RULES:
- "confirmed" — the title contains the target album name (or a clear
  abbreviation / translation / partial form) and, when an artist is given,
  there is a matching artist token. Edition/format suffixes are fine
  (LP, 12", reissue year, gatefold, deluxe). Different pressings of the same
  release count as confirmed.
- "reject" — the title clearly names a DIFFERENT release: another album by
  the same artist, a different artist entirely, a compilation that does
  not include the target release, or non-record merchandise (book, poster,
  T-shirt, equipment).
- "unsure" — the title is too generic (e.g. just the artist name, just a
  catalogue number, or truncated) to make a confident decision either way.
  Use this sparingly; prefer "confirmed" or "reject" when the title gives
  any usable cue.

OUTPUT — strict JSON only (no prose, no markdown):
{
  "verdicts": [
    {"url": "<exact url from input>", "verdict": "confirmed|reject|unsure", "reason": "<short>"}
  ]
}

`reason` must be ≤ 80 chars. There MUST be exactly one verdict per input
listing, identified by the exact ``url`` value.
""".strip()


async def verify_album_match(
    listings: list[Listing],
    *,
    artist: str | None,
    album_title: str,
) -> dict[str, AlbumMatchVerdict]:
    """Return ``{url -> AlbumMatchVerdict}``. Empty dict on missing inputs / failure."""
    if not listings or not (album_title or "").strip():
        return {}

    settings = get_settings()
    if not settings.openai_api_key:
        return {}

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    payload = {
        "target_artist": (artist or "").strip(),
        "target_album": album_title.strip(),
        "listings": [
            {
                "url": str(getattr(lst, "url", "") or ""),
                "title": (getattr(lst, "title", "") or "")[:240],
            }
            for lst in listings
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
        return {}

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
        return {}

    out: dict[str, AlbumMatchVerdict] = {}
    for row in data.get("verdicts", []) or []:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "").strip()
        verdict = str(row.get("verdict") or "").strip().lower()
        if not url or verdict not in {"confirmed", "reject", "unsure"}:
            continue
        out[url] = AlbumMatchVerdict(
            url=url,
            verdict=verdict,
            reason=str(row.get("reason") or "")[:120],
        )
    return out
