"""LLM step 2 — turn raw web-search snippets into ``Listing`` objects.

One LLM call with lenient extraction (unknown price/currency allowed), then
deterministic assembly. Returns :class:`ExtractListingsReport` with diagnostics.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from openai import AsyncOpenAI
from pydantic import ValidationError

from app.config import get_settings
from app.domain.listing_schema import Listing

logger = logging.getLogger(__name__)

_LLM_MAX_INPUT = 15

_SNIPPET_CHAR_CAP = 1500

EXTRACTOR_SYSTEM_PROMPT = """You extract product listings from search snippets (title + content) for vinyl record shops.

RULES
- You MUST output exactly one JSON object per input row, in the same order, with matching "url".
- ALWAYS emit a listing for every input row unless the snippet has no product URL (never skip a row you were given).
- **price**: Extract from snippet if present; otherwise use `null` or `0.0` (unknown is allowed).
- **currency**: Use ISO 4217 (EUR, GBP, USD, …) when clear from € £ $ or text; otherwise `null` (caller defaults to EUR).
- **in_stock**:
    - `true` only if the snippet clearly says available / in stock / can buy / add to cart.
    - `false` ONLY if the snippet explicitly says sold out / out of stock / unavailable / nicht verfügbar.
    - If availability is unclear, use `null` — do NOT guess "out of stock".
- **title**: Prefer the real product title from the snippet; never invent an unrelated title.
- **store**: Shop name or domain hint if obvious; else `null`.

OUTPUT MUST BE JSON ONLY:
{
  "listings": [
    {
      "url": "string",
      "title": "string",
      "price": number | null,
      "currency": "string | null",
      "in_stock": true | false | null,
      "store": "string | null"
    }
  ]
}
"""


@dataclass
class ExtractListingsReport:
    listings: list[Listing]
    diagnostic: dict[str, Any] = field(default_factory=dict)


def _normalize_domain(url_or_domain: str) -> str | None:
    if not url_or_domain:
        return None
    candidate = url_or_domain.strip()
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    try:
        netloc = urlsplit(candidate).netloc.lower().strip()
    except ValueError:
        return None
    if not netloc:
        return None
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc.split(":", 1)[0]


def _normalize_allowed_domains(domains: set[str]) -> set[str]:
    out = set()
    for d in domains:
        n = _normalize_domain(d)
        if n:
            out.add(n)
    return out


def _host_matches_whitelist(host: str, allowed: set[str]) -> bool:
    if not host:
        return False
    h = host.lower()
    if h.startswith("www."):
        h = h[4:]
    return h in allowed or any(h.endswith("." + d) for d in allowed)


async def _llm_extract(
    candidates: list[dict[str, Any]],
    diagnostic: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    """Returns (parsed listing dicts, raw message content string)."""
    if not candidates:
        return [], ""

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "json\n" + json.dumps({"listings": candidates}, ensure_ascii=False),
            },
        ],
    )

    raw = response.choices[0].message.content or "{}"
    logger.info(
        "extract_listings_llm_raw_json",
        extra={"stage": "extractor", "raw_response": raw[:20000]},
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        diagnostic["json_parse_ok"] = False
        logger.warning(
            "extract_listings_json_decode_error",
            extra={"stage": "extractor", "error": str(exc), "raw_head": raw[:500]},
        )
        return [], raw

    items = [x for x in data.get("listings", []) if isinstance(x, dict)]
    return items, raw


def _coerce_price_currency(item: dict[str, Any]) -> tuple[float, str]:
    raw_p = item.get("price")
    try:
        p = float(raw_p) if raw_p is not None else 0.0
    except (TypeError, ValueError):
        p = 0.0
    if p < 0.0:
        p = 0.0

    cur = item.get("currency")
    s = str(cur).strip().upper() if cur is not None else ""
    if len(s) != 3 or not s.isalpha():
        s = "EUR"
    return p, s


def _coerce_in_stock(item: dict[str, Any]) -> bool:
    """Ambiguous → treat as available; only False when explicitly sold out."""
    v = item.get("in_stock")
    if v is True:
        return True
    if v is False:
        return False
    return True


async def extract_listings(
    raw_results: list[dict[str, Any]],
    *,
    artist: str | None,
    album: str,
    allowed_domains: set[str],
) -> ExtractListingsReport:
    diagnostic: dict[str, Any] = {
        "empty_reason": None,
        "prefilter_candidates": 0,
        "dropped_intent_mismatch": 0,
        "llm_rows_returned": 0,
        "json_parse_ok": True,
        "drop_url_not_in_candidates": 0,
        "drop_title_gate": 0,
        "drop_pydantic": 0,
    }

    if not raw_results or not album:
        diagnostic["empty_reason"] = "no_raw_results_or_album"
        return ExtractListingsReport(listings=[], diagnostic=diagnostic)

    allowed = _normalize_allowed_domains(allowed_domains)
    if not allowed:
        diagnostic["empty_reason"] = "empty_allowed_domains"
        return ExtractListingsReport(listings=[], diagnostic=diagnostic)

    artist_l = (artist or "").strip().lower() or None
    album_l = album.strip().lower()

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    dropped_intent = 0

    for r in raw_results:
        url = str(r.get("url") or "").strip()
        if not url or url in seen:
            continue

        domain = _normalize_domain(url)
        if not domain or not _host_matches_whitelist(domain, allowed):
            continue

        raw_title = str(r.get("title") or "").strip()
        raw_content = str(r.get("content") or "")[:_SNIPPET_CHAR_CAP]
        blob = f"{raw_title} {raw_content}".lower()

        if album_l not in blob:
            dropped_intent += 1
            continue
        if artist_l and artist_l not in blob:
            dropped_intent += 1
            continue

        seen.add(url)
        candidates.append(
            {
                "url": url,
                "title": raw_title,
                "content": raw_content,
            }
        )

    diagnostic["prefilter_candidates"] = len(candidates)
    diagnostic["dropped_intent_mismatch"] = dropped_intent

    logger.info(
        "extract_listings_prefilter",
        extra={
            "stage": "extractor",
            "candidate_count": len(candidates),
            "dropped_intent_mismatch": dropped_intent,
        },
    )

    if not candidates:
        diagnostic["empty_reason"] = "prefilter_zero_candidates_intent_mismatch"
        return ExtractListingsReport(listings=[], diagnostic=diagnostic)

    extracted, raw_json = await _llm_extract(candidates[:_LLM_MAX_INPUT], diagnostic)
    diagnostic["llm_rows_returned"] = len(extracted)
    if not raw_json.strip() or raw_json.strip() == "{}":
        diagnostic["empty_reason"] = "llm_empty_response"
    if not extracted:
        diagnostic["empty_reason"] = diagnostic.get("empty_reason") or "llm_returned_empty_listings_array"

    by_url_blob = {c["url"]: (c["title"] + " " + c["content"]).lower() for c in candidates}
    by_url_raw_title = {c["url"]: c["title"] for c in candidates}
    allowed_urls = set(by_url_blob.keys())

    results: dict[str, Listing] = {}

    for item in extracted:
        url = str(item.get("url") or "").strip()
        if url not in allowed_urls:
            diagnostic["drop_url_not_in_candidates"] += 1
            continue

        blob = by_url_blob.get(url, "")
        llm_title = (item.get("title") or "").strip()
        raw_title = (by_url_raw_title.get(url) or "").strip()

        if album_l in llm_title.lower():
            pick_title = llm_title
        elif album_l in raw_title.lower():
            pick_title = raw_title
        else:
            pick_title = f"{(artist or '').strip()} {album}".strip() or album

        if album_l not in pick_title.lower():
            diagnostic["drop_title_gate"] += 1
            continue
        if artist_l and artist_l not in pick_title.lower():
            diagnostic["drop_title_gate"] += 1
            continue

        domain = _normalize_domain(url) or ""
        store_raw = item.get("store")
        store = (str(store_raw).strip() if store_raw else "") or domain

        in_stock = _coerce_in_stock(item)
        price_v, currency = _coerce_price_currency(item)

        try:
            listing = Listing(
                title=pick_title,
                price=price_v,
                currency=currency,
                in_stock=in_stock,
                url=url,
                store=store,
            )
        except (ValidationError, TypeError, ValueError):
            diagnostic["drop_pydantic"] += 1
            continue

        existing = results.get(url)
        if existing is None or (listing.in_stock and not existing.in_stock):
            results[url] = listing

    out_list = list(results.values())
    if out_list:
        diagnostic["empty_reason"] = None
    elif extracted:
        diagnostic["empty_reason"] = "post_llm_all_dropped"
    else:
        diagnostic["empty_reason"] = diagnostic.get("empty_reason") or (
            "llm_json_empty_or_failed" if not diagnostic.get("json_parse_ok", True) else "llm_returned_empty_listings_array"
        )

    diagnostic["final_count"] = len(out_list)
    return ExtractListingsReport(listings=out_list, diagnostic=diagnostic)
