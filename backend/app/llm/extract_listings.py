"""LLM step 2 — turn raw web-search snippets into validated `Listing` objects.

Three deterministic stages around ONE LLM call:

    1. pre-filter    — domain whitelist + currency-hint sanity. Pure Python.
    2. LLM extract   — gpt-4o-mini cleans price / currency / in_stock / title.
                       The model NEVER decides validity.
    3. hard validate — strict substring match (artist + album), domain
                       re-check, Pydantic `Listing` schema (price > 0,
                       3-letter currency, etc.). Pure Python.

There is NO RAG, NO marketplace DB, NO scoring, NO fuzzy match, NO retries.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import urlsplit

from openai import AsyncOpenAI
from pydantic import ValidationError

from app.config import get_settings
from app.domain.listing_schema import Listing

logger = logging.getLogger(__name__)

# Cap candidates fed into the single LLM call. Tavily's per-domain cap means
# survivors above this are highly diminishing returns.
_LLM_MAX_INPUT = 15

# Optional pre-filter: prefer snippets that mention money (reduces junk) but
# do not drop allowlisted product URLs when no hint — Tavily excerpts often omit €.
_CURRENCY_HINT_PATTERN = re.compile(
    r"(?:€|£|\$|¥|kr\b|zł\b|"
    r"EUR|GBP|USD|SEK|DKK|NOK|PLN|CZK|RSD|HUF|BGN|RON|CHF|HRK|ISK)",
    re.IGNORECASE,
)

# Snippet content cap — extractor input length is the dominant cost driver.
_SNIPPET_CHAR_CAP = 1500

EXTRACTOR_SYSTEM_PROMPT = """You are a strict listing extractor. Convert each raw \
search snippet into a structured candidate. You DO NOT decide validity — the caller \
filters deterministically. NEVER invent data.

OUTPUT JSON (single object, no markdown, no commentary):
{
  "listings": [
    {
      "url":      "string",
      "title":    "string",
      "price":    number | null,
      "currency": "string (ISO 4217, e.g. EUR/GBP/USD/SEK/DKK/NOK/PLN/CZK/RSD) | null",
      "in_stock": true | false | null,
      "store":    "string | null"
    }
  ]
}

RULES
- Emit exactly ONE entry per input snippet, preserving `url` verbatim.
- TITLE: copy the exact listing / product title. Strip ONLY marketplace
  boilerplate ("| HHV", "Buy at JPC", "- Recordsale", "kaufen", "Online Shop").
  Do NOT translate, paraphrase, or rewrite.
- PRICE: parse the numeric listing price from title+content
  ("29,99 €" → 29.99, "£24.00" → 24.00, "1.890 RSD" → 1890).
  Always use period as decimal separator. If no clear price → null.
- CURRENCY: 3-letter ISO 4217 derived from the symbol or text:
    €  → EUR
    £  → GBP
    $  → USD
    kr → SEK / DKK / NOK (use whichever the page hints at)
    written codes (RSD, PLN, CZK, …) → use as-is.
  If you cannot tell → null.
- IN_STOCK:
    true   ⇔ snippet explicitly says available ("in stock", "auf Lager",
            "verfügbar", "available", "preorder available", "na zalogi").
    false  ⇔ explicit sold-out signal ("sold out", "ausverkauft",
            "indisponible", "épuisé", "agotado", "rasprodato").
    null   otherwise.
- STORE: domain or readable store name (e.g. "hhv.de", "JPC", "Recordsale").
  If unsure → null.

Output JSON only. Null over fabrication."""


# ---------------------------------------------------------------------------
# Domain helpers (deterministic, no logic beyond URL parsing)
# ---------------------------------------------------------------------------

def _normalize_domain(url_or_domain: str) -> str | None:
    """Return a stable lowercase base domain (no scheme, no `www.`, no port)."""
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
    netloc = netloc.split(":", maxsplit=1)[0]
    return netloc or None


def _normalize_allowed_domains(domains: set[str]) -> set[str]:
    out: set[str] = set()
    for raw in domains:
        norm = _normalize_domain(raw)
        if norm:
            out.add(norm)
    return out


def _host_matches_whitelist(host: str, allowed: set[str]) -> bool:
    """True if ``host`` equals a whitelisted store domain or is a subdomain of one."""
    if not host:
        return False
    h = host.strip().lower()
    if h.startswith("www."):
        h = h[4:]
    if h in allowed:
        return True
    return any(h.endswith("." + d) for d in allowed)


def _has_currency_hint(text: str) -> bool:
    return bool(_CURRENCY_HINT_PATTERN.search(text or ""))


# ---------------------------------------------------------------------------
# LLM call (the ONLY LLM use in this module)
# ---------------------------------------------------------------------------

async def _llm_extract(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One JSON-mode call. Returns the LLM's raw `listings` array or []."""
    if not candidates:
        return []

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
                "content": json.dumps({"listings": candidates}, ensure_ascii=False),
            },
        ],
    )

    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)
    items = data.get("listings", []) or []
    return [it for it in items if isinstance(it, dict)]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def extract_listings(
    raw_results: list[dict[str, Any]],
    *,
    artist: str | None,
    album: str,
    allowed_domains: set[str],
) -> list[Listing]:
    """Convert noisy search snippets into clean, validated `Listing` objects.

    Parameters
    ----------
    raw_results
        Tavily-shaped dicts with at least `url`, `title`, `content`.
    artist
        Parsed artist string. Used as a case-insensitive substring gate
        against each candidate's title. `None` skips the artist gate.
    album
        Canonical album title (post-Discogs). Required substring gate.
    allowed_domains
        Whitelist of base domains (e.g. ``hhv.de``). Hosts must match exactly
        or be subdomains (e.g. ``shop.hhv.de``).
    """
    if not raw_results or not (album or "").strip():
        return []

    allowed = _normalize_allowed_domains(allowed_domains)
    if not allowed:
        logger.warning(
            "extract_listings_empty_allowed_domains",
            extra={"stage": "extractor", "status": "empty", "raw_count": len(raw_results)},
        )
        return []

    # ===== Stage 1: deterministic pre-filter =====
    candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for raw in raw_results:
        url = str(raw.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        domain = _normalize_domain(url)
        if not domain or not _host_matches_whitelist(domain, allowed):
            continue
        seen_urls.add(url)
        candidates.append(
            {
                "url": url,
                "title": str(raw.get("title") or "").strip(),
                "content": str(raw.get("content") or "")[:_SNIPPET_CHAR_CAP],
            }
        )

    with_currency = sum(
        1
        for c in candidates
        if _has_currency_hint(f"{c.get('title', '')} {c.get('content', '')}")
    )
    logger.info(
        "extract_listings_prefilter",
        extra={
            "stage": "extractor",
            "status": "success" if candidates else "empty",
            "raw_count": len(raw_results),
            "candidate_count": len(candidates),
            "candidates_with_currency_hint": with_currency,
        },
    )

    if not candidates:
        logger.warning(
            "extract_listings_prefilter_empty",
            extra={"stage": "extractor", "status": "empty", "input": len(raw_results)},
        )
        return []

    # ===== Stage 2: single LLM extraction call =====
    extracted = await _llm_extract(candidates[:_LLM_MAX_INPUT])
    settings = get_settings()
    logger.info(
        "extract_listings_llm",
        extra={
            "stage": "extractor",
            "status": "success" if extracted else "empty",
            "llm_row_count": len(extracted),
            "candidates_to_llm": min(len(candidates), _LLM_MAX_INPUT),
        },
    )

    # ===== Stage 3: hard validation =====
    artist_lower: str | None = (artist or "").strip().lower() or None
    album_lower: str = album.strip().lower()

    by_url: dict[str, Listing] = {}

    for item in extracted:
        url = str(item.get("url") or "").strip()
        # LLM hallucinated a URL we never sent → drop.
        if url not in seen_urls:
            continue
        domain = _normalize_domain(url)
        if not domain or not _host_matches_whitelist(domain, allowed):
            continue

        title = str(item.get("title") or "").strip()
        if not title:
            continue
        title_lower = title.lower()
        if artist_lower and artist_lower not in title_lower:
            logger.info(
                "extract_listings_gate_drop",
                extra={
                    "stage": "extractor",
                    "status": "reject",
                    "reason": "artist_not_in_title",
                    "url": url[:400],
                    "title_sample": title[:120],
                },
            )
            continue
        if album_lower not in title_lower:
            logger.info(
                "extract_listings_gate_drop",
                extra={
                    "stage": "extractor",
                    "status": "reject",
                    "reason": "album_not_in_title",
                    "url": url[:400],
                    "title_sample": title[:120],
                },
            )
            continue

        currency_raw = item.get("currency")
        currency = str(currency_raw).strip().upper() if currency_raw else ""

        in_stock_raw = item.get("in_stock")
        in_stock = in_stock_raw if isinstance(in_stock_raw, bool) else False

        store_raw = item.get("store")
        store = (str(store_raw).strip() if store_raw else "") or domain

        price_v = item.get("price")
        if settings.debug:
            try:
                pnum = float(price_v) if price_v is not None else 0.0
            except (TypeError, ValueError):
                pnum = 0.0
            if pnum <= 0 or len(currency) != 3:
                price_v = 0.01
                currency = "EUR"

        try:
            listing = Listing(
                title=title,
                price=price_v,
                currency=currency,
                in_stock=in_stock,
                url=url,
                store=store,
            )
        except (ValidationError, TypeError, ValueError):
            logger.info(
                "extract_listings_schema_drop",
                extra={
                    "stage": "extractor",
                    "status": "reject",
                    "reason": "Listing_validation_failed",
                    "url": url[:500],
                },
            )
            continue

        # ===== Stage 4: dedup by URL — prefer in_stock survivor =====
        existing = by_url.get(url)
        if existing is None:
            by_url[url] = listing
        elif listing.in_stock and not existing.in_stock:
            by_url[url] = listing

    return list(by_url.values())
