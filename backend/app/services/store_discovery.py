"""Dynamic indie record-store discovery (Tavily + LLM → DB upsert).

When the curated whitelist has too few ``local_shop`` rows for a resolved city,
:func:`discover_new_stores` widens coverage on demand:

    1. Tavily probe (no include_domains) for "best independent record stores in {city}".
    2. LLM (gpt-4o-mini, JSON-only, temperature 0) verifies each candidate as a
       *physical* indie record shop, extracts canonical domain + display name.
    3. Upsert into ``whitelist_stores``: insert new rows with ``store_type='local_shop'``;
       for existing rows only *back-fill* nullable fields (``city``, ``store_type``,
       ``country_code``, …). ``priority`` is NEVER overwritten on existing rows.

No fallback to web crawling. Deterministic JSON contract. Logs every reject reason.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI
from sqlalchemy import case, func, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import get_settings
from app.db.database import WhitelistStoreORM, session_factory
from app.policies.geo_scope import country_to_region
from app.policies.store_domain import canonical_store_domain

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_TAVILY_TIMEOUT_S = 15.0
_TAVILY_MAX_RESULTS = 10

#: Hosts that are never indie record shops — skipped before the LLM call.
_DOMAIN_BLACKLIST: frozenset[str] = frozenset(
    {
        "amazon.com",
        "ebay.com",
        "discogs.com",
        "wikipedia.org",
        "reddit.com",
        "tripadvisor.com",
        "tripadvisor.co.uk",
        "yelp.com",
        "yelp.co.uk",
        "facebook.com",
        "instagram.com",
        "twitter.com",
        "x.com",
        "youtube.com",
        "google.com",
        "maps.google.com",
        "spotify.com",
        "soundcloud.com",
        "bandcamp.com",
        "ra.co",
        "timeout.com",
        "vinylhub.com",
        "boilerroom.tv",
        "medium.com",
        "tumblr.com",
        "pitchfork.com",
        "factmag.com",
        "stereogum.com",
    }
)

#: Default merchant trust for a freshly discovered indie. Curated rows from
#: ``ALLOWED_STORES`` stay at 8–10; discovery rows must not jump above them.
_DISCOVERED_PRIORITY: int = 7
_DISCOVERED_LISTING_QUALITY: int = 6


@dataclass(frozen=True, slots=True)
class DiscoveredStoreCandidate:
    """LLM-verified candidate before DB upsert."""

    name: str
    domain: str
    city: str
    country_code: str
    confidence: float


@dataclass(slots=True)
class DiscoveryReport:
    inserted: int = 0
    updated: int = 0
    rejected: int = 0
    candidates: int = 0
    domains_inserted: list[str] | None = None
    domains_updated: list[str] | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "inserted": self.inserted,
            "updated": self.updated,
            "rejected": self.rejected,
            "candidates": self.candidates,
            "domains_inserted": list(self.domains_inserted or ()),
            "domains_updated": list(self.domains_updated or ()),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Tavily probe
# ---------------------------------------------------------------------------


async def _tavily_probe(city: str, country_code: str) -> list[dict[str, str]]:
    """Search-engine probe for indie record shops in the target city.

    Returns a list of raw ``{title, url, content}`` dicts (Tavily payload subset).
    Empty list on any HTTP / parse error.
    """
    settings = get_settings()
    query = (
        f'best independent record stores in {city}, {country_code} vinyl shop'
    )
    payload: dict[str, object] = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results_per_query": _TAVILY_MAX_RESULTS,
    }
    async with httpx.AsyncClient(timeout=_TAVILY_TIMEOUT_S) as client:
        try:
            resp = await client.post(TAVILY_SEARCH_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning(
                "store_discovery_tavily_http_error",
                extra={"stage": "store_discovery", "reason": str(exc)},
            )
            return []
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "store_discovery_tavily_unexpected",
                extra={"stage": "store_discovery", "reason": str(exc)},
            )
            return []

    raw_items = data.get("results", []) or []
    cleaned: list[dict[str, str]] = []
    for item in raw_items:
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        cleaned.append(
            {
                "title": str(item.get("title", "")).strip()[:240],
                "url": url,
                "content": str(item.get("content", "")).strip()[:1500],
            }
        )
    return cleaned


# ---------------------------------------------------------------------------
# LLM verification
# ---------------------------------------------------------------------------


_DISCOVERY_SYSTEM_PROMPT = """
You are AiCrateDigger's STORE DISCOVERY agent.

Your input is a list of search-engine snippets about independent record shops
in a specific city. Your job is to extract ONLY genuine, physical (brick-and-mortar)
indie record shops with their own e-commerce or info domain.

REJECT (do NOT emit) any of the following:
- Mailorder-only / online-only operators (HHV, JPC, Juno, Decks, Recordsale, Rough Trade global chain unless the page is the local store).
- Marketplaces / aggregators (Discogs, eBay, Amazon, CDandLP, Fnac, Bandcamp).
- Blogs, "best of" lists, news articles, magazines, TripAdvisor, Yelp, Reddit, Wikipedia, YouTube, social media.
- Coffee shops, bars, venues, festivals, labels-only (no shop), instrument shops, hi-fi gear shops.
- DJ-only equipment shops with no records catalogue.

ACCEPT only when at least one snippet provides strong evidence:
- a clear shop name (proper noun, often present in title),
- a verified own-domain link (shop website, not a hub),
- the shop is physically located in the requested city (street address or "based in {city}" wording).

OUTPUT — STRICT JSON ONLY (no prose):
{
  "stores": [
    {
      "name": "string (display name, max 120 chars)",
      "domain": "string (canonical hostname, no scheme, no path, no www.)",
      "city": "string (canonical city name)",
      "country_code": "string (ISO-3166-1 alpha-2)",
      "confidence": 0.0
    }
  ]
}

`confidence` is in [0.0, 1.0] — your subjective certainty the row is a genuine
physical indie record shop. Use 0.6 baseline; bump up for explicit addresses,
established names; bump down for ambiguity. Rows below 0.5 will be rejected.
""".strip()


async def _llm_extract_candidates(
    *,
    city: str,
    country_code: str,
    snippets: list[dict[str, str]],
) -> list[DiscoveredStoreCandidate]:
    if not snippets:
        return []

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    payload = {
        "target_city": city,
        "target_country_code": country_code,
        "snippets": snippets,
    }
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _DISCOVERY_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "store_discovery_llm_failed",
            extra={"stage": "store_discovery", "reason": str(exc)},
        )
        return []

    raw = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(
            "store_discovery_llm_json_decode_error",
            extra={"stage": "store_discovery", "reason": str(exc), "raw_head": raw[:300]},
        )
        return []

    rows = data.get("stores", []) or []
    out: list[DiscoveredStoreCandidate] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        domain = canonical_store_domain(r.get("domain"))
        if not domain or domain in _DOMAIN_BLACKLIST:
            continue
        name = str(r.get("name") or "").strip()
        if not name:
            continue
        try:
            conf = float(r.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < 0.5:
            continue
        city_out = (str(r.get("city") or "").strip()) or city
        cc_raw = (str(r.get("country_code") or "").strip().upper()) or country_code.upper()
        if cc_raw == "UK":
            cc_raw = "GB"
        if len(cc_raw) != 2 or not cc_raw.isalpha():
            cc_raw = country_code.upper()
        out.append(
            DiscoveredStoreCandidate(
                name=name[:120],
                domain=domain,
                city=city_out[:128],
                country_code=cc_raw,
                confidence=max(0.0, min(1.0, conf)),
            )
        )
    return out


# ---------------------------------------------------------------------------
# DB upsert
# ---------------------------------------------------------------------------


def _dedupe_candidates_by_domain(
    candidates: list[DiscoveredStoreCandidate],
) -> list[DiscoveredStoreCandidate]:
    """Keep one row per canonical ``domain`` (highest ``confidence`` wins).

    Prevents duplicate rows in ``values_list`` that would confuse PostgreSQL
    bulk INSERT or trigger redundant conflict handling.
    """
    best_by_domain: dict[str, DiscoveredStoreCandidate] = {}
    for cand in candidates:
        prev = best_by_domain.get(cand.domain)
        if prev is None or cand.confidence > prev.confidence:
            best_by_domain[cand.domain] = cand
    return list(best_by_domain.values())


async def save_discovered_stores(
    candidates: list[DiscoveredStoreCandidate],
) -> tuple[list[str], list[str]]:
    """Bulk PostgreSQL UPSERT for discovered stores.

    Uses ``INSERT ... ON CONFLICT (domain) DO UPDATE`` so duplicate domains from
    Tavily/LLM never raise ``UniqueViolationError``. Conflicting rows refresh
    empty/null ``city``, ``country_code``, ``region``, ``store_type`` (when not
    yet set to a known type), and ``is_active``. Never updates ``priority``,
    ``name``, ``ships_to_json``, ``latitude``, or ``longitude``.

    Returns ``(inserted_domains, updated_domains)`` relative to rows that existed
    before this call (domains present only in this batch).
    """
    if not candidates:
        return [], []

    unique = _dedupe_candidates_by_domain(candidates)
    domains_in_batch = [c.domain for c in unique]

    try:
        sf = session_factory()
    except RuntimeError:
        logger.warning(
            "store_discovery_no_db",
            extra={"stage": "store_discovery", "reason": "session_factory_unavailable"},
        )
        return [], []

    values_list: list[dict[str, object]] = []
    for cand in unique:
        reg = country_to_region(cand.country_code)
        leg = cand.country_code[:8] if cand.country_code else "XX"
        values_list.append(
            {
                "name": cand.name,
                "domain": cand.domain,
                "country": leg,
                "country_code": cand.country_code,
                "region": reg,
                "ships_to_json": json.dumps(["EU"]),
                "priority": _DISCOVERED_PRIORITY,
                "is_active": True,
                "city": cand.city,
                "latitude": None,
                "longitude": None,
                "store_type": "local_shop",
            }
        )

    async with sf() as session:
        existing_rows = (
            await session.scalars(
                select(WhitelistStoreORM.domain).where(
                    WhitelistStoreORM.domain.in_(domains_in_batch)
                )
            )
        ).all()
        existing_before = set(existing_rows)

        stmt = pg_insert(WhitelistStoreORM).values(values_list)
        stmt = stmt.on_conflict_do_update(
            index_elements=["domain"],
            set_={
                "city": func.coalesce(
                    func.nullif(func.trim(WhitelistStoreORM.city), ""),
                    stmt.excluded.city,
                ),
                "country_code": func.coalesce(
                    func.nullif(func.trim(WhitelistStoreORM.country_code), ""),
                    stmt.excluded.country_code,
                ),
                "region": func.coalesce(
                    func.nullif(func.trim(WhitelistStoreORM.region), ""),
                    stmt.excluded.region,
                ),
                "store_type": case(
                    (
                        WhitelistStoreORM.store_type.in_(
                            ("local_shop", "regional_ecommerce", "marketplace")
                        ),
                        WhitelistStoreORM.store_type,
                    ),
                    else_=stmt.excluded.store_type,
                ),
                "is_active": or_(WhitelistStoreORM.is_active, stmt.excluded.is_active),
            },
        )
        await session.execute(stmt)
        await session.commit()

    inserted = [d for d in domains_in_batch if d not in existing_before]
    updated = [d for d in domains_in_batch if d in existing_before]
    return inserted, updated


async def _upsert_candidates(
    candidates: list[DiscoveredStoreCandidate],
) -> tuple[list[str], list[str]]:
    """Backward-compatible alias for :func:`save_discovered_stores`."""
    return await save_discovered_stores(candidates)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def discover_new_stores(city: str, country_code: str) -> DiscoveryReport:
    """End-to-end discovery: Tavily → LLM → DB upsert.

    No-op when the required env keys / DB session are missing.
    """
    report = DiscoveryReport()
    settings = get_settings()

    city_clean = (city or "").strip()
    cc_clean = (country_code or "").strip().upper()
    if cc_clean == "UK":
        cc_clean = "GB"
    if not city_clean or not cc_clean or len(cc_clean) != 2:
        report.error = "missing_city_or_country_code"
        return report
    if not settings.tavily_api_key or not settings.openai_api_key:
        report.error = "missing_api_keys"
        return report
    if not settings.database_url:
        report.error = "no_database_url"
        return report

    logger.info(
        "store_discovery_start",
        extra={"stage": "store_discovery", "city": city_clean, "country_code": cc_clean},
    )

    snippets = await _tavily_probe(city_clean, cc_clean)
    if not snippets:
        report.error = "tavily_no_results"
        return report

    candidates = await _llm_extract_candidates(
        city=city_clean,
        country_code=cc_clean,
        snippets=snippets,
    )
    report.candidates = len(candidates)
    if not candidates:
        report.error = "llm_no_verified_stores"
        return report

    inserted, updated = await _upsert_candidates(candidates)
    report.inserted = len(inserted)
    report.updated = len(updated)
    report.rejected = report.candidates - report.inserted - report.updated
    report.domains_inserted = inserted
    report.domains_updated = updated

    logger.info(
        "store_discovery_done",
        extra={
            "stage": "store_discovery",
            "city": city_clean,
            "country_code": cc_clean,
            "inserted": report.inserted,
            "updated": report.updated,
            "rejected": report.rejected,
            "candidates": report.candidates,
        },
    )
    return report


# ---------------------------------------------------------------------------
# DB inspection helper (used by ensure_local_coverage)
# ---------------------------------------------------------------------------


async def count_local_shops_in_city(city: str, country_code: str) -> int:
    """Count active ``local_shop`` rows whose ``city`` (case-insensitive) matches."""
    try:
        sf = session_factory()
    except RuntimeError:
        return 0
    c = (city or "").strip()
    cc = (country_code or "").strip().upper()
    if cc == "UK":
        cc = "GB"
    if not c or not cc:
        return 0
    async with sf() as session:
        n = await session.scalar(
            select(func.count())
            .select_from(WhitelistStoreORM)
            .where(WhitelistStoreORM.is_active.is_(True))
            .where(WhitelistStoreORM.store_type == "local_shop")
            .where(WhitelistStoreORM.country_code == cc)
            .where(func.lower(WhitelistStoreORM.city) == c.lower())
        )
    return int(n or 0)
