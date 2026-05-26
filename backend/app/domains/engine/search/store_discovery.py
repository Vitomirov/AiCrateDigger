"""Dynamic indie record-store discovery (Tavily + LLM → DB upsert).

When the curated whitelist has too few ``local_shop`` rows for a resolved city,
:func:`discover_new_stores` widens coverage on demand:

    1. Tavily probes (no ``include_domains``) — multiple shop-shaped queries
       broaden recall in cities where the English "best of" listicles dominate
       the SERP. Each request passes the structured ``country`` field so Tavily
       prioritises geographically-matching results.
    2. LLM (gpt-4o-mini, JSON-only, temperature 0) verifies each candidate as a
       *physical* indie record shop OR a small local vinyl mailorder, extracts
       canonical domain + display name.
    3. Upsert into ``whitelist_stores``: insert new rows with ``store_type='local_shop'``;
       for existing rows only *back-fill* nullable fields (``city``, ``store_type``,
       ``country_code``, …). ``priority`` is NEVER overwritten on existing rows.

:func:`discover_stores_from_snippets` exposes the same LLM verification + upsert
path against arbitrary externally-sourced snippets so the pipeline can also
run **opportunistic discovery** on the main consolidated Tavily call's results
(see :mod:`app.domains.search_pipeline.vinyl_search`).

No fallback to web crawling. Deterministic JSON contract. Logs every reject reason.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI
from sqlalchemy import case, func, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.config import get_settings
from app.core.db.database import WhitelistStoreORM, is_database_configured, session_factory
from app.domains.engine.policies.geo_scope import country_to_region
from app.domains.engine.policies.store_domain import canonical_store_domain, is_valid_store_host

from app.domains.engine.search import fetch_tavily_results_body
from app.domains.engine.search.country_boost import tavily_country_from_iso3166_alpha2

_TAVILY_TIMEOUT_S = 15.0
_TAVILY_MAX_RESULTS = 10
#: Conservative-but-not-strangling confidence floor for the LLM verifier.
#: Lowered from the historical 0.5 — gpt-4o-mini emits 0.4–0.5 for plausible
#: indie shops backed by listicle-only evidence, which is exactly what the probe
#: returns for poorly-covered cities like Hannover, Porto, smaller Balkans towns.
_MIN_CONFIDENCE: float = 0.4


logger = logging.getLogger(__name__)

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
        "leila.rs",
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


def _build_probe_queries(city: str, country_code: str) -> list[str]:
    """Return the diverse probe queries fired against Tavily.

    Two complementary queries: a "best of" listicle hook (great for landing on
    listicle pages whose snippets enumerate concrete shop domains) and a
    direct shop-shaped query (great for landing on shop home / contact pages).
    The country code is appended so Tavily prefers TLD-matching results even
    before the structured ``country`` parameter kicks in.
    """
    cc = (country_code or "").strip().upper()
    city_clean = city.strip()
    base = [
        f"best independent record stores {city_clean} {cc} vinyl shop",
        f"vinyl record shop {city_clean} {cc} buy LP",
    ]
    return [q.strip() for q in base if q.strip()]


async def _tavily_single_probe(
    *,
    client: httpx.AsyncClient,
    query: str,
    tavily_country: str | None,
) -> list[dict[str, str]]:
    """One Tavily HTTP call — returns the cleaned ``{title,url,content}`` list."""
    settings = get_settings()
    payload: dict[str, object] = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": "advanced",
        # Bug fix: Tavily's REST API parameter is ``max_results``, not
        # ``max_results_per_query``. The mis-named key was silently ignored,
        # so the probe was returning Tavily's default (5) instead of the
        # requested cap — starving the LLM verifier of evidence.
        "max_results": _TAVILY_MAX_RESULTS,
    }
    if tavily_country:
        payload["topic"] = "general"
        payload["country"] = tavily_country
    data = await fetch_tavily_results_body(client, payload, query_for_log=query)
    if data is None:
        logger.warning(
            "store_discovery_tavily_http_error",
            extra={
                "stage": "store_discovery",
                "reason": "tavily_request_failed_after_retries",
                "query": query[:200],
            },
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


async def _tavily_probe(city: str, country_code: str) -> list[dict[str, str]]:
    """Multi-query search-engine probe for indie record shops in the target city.

    Returns a deduplicated list of raw ``{title, url, content}`` dicts across
    every probe query. Empty list on any HTTP / parse error.
    """
    tavily_country = tavily_country_from_iso3166_alpha2(country_code)
    queries = _build_probe_queries(city, country_code)

    async with httpx.AsyncClient(timeout=_TAVILY_TIMEOUT_S) as client:
        probes = await asyncio.gather(
            *(
                _tavily_single_probe(
                    client=client,
                    query=q,
                    tavily_country=tavily_country,
                )
                for q in queries
            ),
            return_exceptions=True,
        )

    merged: dict[str, dict[str, str]] = {}
    for batch in probes:
        if isinstance(batch, BaseException):
            logger.warning(
                "store_discovery_probe_error",
                extra={"stage": "store_discovery", "reason": str(batch)[:200]},
            )
            continue
        for item in batch:
            url = item.get("url", "")
            if not url or url in merged:
                continue
            merged[url] = item
    return list(merged.values())


# ---------------------------------------------------------------------------
# LLM verification
# ---------------------------------------------------------------------------


_DISCOVERY_SYSTEM_PROMPT = """
You are AiCrateDigger's STORE DISCOVERY agent.

INPUT
You receive a JSON object with `target_city`, `target_country_code`, and a list
of search-engine `snippets` (each = title + url + content). Some snippets are
listicles / "best of" articles that enumerate concrete shop domains in their
body or link target. Others are shop home pages, contact pages, product pages,
or label-with-online-store pages. Your job is to extract every site that
plausibly sells PHYSICAL music (vinyl LP / CD / cassette) from a small,
locally-rooted operator in the target city or country — INCLUDING small
country-level mailorders that ship from there.

ACCEPT a candidate (emit a row) when ANY of the following is true:
  * The snippet (title or content) names a record shop and the URL is the
    shop's own domain (NOT a hub / listicle / encyclopedia / blog host).
  * A listicle-style snippet enumerates a shop name AND the URL it links to
    is that shop's own domain.
  * The URL's TLD aligns with the target country (e.g. ``.de`` for Germany,
    ``.pt`` for Portugal, ``.rs`` for Serbia) AND the page describes a
    record / vinyl / CD shop (look for tokens: vinyl, LP, record, "Schallplatte",
    "Plattenladen", "discos", "винил", "ploče", "vinili"…). Country-aligned
    small labels that operate their own online shop count as record shops.
  * The snippet explicitly says the shop is based in / from / in the target
    city or country (street address, "in {city}", "from {country}").

REJECT (do NOT emit) only the following:
  * Marketplaces / aggregators: Discogs, eBay, Amazon, CDandLP, Fnac,
    Bandcamp, Etsy, Allegro, Avito.
  * Encyclopedias / news / blogs / "best of" listicles whose own host is the
    listicle host (Wikipedia, Reddit, Pitchfork, Stereogum, TripAdvisor, Yelp,
    Time Out, RA, Resident Advisor, news portals). NOTE: if the listicle
    *mentions* a shop's own domain in the snippet body and that domain meets
    the ACCEPT criteria, emit the shop's domain — NOT the listicle host.
  * Coffee shops, bars, venues, festivals, instrument-only / hi-fi-only shops
    that do not sell records, DJ-equipment-only shops.
  * Mega global mailorders (HHV, JPC, Juno, Decks, Recordsale, the global
    Rough Trade chain) — unless the snippet explicitly identifies the page as
    a local branch in the target city.
  * Hosts whose domain looks like a placeholder ("none", "unknown",
    "example.com", single-label hosts without a real TLD, "domain.com").

OUTPUT — STRICT JSON ONLY (no prose):
{
  "stores": [
    {
      "name": "string (display name, max 120 chars)",
      "domain": "string (canonical hostname, lowercase, no scheme, no path, no www.)",
      "city": "string (canonical city name in English)",
      "country_code": "string (ISO-3166-1 alpha-2, uppercase)",
      "confidence": 0.0
    }
  ]
}

`confidence` (0.0–1.0):
  * 0.85+ — explicit address in target city + own-domain shop link.
  * 0.7  — established shop name + own-domain link, no explicit address.
  * 0.5–0.6 — country-aligned TLD + vinyl/record vocabulary on the page.
  * 0.4  — plausible local shop, only one of the ACCEPT signals.
  * <0.4 — too uncertain; the caller will reject the row.

Be GENEROUS. Local users in poorly-covered cities (Porto, Hannover, smaller
Balkans towns) see nothing if you emit zero stores. Downstream stages enforce
URL hygiene and reject placeholder hosts — your job is recall, not precision.
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
        # Reject ``"none"``/``"unknown"``/single-label noise that GPT sometimes
        # invents when no real domain appears in the snippets. Logging here makes
        # the discovery LLM's failure modes visible without breaking the run.
        if not is_valid_store_host(domain):
            logger.info(
                "store_discovery_invalid_domain_rejected",
                extra={
                    "stage": "store_discovery",
                    "raw_domain": str(r.get("domain") or "")[:64],
                    "normalized": domain[:64],
                },
            )
            continue
        name = str(r.get("name") or "").strip()
        if not name:
            continue
        try:
            conf = float(r.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < _MIN_CONFIDENCE:
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


def _normalize_city_country(
    city: str | None, country_code: str | None
) -> tuple[str, str]:
    """Trim + ISO-2 normalize the (city, country_code) inputs. ``UK`` → ``GB``."""
    city_clean = (city or "").strip()
    cc_clean = (country_code or "").strip().upper()
    if cc_clean == "UK":
        cc_clean = "GB"
    return city_clean, cc_clean


async def _verify_and_upsert_snippets(
    *,
    city: str,
    country_code: str,
    snippets: list[dict[str, str]],
    report: DiscoveryReport,
) -> None:
    """Run LLM verification on ``snippets`` and upsert verified rows into DB.

    Mutates ``report`` in place — the caller decides whether the inputs came
    from a dedicated probe (:func:`_tavily_probe`) or from an external Tavily
    call (e.g. the main consolidated search in
    :mod:`app.domains.search_pipeline.vinyl_search`).
    """
    candidates = await _llm_extract_candidates(
        city=city,
        country_code=country_code,
        snippets=snippets,
    )
    report.candidates = len(candidates)
    if not candidates:
        report.error = "llm_no_verified_stores"
        return

    inserted, updated = await _upsert_candidates(candidates)
    report.inserted = len(inserted)
    report.updated = len(updated)
    report.rejected = report.candidates - report.inserted - report.updated
    report.domains_inserted = inserted
    report.domains_updated = updated


async def discover_new_stores(city: str, country_code: str) -> DiscoveryReport:
    """End-to-end discovery: Tavily probe → LLM → DB upsert.

    No-op when the required env keys / DB session are missing.
    """
    report = DiscoveryReport()
    settings = get_settings()

    city_clean, cc_clean = _normalize_city_country(city, country_code)
    if not city_clean or not cc_clean or len(cc_clean) != 2:
        report.error = "missing_city_or_country_code"
        return report
    if not settings.tavily_api_key or not settings.openai_api_key:
        report.error = "missing_api_keys"
        return report
    if not is_database_configured():
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

    await _verify_and_upsert_snippets(
        city=city_clean,
        country_code=cc_clean,
        snippets=snippets,
        report=report,
    )

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


async def discover_stores_from_snippets(
    *,
    city: str | None,
    country_code: str | None,
    snippets: list[dict[str, str]],
) -> DiscoveryReport:
    """Run LLM verification + DB upsert against arbitrary external snippets.

    Companion to :func:`discover_new_stores`: instead of firing its own Tavily
    probe, this entry-point lets the pipeline reuse snippets from the main
    consolidated Tavily call. That captures cases where Tavily already surfaced
    a real local shop URL (``rockers.de``, ``van-records.com``, …) in the
    answer to the user's artist/album query, but the prefilter would otherwise
    drop it as an unknown host without a PDP-shaped URL.

    No-op when env keys / DB are missing, or the snippets list is empty.
    Returns a populated :class:`DiscoveryReport`; the caller is expected to
    merge :attr:`DiscoveryReport.domains_inserted` /
    :attr:`DiscoveryReport.domains_updated` into the in-memory prefilter
    whitelist so they take effect on the *current* request.
    """
    report = DiscoveryReport()
    settings = get_settings()

    city_clean, cc_clean = _normalize_city_country(city, country_code)
    if not city_clean or not cc_clean or len(cc_clean) != 2:
        report.error = "missing_city_or_country_code"
        return report
    if not settings.openai_api_key:
        report.error = "missing_api_keys"
        return report
    if not is_database_configured():
        report.error = "no_database_url"
        return report
    if not snippets:
        report.error = "no_snippets"
        return report

    # Bound the LLM payload so a 20-row main Tavily call doesn't push the
    # discovery call into the high-token tier. We already deduplicate by URL
    # downstream when merging insert/update domain lists.
    bounded = snippets[: 2 * _TAVILY_MAX_RESULTS]

    logger.info(
        "store_discovery_from_snippets_start",
        extra={
            "stage": "store_discovery",
            "city": city_clean,
            "country_code": cc_clean,
            "snippet_count": len(bounded),
        },
    )

    await _verify_and_upsert_snippets(
        city=city_clean,
        country_code=cc_clean,
        snippets=bounded,
        report=report,
    )

    logger.info(
        "store_discovery_from_snippets_done",
        extra={
            "stage": "store_discovery",
            "city": city_clean,
            "country_code": cc_clean,
            "inserted": report.inserted,
            "updated": report.updated,
            "rejected": report.rejected,
            "candidates": report.candidates,
            "error": report.error,
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
