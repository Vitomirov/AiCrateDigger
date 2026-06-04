"""LLM verification of search snippets as indie record-shop candidates."""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.quota import QuotaExceededError, QuotaUnavailableError, openai_extract_quota_scope
from app.domains.engine.policies.store_domain import canonical_store_domain, is_valid_store_host
from app.domains.engine.search.store_discovery.models import (
    DOMAIN_BLACKLIST,
    MIN_CONFIDENCE,
    DiscoveredStoreCandidate,
)

logger = logging.getLogger(__name__)

DISCOVERY_SYSTEM_PROMPT = """
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


async def llm_extract_candidates(
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
        async with openai_extract_quota_scope():
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
    except (QuotaExceededError, QuotaUnavailableError):
        raise
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
        if not domain or domain in DOMAIN_BLACKLIST:
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
        if conf < MIN_CONFIDENCE:
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
