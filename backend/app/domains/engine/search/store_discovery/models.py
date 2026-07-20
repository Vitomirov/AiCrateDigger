"""Discovery datatypes and module-level constants."""

from __future__ import annotations

from dataclasses import dataclass

from app.domains.engine.search.prefilter.constants import (
    DIGITAL_MUSIC_HOST_SUBSTRINGS,
    EVENT_TICKETING_HOST_SUBSTRINGS,
    MERCH_HOST_SUBSTRINGS,
)

TAVILY_TIMEOUT_S = 15.0
TAVILY_MAX_RESULTS = 10
#: Conservative-but-not-strangling confidence floor for the LLM verifier.
#: Lowered from the historical 0.5 — gpt-4o-mini emits 0.4–0.5 for plausible
#: indie shops backed by listicle-only evidence, which is exactly what the probe
#: returns for poorly-covered cities like Hannover, Porto, smaller Balkans towns.
MIN_CONFIDENCE: float = 0.4

#: Hosts that are never indie record shops — skipped before the LLM call.
#: Unioned with the merch/digital-download/ticketing substrings from the
#: prefilter constants so a print-on-demand or digital-only platform can
#: never be auto-verified into ``whitelist_stores`` as a "local shop" in
#: the first place.
DOMAIN_BLACKLIST: frozenset[str] = frozenset(
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
    | set(MERCH_HOST_SUBSTRINGS)
    | set(DIGITAL_MUSIC_HOST_SUBSTRINGS)
    | set(EVENT_TICKETING_HOST_SUBSTRINGS)
)

#: Default merchant trust for a freshly discovered indie. Curated rows from
#: ``ALLOWED_STORES`` stay at 8–10; discovery rows must not jump above them.
DISCOVERED_PRIORITY: int = 7
DISCOVERED_LISTING_QUALITY: int = 6


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
