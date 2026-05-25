"""Programmatic pre-filter for raw Tavily results.

Goal: shrink the Tavily SERP pool (size matches ``tavily_single_call_max_results``,
typically ~10 rows) to a high-signal slice (~7–10 candidates)
**before** they ever reach the LLM extractor, so we spend OpenAI tokens only on
URLs that have a realistic chance of being a buyable physical-music product
page on a multi-shop European pool.

Three layered cuts:

1. **Negative pattern blacklist** — discard known-noise hosts (video,
   encyclopedia, social, streaming, news portals). *No shop domains hardcoded*
   — only domain-substrings we know never serve PDP rows for vinyl/CD/cassette.
2. **Positive dynamic whitelist** — hosts present in the Postgres
   ``whitelist_stores`` table (curated + auto-discovered indies) always pass.
   This is what makes the system fully dynamic: as :mod:`app.services.store_discovery`
   adds new indie shops, the prefilter automatically trusts them on the next request.
3. **PDP-required gate for unknown hosts** — hosts that are NEITHER in the
   blacklist NOR in the whitelist must show a product-shaped URL path (e.g.
   ``/products/``, ``/p/``, ``/vinyl/``, ``-p-1234.html``) to survive. Pure
   editorial / landing / category URLs from unknown hosts are dropped before
   LLM tokens are spent.

Per-host dedupe: Tavily often returns multiple deep links from the same store.
We keep the top ``max_per_host`` highest-scored rows per host so the LLM batch
shows **variety across many shops**, not 7 hits from one marketplace.

Output is ordered by Tavily's relevance score and hard-capped at
``max_candidates`` (default ~10).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from app.services.tavily.scoring import product_signal_multiplier
from app.validators.listings import url_suggests_product_detail_page

logger = logging.getLogger(__name__)


#: Hosts (or substrings) we never want to send to the LLM. Pure noise filter —
#: zero shop domains are listed here. Match is a case-insensitive substring on
#: the registrable host so subdomains (e.g. ``music.youtube.com``) are caught.
_BLACKLIST_HOST_SUBSTRINGS: tuple[str, ...] = (
    # Video / streaming / lyrics
    "youtube.com",
    "youtu.be",
    "spotify.com",
    "music.apple.com",
    "soundcloud.com",
    "deezer.com",
    "tidal.com",
    "pandora.com",
    "bandcamp.com",  # discovery / fan platform, not a multi-artist PDP shop
    "genius.com",
    "azlyrics.com",
    "lyrics.com",
    "songkick.com",
    "setlist.fm",
    # Encyclopedia / metadata
    "wikipedia.org",
    "wikidata.org",
    "musicbrainz.org",
    "allmusic.com",
    "rateyourmusic.com",
    "last.fm",
    "lastfm.",
    "imdb.com",
    "fandom.com",
    "rollingstone.com",
    "pitchfork.com",
    "stereogum.com",
    "factmag.com",
    # Social
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "reddit.com",
    "pinterest.com",
    "tumblr.com",
    "threads.net",
    "linkedin.com",
    "vk.com",
    # News / aggregators / search engines
    "news.google.",
    "bing.com",
    "duckduckgo.com",
    "yahoo.com",
    "msn.com",
    "tripadvisor.",
    "yelp.",
    "timeout.com",
    # Discogs metadata (we hit the API server-side; the public site is a hub
    # without consistent buy-now PDP rows for our shop-pool intent).
    "discogs.com",
    "vinylhub.com",
    # Document hosts / archives that occasionally show up in SERPs
    "archive.org",
    "scribd.com",
    "medium.com",
    "boilerroom.tv",
    "ra.co",
)


#: News-portal / general media substrings — substring match anywhere in the
#: registrable host. Catches large European news outlets that publish "best
#: albums of 2026" listicles and surface in Tavily SERPs without ever selling
#: vinyl. *Negative* patterns only — no shop domains here.
_NEWS_HOST_SUBSTRINGS: tuple[str, ...] = (
    # Generic news / info host words (multi-language)
    "blic.",
    "n1info.",
    "telegraf.",
    "kurir.",
    "novosti.",
    "espreso.",
    "danas.rs",
    "b92.",
    "rts.rs",
    "rtl.",
    "vesti.",
    "24sata.",
    "jutarnji.",
    "vecernji.",
    "index.hr",
    "hrt.hr",
    "dnevnik.",
    "tportal.",
    "spiegel.",
    "welt.",
    "faz.net",
    "sueddeutsche.",
    "bild.",
    "tagesschau.",
    "tagesspiegel.",
    "zeit.de",
    "stern.de",
    "focus.de",
    "lemonde.",
    "lefigaro.",
    "liberation.",
    "leparisien.",
    "lepoint.",
    "lexpress.",
    "ouest-france.",
    "bbc.co",
    "bbc.com",
    "theguardian.",
    "telegraph.co",
    "independent.co",
    "thetimes.",
    "dailymail.",
    "express.co",
    "mirror.co",
    "cnn.com",
    "nytimes.",
    "washingtonpost.",
    "reuters.",
    "apnews.",
    "aljazeera.",
    "euronews.",
    "rt.com",
    "tass.",
    "ria.ru",
    "corriere.",
    "repubblica.",
    "lastampa.",
    "gazzetta.",
    "elpais.",
    "elmundo.",
    "abc.es",
    "marca.com",
    "ansa.it",
    "publico.",
    "expresso.",
    "rtp.pt",
    "wp.pl",
    "onet.pl",
    "interia.pl",
    "tvn24.pl",
    "polskieradio.",
    "nu.nl",
    "telegraaf.nl",
    "ad.nl",
    "nrc.nl",
    "rte.ie",
    "ert.gr",
    "kathimerini.",
    "tovima.",
    "iefimerida.",
    "sport-klub.",
    "sportklub.",
    "hurriyet.",
    "milliyet.",
)


#: Path fragments that are almost always editorial / non-PDP even on legit
#: shop hosts (e.g. a record store's blog). Used to demote in scoring AND to
#: hard-reject *unknown* hosts that show no PDP signal.
_EDITORIAL_PATH_SUBSTRINGS: tuple[str, ...] = (
    "/blog",
    "/news",
    "/article",
    "/articles",
    "/clanak/",
    "/clanci/",
    "/vesti/",
    "/magazine",
    "/feature",
    "/features",
    "/story",
    "/stories",
    "/playlist",
    "/playlists",
    "/podcast",
    "/podcasts",
    "/tag/",
    "/tags/",
    "/forum/",
    "/community/",
)


_WWW_PREFIX_RE = re.compile(r"^www\.", re.IGNORECASE)


def _registrable_host(url: str) -> str | None:
    """Lowercase host with leading ``www.`` stripped, or ``None`` if unparseable."""
    if not url:
        return None
    try:
        netloc = urlparse(url.strip()).netloc.lower()
    except Exception:
        return None
    if not netloc:
        return None
    return _WWW_PREFIX_RE.sub("", netloc).split(":", 1)[0]


def _is_blacklisted(host: str) -> bool:
    """``True`` when ``host`` matches any blacklist or news-portal substring."""
    if not host:
        return True
    h = host.lower()
    if any(token in h for token in _BLACKLIST_HOST_SUBSTRINGS):
        return True
    if any(token in h for token in _NEWS_HOST_SUBSTRINGS):
        return True
    return False


def _path_looks_editorial(url: str) -> bool:
    try:
        path = (urlparse(url).path or "").lower()
    except Exception:
        return False
    return any(token in path for token in _EDITORIAL_PATH_SUBSTRINGS)


def _looks_like_product_url(url: str) -> bool:
    """PDP-shaped URL heuristic for *unknown* hosts (not in the DB whitelist).

    Two independent signals — either is sufficient:
    * :func:`url_suggests_product_detail_page` already used by the validator,
      catches ``/product/``, ``/products/``, ``/p/``, ``/item/``, ``-p-1234``,
      slug-id suffixes, etc.
    * :func:`product_signal_multiplier` ≥ 1.0 → URL path hits a retail keyword.
    """
    if url_suggests_product_detail_page(url):
        return True
    try:
        path = (urlparse(url).path or "/")
    except Exception:
        return False
    return product_signal_multiplier(path) >= 1.0


def _host_in_whitelist(host: str, whitelist: frozenset[str] | None) -> bool:
    """``host`` is a known store from ``whitelist_stores`` (subdomain-safe)."""
    if not host or not whitelist:
        return False
    h = host.lower()
    if h.startswith("www."):
        h = h[4:]
    if h in whitelist:
        return True
    return any(h.endswith("." + d) for d in whitelist)


def _result_score(row: dict[str, Any], *, is_known_shop: bool) -> float:
    """Tavily ``score`` boosted by the PDP-vs-editorial path signal.

    Whitelisted shop hosts get a flat boost so a slightly-lower Tavily score
    on a known indie doesn't lose to a high-score editorial URL on an unknown
    host.
    """
    base = 0.0
    try:
        base = float(row.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        base = 0.0
    url = str(row.get("url") or "")
    try:
        path = (urlparse(url).path or "/")
    except Exception:
        path = "/"
    mult = product_signal_multiplier(path)
    if _path_looks_editorial(url):
        mult = min(mult, 0.35)
    score = base * max(mult, 0.05)
    if is_known_shop:
        score = max(score * 1.25, score + 0.10)
    return score


def prefilter_tavily_results(
    raw_results: list[dict[str, Any]],
    *,
    max_candidates: int = 10,
    max_per_host: int = 2,
    known_shop_hosts: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sanitize ``raw_results`` into an LLM-ready candidate list.

    Args:
        raw_results: Tavily result dicts (``url`` / ``title`` / ``content`` / ``score``).
        max_candidates: hard cap on returned candidate count.
        max_per_host: keep at most this many deep links per host (variety).
        known_shop_hosts: hostnames present in the ``whitelist_stores`` table.
            Whitelist hosts always pass the noise gate and get a score boost;
            non-whitelisted hosts must show a PDP-shaped URL to survive.

    Returns ``(kept_candidates, diagnostic_dict)``.
    """
    whitelist: frozenset[str] = frozenset(
        h.strip().lower().removeprefix("www.")
        for h in (known_shop_hosts or [])
        if (h or "").strip()
    )

    diagnostic: dict[str, Any] = {
        "raw_count": len(raw_results),
        "whitelist_size": len(whitelist),
        "missing_url": 0,
        "blacklisted_hosts": 0,
        "rejected_no_pdp_signal": 0,
        "per_host_capped": 0,
        "kept_count": 0,
        "kept_unique_hosts": 0,
        "kept_known_shop": 0,
        "kept_top_hosts": [],
    }

    if not raw_results:
        return [], diagnostic

    blacklist_hosts_seen: set[str] = set()
    rejected_unknown_hosts: set[str] = set()
    annotated: list[tuple[float, str, bool, dict[str, Any]]] = []

    for row in raw_results:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "").strip()
        if not url:
            diagnostic["missing_url"] += 1
            continue
        host = _registrable_host(url)
        if host is None:
            diagnostic["missing_url"] += 1
            continue
        if _is_blacklisted(host):
            diagnostic["blacklisted_hosts"] += 1
            blacklist_hosts_seen.add(host)
            continue

        is_known_shop = _host_in_whitelist(host, whitelist)
        if not is_known_shop and not _looks_like_product_url(url):
            diagnostic["rejected_no_pdp_signal"] += 1
            rejected_unknown_hosts.add(host)
            continue

        score = _result_score(row, is_known_shop=is_known_shop)
        annotated.append((score, host, is_known_shop, row))

    annotated.sort(key=lambda x: x[0], reverse=True)

    per_host_counts: dict[str, int] = {}
    capped: list[dict[str, Any]] = []
    for score, host, is_known_shop, row in annotated:
        if len(capped) >= max_candidates:
            break
        cnt = per_host_counts.get(host, 0)
        if cnt >= max_per_host:
            diagnostic["per_host_capped"] += 1
            continue
        per_host_counts[host] = cnt + 1
        candidate = {
            "url": str(row.get("url") or "").strip(),
            "title": str(row.get("title") or "").strip(),
            "content": str(row.get("content") or "").strip(),
            "score": float(score),
            "host": host,
            "is_known_shop": is_known_shop,
        }
        capped.append(candidate)

    diagnostic["kept_count"] = len(capped)
    diagnostic["kept_unique_hosts"] = len({c["host"] for c in capped})
    diagnostic["kept_known_shop"] = sum(1 for c in capped if c["is_known_shop"])
    diagnostic["kept_top_hosts"] = [c["host"] for c in capped][:10]
    if blacklist_hosts_seen:
        diagnostic["blacklisted_sample"] = sorted(blacklist_hosts_seen)[:10]
    if rejected_unknown_hosts:
        diagnostic["rejected_no_pdp_sample"] = sorted(rejected_unknown_hosts)[:10]

    logger.info(
        "tavily_prefilter",
        extra={"stage": "tavily_prefilter", **diagnostic},
    )

    return capped, diagnostic


__all__ = ["prefilter_tavily_results"]
