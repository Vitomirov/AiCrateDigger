"""Consolidated \"power queries\" for low-credit Tavily runs.

Bundles artist + album + physical-format hints into one quoted intent string,
optionally augmented with ``(site:d1 OR site:d2 ...)`` groups. Chunking honours
approximate Tavily/query-engine length limits without injecting city/country
tokens (geo is enforced via whitelist ``include_domains`` + optional Tavily
``country`` fallback).
"""

from __future__ import annotations


PHYSICAL_FORMAT_GROUP = "(vinyl OR LP)"


def escape_for_quoted_term(value: str) -> str:
    """Escape backslashes and double quotes inside a phrase wrapped in outer ``\"``. """
    trimmed = str(value or "").strip()
    if not trimmed:
        return ""
    return trimmed.replace("\\", "\\\\").replace('"', '\\"')


def quoted_search_term(value: str) -> str:
    inner = escape_for_quoted_term(value)
    return f'"{inner}"' if inner else ""


def build_physical_power_query_base(*, artist: str | None, album_title: str) -> str:
    """Quoted artist + album spine + mandatory physical hints (no location tokens).

    Omitting blank segments keeps short queries workable for cramped indexes.
    """
    qp_art = quoted_search_term(str(artist or "").strip())
    qp_alb = quoted_search_term(str(album_title or "").strip())

    spine: list[str] = []
    if qp_art:
        spine.append(qp_art)
    if qp_alb:
        spine.append(qp_alb)
    head = " ".join(spine).strip()
    if not head:
        return PHYSICAL_FORMAT_GROUP.strip()
    return f"{head} {PHYSICAL_FORMAT_GROUP}".strip()


def format_site_operator_group(domains: list[str]) -> str:
    """``(site:a.com OR site:b.com ...)`` from canonical hostnames."""
    parts = [d.strip().lower().removeprefix("www.") for d in domains if str(d).strip()]
    inner = " OR ".join(f"site:{host}" for host in parts if host)
    return f"({inner})" if inner else ""


def build_power_query_with_sites(base_query: str, domain_chunk: list[str]) -> str:
    grp = format_site_operator_group(domain_chunk)
    base = base_query.strip()
    if not grp:
        return base
    return f"{base} {grp}".strip()


def chunk_domains_for_power_queries(
    base_query: str,
    domains: list[str],
    *,
    max_chars: int,
    max_domains_per_chunk: int,
) -> list[tuple[list[str], str]]:
    """Produce ``(domains_for_include, full_query_text)`` rows.

    * Greedily packs domains until the ``site:`` group would exceed ``max_chars``
      or ``max_domains_per_chunk``.
    * If even a lone ``site:`` suffix overflows (very long quoted titles),
      emits ``(host,)`` paired with ``base_query`` only so callers can rely on
      ``include_domains`` for host restriction instead of query text length.
    """
    if not domains:
        return []

    upper_max = max(1, min(int(max_domains_per_chunk), 32))
    cap = max(120, int(max_chars))
    cleaned = [
        str(d).strip().lower().removeprefix("www.") for d in domains if str(d).strip()
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for host in cleaned:
        if host in seen:
            continue
        seen.add(host)
        ordered.append(host)

    out: list[tuple[list[str], str]] = []
    i = 0
    base = base_query.strip()
    while i < len(ordered):
        chunk: list[str] = []
        while i < len(ordered) and len(chunk) < upper_max:
            trial = [*chunk, ordered[i]]
            candidate = build_power_query_with_sites(base, trial)
            if len(candidate) <= cap:
                chunk = trial
                i += 1
                continue
            break
        if chunk:
            out.append((chunk, build_power_query_with_sites(base, chunk)))
            continue
        lone = ordered[i]
        i += 1
        overflow_base = base
        one_site = build_power_query_with_sites(base, [lone])
        if len(one_site) <= cap:
            out.append(([lone], one_site))
        else:
            out.append(([lone], overflow_base))
    return out
