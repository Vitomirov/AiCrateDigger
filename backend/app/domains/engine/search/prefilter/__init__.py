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
   This is what makes the system fully dynamic: as :mod:`app.domains.engine.search.store_discovery`
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

Implementation modules: ``constants``, ``hosts``, ``signals``, ``filter``.
"""

from app.domains.engine.search.prefilter.filter import prefilter_tavily_results
from app.domains.engine.search.prefilter.hosts import (
    host_in_whitelist,
    is_blacklisted,
    registrable_host,
)

__all__ = [
    "host_in_whitelist",
    "is_blacklisted",
    "prefilter_tavily_results",
    "registrable_host",
]
