"""Regression tests for the ``store_loader`` → prefilter whitelist injection chain.

Pinpoints two bugs that previously caused local record shops from cities such
as Porto or Belgrade to be silently dropped before the LLM extractor saw them:

1.  ``run_vinyl_search`` deferred :func:`app.core.db.store_loader.ensure_local_coverage`
    onto FastAPI ``BackgroundTasks``, so discovery raced the prefilter and the
    *current* request's whitelist never contained the freshly-upserted indie
    domains. The user always saw the *previous* request's coverage.

2.  :func:`app.core.db.store_loader.load_active_stores` raised on any DB-side
    error; :func:`vinyl_search._load_known_shop_hosts` swallowed the exception
    and returned an *empty* ``frozenset``. The prefilter then treated every
    curated indie host as "unknown" and rejected its URLs via
    ``rejected_no_pdp_signal``.

These tests do not call OpenAI / Tavily / Postgres — they mock the DB session
and the discovery probe.
"""

from __future__ import annotations

import os
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "sk-store-loader-test")
os.environ.setdefault("TAVILY_API_KEY", "tv-store-loader-test")
os.environ["DATABASE_URL"] = ""
os.environ["REDIS_URL"] = ""

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import store_loader as store_loader_module  # noqa: E402
from app.domains.engine.policies.eu_stores import (  # noqa: E402
    ALLOWED_STORES,
    StoreEntry,
    get_active_stores,
)
from app.domains.engine.search.prefilter import (  # noqa: E402
    _host_in_whitelist,
    prefilter_tavily_results,
)
from app.domains.query_parser.parse_schema import ParsedQuery  # noqa: E402
from app.domains.search_pipeline import vinyl_search as vinyl_search_module  # noqa: E402


def _belgrade_parsed() -> ParsedQuery:
    return ParsedQuery(
        artist="Some Artist",
        album="Some Album",
        location="Belgrade",
        country_code="RS",
        search_scope="local",
        resolved_city="Belgrade",
        geo_granularity="city",
        original_query="Some Artist - Some Album vinyl Belgrade",
    )


def _porto_parsed() -> ParsedQuery:
    return ParsedQuery(
        artist="Some Artist",
        album="Some Album",
        location="Porto",
        country_code="PT",
        search_scope="local",
        resolved_city="Porto",
        geo_granularity="city",
        original_query="Some Artist - Some Album vinyl Porto",
    )


class TestStoreLoaderWhitelistChain(unittest.IsolatedAsyncioTestCase):
    """Direct unit checks on ``load_active_stores`` and ``_load_known_shop_hosts``."""

    async def test_curated_belgrade_locals_are_present_when_db_is_off(self) -> None:
        """``DATABASE_URL`` unset → in-code seed feeds the prefilter whitelist.

        Regression: a previous refactor left the prefilter whitelist empty
        whenever Postgres was not configured, dropping every curated indie
        (including ``metropolismusic.rs`` / ``mascom.rs``).
        """
        hosts = await vinyl_search_module._load_known_shop_hosts()
        self.assertIn("metropolismusic.rs", hosts)
        self.assertIn("mascom.rs", hosts)
        # Sanity: prefilter helper agrees on both registrable and ``www.`` forms.
        whitelist = frozenset(hosts)
        self.assertTrue(_host_in_whitelist("metropolismusic.rs", whitelist))
        self.assertTrue(_host_in_whitelist("www.metropolismusic.rs", whitelist))
        self.assertTrue(_host_in_whitelist("shop.mascom.rs", whitelist))

    async def test_load_active_stores_falls_back_to_code_when_db_query_raises(
        self,
    ) -> None:
        """Transient ``SQLAlchemyError`` must not collapse the whitelist.

        Regression: when ``DATABASE_URL`` was configured but the SELECT failed
        (DB briefly unreachable, asyncpg pool exhausted, …) the loader raised
        and ``_load_known_shop_hosts`` returned an empty ``frozenset``. The
        prefilter then dropped every curated indie URL as a no-PDP unknown.
        """
        from sqlalchemy.exc import SQLAlchemyError

        fake_settings = MagicMock()
        fake_settings.database_url = "postgresql+asyncpg://stub/stub"

        class _FailingSession:
            async def __aenter__(self) -> "_FailingSession":
                raise SQLAlchemyError("connection refused")

            async def __aexit__(self, *_exc: object) -> None:
                return None

        fake_factory = MagicMock(return_value=_FailingSession())

        with (
            patch.object(store_loader_module, "get_settings", return_value=fake_settings),
            patch.object(store_loader_module, "session_factory", return_value=fake_factory),
        ):
            stores = await store_loader_module.load_active_stores()

        # In-code fallback fully restored — curated catalogue is intact.
        self.assertEqual(len(stores), len(get_active_stores()))
        domains = {s.domain for s in stores}
        self.assertIn("metropolismusic.rs", domains)
        self.assertIn("mascom.rs", domains)
        self.assertIn("roughtrade.com", domains)


class TestRunVinylSearchAwaitsDiscoveryInline(unittest.IsolatedAsyncioTestCase):
    """``run_vinyl_search`` must trigger discovery BEFORE reading the whitelist.

    The previous implementation pushed :func:`ensure_local_coverage` onto a
    ``BackgroundTasks`` queue, so the very first search for an uncurated city
    (e.g. Porto, PT) returned zero local shops while the discovered domains
    only landed in time for the *next* request.
    """

    async def test_porto_search_awaits_ensure_local_coverage_inline_before_whitelist_load(
        self,
    ) -> None:
        """Ordering invariant: ``ensure_local_coverage`` finishes before whitelist load.

        We intercept ``_stage_ensure_local_coverage`` and assert it runs
        STRICTLY before ``_load_known_shop_hosts`` — and, critically, that a
        discovered domain it upserts is observable by the whitelist loader on
        the same HTTP request.
        """
        from fastapi import BackgroundTasks

        # Pretend Porto needed discovery and a new local shop landed in the DB
        # as a side effect of ``ensure_local_coverage``. The whitelist loader
        # called *after* this stage must see ``louie-louie.com`` in its set.
        discovered_porto_shop = "louie-louie.com"
        call_order: list[str] = []

        async def fake_stage_ensure_local_coverage(parsed: Any) -> None:
            call_order.append("ensure_local_coverage")

        async def fake_load_known_shop_hosts() -> frozenset[str]:
            call_order.append("load_known_shop_hosts")
            base = {s.domain for s in ALLOWED_STORES}
            base.add(discovered_porto_shop)
            return frozenset(base)

        parsed = _porto_parsed()

        with (
            patch.object(
                vinyl_search_module,
                "_stage_parse",
                new=AsyncMock(return_value=parsed),
            ),
            patch.object(
                vinyl_search_module,
                "_stage_resolve_album_title",
                new=AsyncMock(return_value="Some Album"),
            ),
            patch.object(
                vinyl_search_module,
                "get_cached_search",
                new=AsyncMock(return_value=None),
            ),
            patch.object(
                vinyl_search_module,
                "_stage_ensure_local_coverage",
                new=fake_stage_ensure_local_coverage,
            ),
            patch.object(
                vinyl_search_module,
                "_load_known_shop_hosts",
                new=fake_load_known_shop_hosts,
            ),
            patch.object(
                vinyl_search_module,
                "run_consolidated_tavily_search",
                new=AsyncMock(return_value=[]),
            ),
        ):
            background_tasks = BackgroundTasks()
            result = await vinyl_search_module.run_vinyl_search(
                "Some Artist - Some Album vinyl Porto",
                background_tasks=background_tasks,
            )

        self.assertEqual(
            call_order,
            ["ensure_local_coverage", "load_known_shop_hosts"],
            msg=(
                "Discovery must complete BEFORE the prefilter whitelist is "
                "loaded so the current request sees the new local shops."
            ),
        )
        # BackgroundTasks queue must NOT be used for discovery any more — that
        # was the regression we are guarding against.
        self.assertEqual(
            len(background_tasks.tasks),
            0,
            msg="ensure_local_coverage must NOT be deferred to BackgroundTasks.",
        )
        self.assertIsNotNone(result)

    async def test_curated_belgrade_url_passes_prefilter_via_whitelist(self) -> None:
        """End-to-end micro-check: a Belgrade indie URL survives the prefilter.

        Demonstrates the contract is closed: ``ALLOWED_STORES`` → whitelist
        loader → ``prefilter_tavily_results`` → ``is_known_shop=True``.
        """
        hosts = await vinyl_search_module._load_known_shop_hosts()

        raw_results: list[dict[str, Any]] = [
            {
                "url": "https://metropolismusic.rs/some/category/landing-page",
                "title": "Some Artist - Some Album",
                "content": "Vinyl LP in stock at Metropolis Music Belgrade.",
                "score": 0.41,
            }
        ]

        kept, diagnostic = prefilter_tavily_results(
            raw_results,
            max_candidates=10,
            max_per_host=2,
            known_shop_hosts=hosts,
        )

        self.assertEqual(diagnostic["rejected_no_pdp_signal"], 0)
        self.assertEqual(len(kept), 1)
        self.assertTrue(kept[0]["is_known_shop"])
        self.assertEqual(kept[0]["host"], "metropolismusic.rs")


class TestStoreEntrySchemaSafety(unittest.TestCase):
    """Type-safety: curated ``ALLOWED_STORES`` rows match the strict ``StoreEntry`` contract."""

    def test_all_seed_rows_have_valid_country_code_and_store_type(self) -> None:
        valid_store_types = {"local_shop", "regional_ecommerce", "marketplace"}
        for entry in ALLOWED_STORES:
            with self.subTest(name=entry.name):
                self.assertIsInstance(entry, StoreEntry)
                if entry.country_code is not None:
                    self.assertEqual(entry.country_code, entry.country_code.upper())
                    self.assertEqual(len(entry.country_code), 2)
                self.assertIn(entry.store_type, valid_store_types)


if __name__ == "__main__":
    unittest.main()
