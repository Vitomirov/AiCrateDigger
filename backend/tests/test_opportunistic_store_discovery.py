"""Regression tests for opportunistic store discovery (Stage 6.5).

These tests pin down the recent fixes that closed the user-visible bug where
searches for cities like Hanover produced zero local results because:

1.  The dedicated discovery probe (Stage 4) sent the wrong Tavily payload
    field (``max_results_per_query`` instead of ``max_results``) so Tavily
    silently returned its 5-result default, starving the LLM verifier.
2.  The LLM rejected every candidate because the confidence floor was 0.5
    and the prompt forced "physically located in the requested city" wording
    — listicle snippets rarely include addresses.
3.  Real local shops (``van-records.com``, ``rockers.de``) **were** present
    in the main search Tavily output, but the prefilter dropped them as
    ``rejected_no_pdp_signal`` AND nothing wrote them to ``whitelist_stores``,
    so DBeaver kept showing zero new rows after every search.

The fix introduces ``discover_stores_from_snippets`` and a Stage 6.5
``_stage_opportunistic_store_discovery`` step that LLM-verifies unknown-host
snippets from the main Tavily call and merges verified hosts into the current
request's prefilter whitelist.
"""

from __future__ import annotations

import os
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "sk-opportunistic-test")
os.environ.setdefault("TAVILY_API_KEY", "tv-opportunistic-test")
os.environ["DATABASE_URL"] = ""
os.environ["REDIS_URL"] = ""

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.domains.engine.search import store_discovery as store_discovery_module  # noqa: E402
from app.domains.query_parser.parse_schema import ParsedQuery  # noqa: E402
from app.domains.search_pipeline import vinyl_search as vinyl_search_module  # noqa: E402


def _hanover_parsed() -> ParsedQuery:
    return ParsedQuery(
        artist="Cascet",
        album="Undead Soil",
        location="Hanover",
        country_code="DE",
        search_scope="local",
        resolved_city="Hanover",
        geo_granularity="city",
        original_query="Undead Soil by Cascet in Hanover",
    )


def _hanover_main_tavily_results() -> list[dict[str, Any]]:
    """Approximates the user's actual Stage 6 output for the Hanover query.

    Includes blacklisted hosts, the curated giants, and the unknown German
    indie hosts that the buggy prefilter dropped.
    """
    return [
        {
            "url": "https://www.ebay.de/itm/123",
            "title": "Cascet Undead Soil on eBay Germany",
            "content": "Buy now",
            "score": 0.6,
        },
        {
            "url": "https://casketdeath.bandcamp.com/album/undead-soil",
            "title": "Undead Soil | Casket",
            "content": "Bandcamp release",
            "score": 0.5,
        },
        {
            "url": "https://www.jpc.de/jpcng/poprock/detail/-/art/casket-undead-soil/hnum/9012345",
            "title": "Casket Undead Soil – JPC",
            "content": "JPC Germany product page",
            "score": 0.5,
        },
        {
            "url": "https://shop.animate-records.com/products/casket-undead-soil",
            "title": "Casket - Undead Soil LP | Animate Records",
            "content": "Vinyl LP available at Animate Records Hannover",
            "score": 0.4,
        },
        {
            "url": "https://supremechaos.de/release/casket-undead-soil/",
            "title": "Casket - Undead Soil (Supreme Chaos Records)",
            "content": "Supreme Chaos Records, Germany",
            "score": 0.4,
        },
        {
            "url": "https://van-records.com/shop/article/casket-undead-soil/",
            "title": "Casket - Undead Soil LP — Van Records",
            "content": "Van Records, German metal label & shop",
            "score": 0.4,
        },
        {
            "url": "https://discogs.com/release/123-casket-undead-soil",
            "title": "Casket - Undead Soil on Discogs",
            "content": "marketplace",
            "score": 0.3,
        },
    ]


class TestSelectUnknownHostSnippets(unittest.TestCase):
    """Pure unit test on the snippet-selection helper."""

    def test_skips_blacklisted_and_whitelisted_hosts(self) -> None:
        raw = _hanover_main_tavily_results()
        # Pretend JPC is curated already.
        whitelist = frozenset({"jpc.de"})

        chosen = vinyl_search_module._select_unknown_host_snippets_for_discovery(
            raw,
            known_shop_hosts=whitelist,
        )
        hosts = {entry["url"].split("/")[2].removeprefix("www.") for entry in chosen}

        # ebay (blacklisted), bandcamp (blacklisted), discogs (blacklisted),
        # jpc (whitelisted) all skipped. Three German indies remain.
        self.assertNotIn("ebay.de", hosts)
        self.assertNotIn("www.ebay.de", hosts)
        self.assertNotIn("casketdeath.bandcamp.com", hosts)
        self.assertNotIn("jpc.de", hosts)
        self.assertNotIn("discogs.com", hosts)
        self.assertIn("shop.animate-records.com", hosts)
        self.assertIn("supremechaos.de", hosts)
        self.assertIn("van-records.com", hosts)

    def test_dedupes_by_host(self) -> None:
        raw = [
            {"url": "https://van-records.com/shop/a", "title": "a", "content": "x", "score": 0.4},
            {"url": "https://van-records.com/shop/b", "title": "b", "content": "y", "score": 0.5},
            {"url": "https://van-records.com/shop/c", "title": "c", "content": "z", "score": 0.6},
        ]
        chosen = vinyl_search_module._select_unknown_host_snippets_for_discovery(
            raw,
            known_shop_hosts=frozenset(),
        )
        self.assertEqual(len(chosen), 1)
        self.assertTrue(chosen[0]["url"].startswith("https://van-records.com/"))


class TestStageOpportunisticDiscovery(unittest.IsolatedAsyncioTestCase):
    """Stage 6.5: discovered hosts must be merged into THIS request's whitelist."""

    async def test_verified_hosts_are_merged_into_whitelist_and_persisted(
        self,
    ) -> None:
        """The closed-loop assertion users wanted: shops surfaced by Tavily AND
        verified by the discovery LLM are both injected into the prefilter
        whitelist for *this* request AND upserted into the DB so future
        queries (and DBeaver) see them too.
        """
        parsed = _hanover_parsed()
        raw_results = _hanover_main_tavily_results()
        initial_whitelist = frozenset({"jpc.de"})

        fake_report = store_discovery_module.DiscoveryReport(
            inserted=2,
            updated=1,
            rejected=0,
            candidates=3,
            domains_inserted=["shop.animate-records.com", "supremechaos.de"],
            domains_updated=["van-records.com"],
            error=None,
        )

        with patch.object(
            store_discovery_module,
            "discover_stores_from_snippets",
            new=AsyncMock(return_value=fake_report),
        ) as mock_discover:
            augmented = await vinyl_search_module._stage_opportunistic_store_discovery(
                parsed=parsed,
                raw_results=raw_results,
                known_shop_hosts=initial_whitelist,
            )

        mock_discover.assert_awaited_once()
        kwargs = mock_discover.await_args.kwargs
        self.assertEqual(kwargs["city"], "Hanover")
        self.assertEqual(kwargs["country_code"], "DE")
        forwarded_hosts = {
            entry["url"].split("/")[2].removeprefix("www.")
            for entry in kwargs["snippets"]
        }
        # Curated + blacklist hosts must NEVER reach the discovery LLM.
        self.assertNotIn("jpc.de", forwarded_hosts)
        self.assertNotIn("www.ebay.de", forwarded_hosts)
        self.assertNotIn("casketdeath.bandcamp.com", forwarded_hosts)
        # Real indies make it through.
        self.assertIn("shop.animate-records.com", forwarded_hosts)
        self.assertIn("supremechaos.de", forwarded_hosts)
        self.assertIn("van-records.com", forwarded_hosts)

        # Closed-loop: hosts the LLM verified are in the augmented whitelist.
        self.assertIn("jpc.de", augmented)
        self.assertIn("shop.animate-records.com", augmented)
        self.assertIn("supremechaos.de", augmented)
        self.assertIn("van-records.com", augmented)

    async def test_disabled_via_settings_returns_input_whitelist_unchanged(
        self,
    ) -> None:
        """Operator-grade kill switch: the Settings flag short-circuits the stage."""
        parsed = _hanover_parsed()
        raw_results = _hanover_main_tavily_results()
        initial_whitelist = frozenset({"jpc.de"})

        fake_settings = MagicMock()
        fake_settings.pipeline_opportunistic_store_discovery_enabled = False
        fake_settings.pipeline_opportunistic_discovery_min_unknown_hosts = 2

        with (
            patch.object(vinyl_search_module, "get_settings", return_value=fake_settings),
            patch.object(
                store_discovery_module,
                "discover_stores_from_snippets",
                new=AsyncMock(),
            ) as mock_discover,
        ):
            augmented = await vinyl_search_module._stage_opportunistic_store_discovery(
                parsed=parsed,
                raw_results=raw_results,
                known_shop_hosts=initial_whitelist,
            )

        mock_discover.assert_not_awaited()
        self.assertEqual(augmented, initial_whitelist)

    async def test_no_city_or_country_short_circuits(self) -> None:
        """Pipeline mustn't burn an LLM call when the parser couldn't resolve geo."""
        parsed = ParsedQuery(
            artist="Some Artist",
            album="Some Album",
            search_scope="global",
            original_query="Some Artist - Some Album vinyl",
        )

        with patch.object(
            store_discovery_module,
            "discover_stores_from_snippets",
            new=AsyncMock(),
        ) as mock_discover:
            augmented = await vinyl_search_module._stage_opportunistic_store_discovery(
                parsed=parsed,
                raw_results=_hanover_main_tavily_results(),
                known_shop_hosts=frozenset({"jpc.de"}),
            )

        mock_discover.assert_not_awaited()
        self.assertEqual(augmented, frozenset({"jpc.de"}))

    async def test_thin_signal_below_threshold_is_skipped(self) -> None:
        """Single unknown host → not worth an LLM call."""
        parsed = _hanover_parsed()
        # Only one non-blacklisted unknown host left; threshold default is 2.
        raw_results = [
            {
                "url": "https://van-records.com/shop/a",
                "title": "a",
                "content": "x",
                "score": 0.4,
            },
            {
                "url": "https://discogs.com/release/x",
                "title": "ds",
                "content": "",
                "score": 0.3,
            },
        ]
        with patch.object(
            store_discovery_module,
            "discover_stores_from_snippets",
            new=AsyncMock(),
        ) as mock_discover:
            augmented = await vinyl_search_module._stage_opportunistic_store_discovery(
                parsed=parsed,
                raw_results=raw_results,
                known_shop_hosts=frozenset({"jpc.de"}),
            )

        mock_discover.assert_not_awaited()
        self.assertEqual(augmented, frozenset({"jpc.de"}))

    async def test_discovery_exception_is_swallowed_and_does_not_corrupt_whitelist(
        self,
    ) -> None:
        """LLM / DB failure inside discovery must NEVER take down the user's search."""
        parsed = _hanover_parsed()
        raw_results = _hanover_main_tavily_results()
        initial_whitelist = frozenset({"jpc.de", "roughtrade.com"})

        with patch.object(
            store_discovery_module,
            "discover_stores_from_snippets",
            new=AsyncMock(side_effect=RuntimeError("LLM 500")),
        ):
            augmented = await vinyl_search_module._stage_opportunistic_store_discovery(
                parsed=parsed,
                raw_results=raw_results,
                known_shop_hosts=initial_whitelist,
            )

        # Whitelist preserved exactly — no widening, no narrowing.
        self.assertEqual(augmented, initial_whitelist)


class TestStoreDiscoveryProbeUsesCorrectTavilyContract(
    unittest.IsolatedAsyncioTestCase
):
    """Bug fix lock-in: the Tavily payload uses ``max_results`` (not the
    silently-ignored ``max_results_per_query``) and passes the structured
    ``country`` field so geo-relevant results rank higher."""

    async def test_probe_payload_uses_max_results_and_country(self) -> None:
        captured: list[dict[str, object]] = []

        async def fake_fetch(
            client: object,
            payload: dict[str, object],
            *,
            query_for_log: str,
        ) -> dict[str, object] | None:
            captured.append(payload)
            return {"results": []}

        with patch.object(
            store_discovery_module,
            "fetch_tavily_results_body",
            new=fake_fetch,
        ):
            await store_discovery_module._tavily_probe("Hanover", "DE")

        # Two probe queries fire (English listicle + shop-shaped).
        self.assertEqual(len(captured), 2)
        for payload in captured:
            self.assertIn(
                "max_results",
                payload,
                msg="Tavily expects ``max_results`` — the legacy ``max_results_per_query`` key was silently ignored.",
            )
            self.assertNotIn(
                "max_results_per_query",
                payload,
                msg="Stale Tavily key crept back into the probe payload.",
            )
            # Country-aware ranking: DE → "germany" structured field.
            self.assertEqual(payload.get("country"), "germany")
            self.assertEqual(payload.get("topic"), "general")
            self.assertEqual(payload.get("search_depth"), "advanced")


class TestLlmConfidenceFloorIsRelaxed(unittest.IsolatedAsyncioTestCase):
    """The minimum confidence is lowered to 0.4 so gpt-4o-mini's cautious
    listicle verdicts (0.4–0.5) no longer get rejected wholesale.
    """

    async def test_candidate_at_confidence_0_4_is_accepted(self) -> None:
        snippets = [
            {
                "title": "Best record shops in Hanover",
                "url": "https://rockers.de/",
                "content": "Rockers Records is a vinyl shop in Hannover, Germany.",
            }
        ]

        class _StubResponse:
            class _Choice:
                class _Msg:
                    content = (
                        '{"stores": [{'
                        '"name": "Rockers Records", '
                        '"domain": "rockers.de", '
                        '"city": "Hannover", '
                        '"country_code": "DE", '
                        '"confidence": 0.4}]}'
                    )

                message = _Msg()

            choices = [_Choice()]

        class _FakeClient:
            def __init__(self, *, api_key: str) -> None:
                self.chat = self

            class _Completions:
                async def create(self, **_: object) -> object:
                    return _StubResponse()

            completions = _Completions()

        with patch.object(store_discovery_module, "AsyncOpenAI", _FakeClient):
            candidates = await store_discovery_module._llm_extract_candidates(
                city="Hanover",
                country_code="DE",
                snippets=snippets,
            )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].domain, "rockers.de")
        self.assertAlmostEqual(candidates[0].confidence, 0.4)


if __name__ == "__main__":
    unittest.main()
