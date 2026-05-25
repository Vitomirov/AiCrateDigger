"""Tests for city-aware Tavily queries, opportunistic short-circuit, prefilter whitelist."""

from __future__ import annotations

import os
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

os.environ.setdefault("OPENAI_API_KEY", "sk-pipeline-fixes-test")
os.environ.setdefault("TAVILY_API_KEY", "tv-pipeline-fixes-test")
os.environ["DATABASE_URL"] = ""
os.environ["REDIS_URL"] = ""

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.domains.engine.search.prefilter import (  # noqa: E402
    _host_in_whitelist,
    prefilter_tavily_results,
)
from app.domains.engine.search.single_call import build_consolidated_query  # noqa: E402
from app.domains.query_parser.parse_schema import ParsedQuery  # noqa: E402
from app.domains.search_pipeline import vinyl_search as vinyl_search_module  # noqa: E402


class TestBuildConsolidatedQuery(unittest.TestCase):
    def test_injects_resolved_city_before_country_code(self) -> None:
        q = build_consolidated_query(
            artist="The Rolling Stones",
            album="In the stars",
            format_token="vinyl",
            country_code="LV",
            resolved_city="Riga",
        )
        self.assertIn('"The Rolling Stones"', q)
        self.assertIn('"In the stars"', q)
        self.assertIn("vinyl shop Riga LV", q)

    def test_omits_city_when_not_provided(self) -> None:
        q = build_consolidated_query(
            artist="Artist",
            album="Album",
            format_token="vinyl",
            country_code="DE",
            resolved_city=None,
        )
        self.assertEqual(q, '"Artist" "Album" vinyl shop DE')


class TestTavilyCityToken(unittest.TestCase):
    def test_uses_resolved_city(self) -> None:
        parsed = ParsedQuery(
            artist="A",
            album="B",
            location="Porto",
            country_code="PT",
            search_scope="local",
            resolved_city="Porto",
            geo_granularity="city",
            original_query="A B vinyl Porto",
        )
        self.assertEqual(vinyl_search_module._tavily_city_token(parsed), "Porto")

    def test_falls_back_to_location_for_city_granularity(self) -> None:
        parsed = ParsedQuery(
            artist="A",
            album="B",
            location="Tallinn",
            country_code="EE",
            search_scope="local",
            resolved_city=None,
            geo_granularity="city",
            original_query="A B vinyl Tallinn",
        )
        self.assertEqual(vinyl_search_module._tavily_city_token(parsed), "Tallinn")


class TestPrimaryDiscoveryShortCircuit(unittest.IsolatedAsyncioTestCase):
    async def test_opportunistic_skipped_when_primary_triggered(self) -> None:
        parsed = ParsedQuery(
            artist="A",
            album="B",
            location="City",
            country_code="XX",
            search_scope="local",
            resolved_city="City",
            geo_granularity="city",
            original_query="query",
        )
        primary_summary: dict[str, object] = {
            "triggered": True,
            "discovery": {"inserted": 0, "updated": 0, "domains_inserted": []},
        }

        with patch(
            "app.domains.engine.search.store_discovery.discover_stores_from_snippets",
            new=AsyncMock(),
        ) as mock_discover:
            result = await vinyl_search_module._stage_opportunistic_store_discovery(
                parsed=parsed,
                raw_results=[{"url": "https://example-shop.com/p/1", "title": "t", "content": "c"}],
                known_shop_hosts=frozenset(),
                primary_discovery_summary=primary_summary,
            )

        mock_discover.assert_not_awaited()
        self.assertEqual(result, frozenset())

    async def test_opportunistic_skipped_when_primary_inserted(self) -> None:
        parsed = ParsedQuery(
            artist="A",
            album="B",
            location="City",
            country_code="XX",
            search_scope="local",
            resolved_city="City",
            geo_granularity="city",
            original_query="query",
        )
        primary_summary: dict[str, object] = {
            "triggered": False,
            "discovery": {"inserted": 2, "domains_inserted": ["indie.example"]},
        }

        with patch(
            "app.domains.engine.search.store_discovery.discover_stores_from_snippets",
            new=AsyncMock(),
        ) as mock_discover:
            result = await vinyl_search_module._stage_opportunistic_store_discovery(
                parsed=parsed,
                raw_results=[
                    {"url": "https://a.example/p/1", "title": "t", "content": "c"},
                    {"url": "https://b.example/p/2", "title": "t2", "content": "c2"},
                ],
                known_shop_hosts=frozenset(),
                primary_discovery_summary=primary_summary,
            )

        mock_discover.assert_not_awaited()
        self.assertEqual(result, frozenset())


class TestPrefilterWhitelistMatching(unittest.TestCase):
    def test_parent_domain_matches_whitelist_subdomain(self) -> None:
        whitelist = frozenset({"shop.indie-records.example"})
        self.assertTrue(_host_in_whitelist("indie-records.example", whitelist))

    def test_whitelisted_local_shop_passes_thin_path(self) -> None:
        raw: list[dict[str, Any]] = [
            {
                "url": "https://indie-records.example/en/catalog/some-release",
                "title": "Some release",
                "content": "In stock",
                "score": 0.42,
            }
        ]
        kept, diag = prefilter_tavily_results(
            raw,
            known_shop_hosts=frozenset({"indie-records.example"}),
        )
        self.assertEqual(diag["rejected_no_pdp_signal"], 0)
        self.assertEqual(len(kept), 1)
        self.assertTrue(kept[0]["is_known_shop"])


class TestMergeDiscoveryDomains(unittest.TestCase):
    def test_merges_inserted_and_updated_domains(self) -> None:
        base = frozenset({"curated.example"})
        summary: dict[str, object] = {
            "discovery": {
                "domains_inserted": ["fresh-local.example"],
                "domains_updated": ["https://www.renewed-local.example/"],
            }
        }
        merged = vinyl_search_module._merge_discovery_domains_into_hosts(base, summary)
        self.assertIn("curated.example", merged)
        self.assertIn("fresh-local.example", merged)
        self.assertIn("renewed-local.example", merged)


if __name__ == "__main__":
    unittest.main()
