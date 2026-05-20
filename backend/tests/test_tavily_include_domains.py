"""Tavily include_domains leakage guard."""

from __future__ import annotations

import unittest

from app.models.search_query import SearchResult
from app.services.tavily import enforce_include_domains_hosts


class TestEnforceIncludeDomainsHosts(unittest.TestCase):
    def test_drops_hosts_outside_allowlist(self) -> None:
        rows = [
            SearchResult(
                title="Wrong region",
                url="https://www.bestbuy.com/site/vinyl/123",
                content="snippet",
                score=0.95,
            ),
            SearchResult(
                title="Local shop",
                url="https://shop.example.gr/product/mgla-vinyl",
                content="snippet",
                score=0.72,
            ),
        ]
        out = enforce_include_domains_hosts(rows, ["example.gr"])
        self.assertEqual(len(out), 1)
        self.assertIn("example.gr", out[0].url)

    def test_keeps_subdomain_of_allowed(self) -> None:
        rows = [
            SearchResult(
                title="t",
                url="https://vinyl.sideone.pl/p/1",
                content="c",
                score=0.8,
            ),
        ]
        out = enforce_include_domains_hosts(rows, ["sideone.pl"])
        self.assertEqual(len(out), 1)

    def test_empty_allowlist_is_noop(self) -> None:
        rows = [
            SearchResult(title="t", url="https://a.com/x", content="c", score=0.5),
        ]
        self.assertEqual(enforce_include_domains_hosts(rows, []), rows)


if __name__ == "__main__":
    unittest.main()
