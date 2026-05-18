"""Smoke tests for Tavily domain batching and extractor coercions (regression guards)."""

from __future__ import annotations

import unittest

from app.llm.coerce_listing_fields import coerce_in_stock
from app.services.tavily_domain_batches import chunk_include_domains
from app.services.tavily_power_query import (
    build_physical_power_query_base,
    chunk_domains_for_power_queries,
)


class TestChunkIncludeDomains(unittest.TestCase):
    def test_single_batch_when_under_cap(self) -> None:
        d = [f"shop{i}.example.com" for i in range(10)]
        chunks = chunk_include_domains(d, 20)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], d)

    def test_splits_at_twenty_one_domains(self) -> None:
        d = [f"s{i}.tld" for i in range(21)]
        chunks = chunk_include_domains(d, 20)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 20)
        self.assertEqual(len(chunks[1]), 1)

    def test_empty_input(self) -> None:
        self.assertEqual(chunk_include_domains([], 20), [])


class TestTavilyPowerQueryChunking(unittest.TestCase):
    def test_base_quotes_artist_album_and_format(self) -> None:
        q = build_physical_power_query_base(
            artist="Iron Maiden",
            album_title="The Number Of The Beast",
        )
        self.assertIn('"Iron Maiden"', q)
        self.assertIn('"The Number Of The Beast"', q)
        self.assertIn("(vinyl OR LP)", q)

    def test_chunking_packs_multiple_domains_when_short(self) -> None:
        base = build_physical_power_query_base(artist="AB", album_title="CD")
        doms = ["a.example", "b.example", "c.example"]
        rows = chunk_domains_for_power_queries(
            base,
            doms,
            max_chars=400,
            max_domains_per_chunk=5,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual({d for c, _ in rows for d in c}, set(doms))
        self.assertIn("site:a.example", rows[0][1])

    def test_chunking_splits_on_length_budget(self) -> None:
        base = build_physical_power_query_base(artist="X", album_title="Y")
        doms = [f"shop{i:02d}.vinyl.test" for i in range(8)]
        rows = chunk_domains_for_power_queries(
            base,
            doms,
            max_chars=120,
            max_domains_per_chunk=5,
        )
        self.assertGreaterEqual(len(rows), 2)
        covered = [h for chunk, _ in rows for h in chunk]
        self.assertEqual(len(covered), len(doms))
        self.assertEqual(set(covered), set(doms))


class TestCoerceInStock(unittest.TestCase):
    def test_bool_passthrough(self) -> None:
        self.assertIs(coerce_in_stock({"in_stock": True}), True)
        self.assertIs(coerce_in_stock({"in_stock": False}), False)

    def test_none_defaults_true(self) -> None:
        self.assertIs(coerce_in_stock({}), True)
        self.assertIs(coerce_in_stock({"in_stock": None}), True)

    def test_strings(self) -> None:
        self.assertIs(coerce_in_stock({"in_stock": "out of stock"}), False)
        self.assertIs(coerce_in_stock({"in_stock": "In Stock!"}), True)
        self.assertIs(coerce_in_stock({"in_stock": "sold out"}), False)

    def test_numeric(self) -> None:
        self.assertIs(coerce_in_stock({"in_stock": 0}), False)
        self.assertIs(coerce_in_stock({"in_stock": 1}), True)

    def test_malformed_no_crash(self) -> None:
        self.assertIs(coerce_in_stock({"in_stock": []}), True)
        self.assertIs(coerce_in_stock({"in_stock": {}}), True)


if __name__ == "__main__":
    unittest.main()
