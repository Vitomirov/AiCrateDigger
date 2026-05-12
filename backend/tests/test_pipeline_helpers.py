"""Smoke tests for Tavily domain batching and extractor coercions (regression guards)."""

from __future__ import annotations

import unittest

from app.llm.coerce_listing_fields import coerce_in_stock
from app.services.tavily_domain_batches import chunk_include_domains


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
