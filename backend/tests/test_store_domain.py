"""Canonical store hostname normalization."""

from __future__ import annotations

import unittest

from app.policies.store_domain import (
    canonical_store_domain,
    is_valid_store_host,
    registrable_host_only,
)


class StoreDomainCanonicalTests(unittest.TestCase):
    def test_niche_records_legacy_hostname_maps_to_live_storefront(self) -> None:
        self.assertEqual(canonical_store_domain("niche-records.ro"), "nicherecords.ro")
        self.assertEqual(
            canonical_store_domain("https://www.nicherecords.ro/catalog/foo.html"),
            "nicherecords.ro",
        )

    def test_registrable_host_only_skips_alias_table(self) -> None:
        self.assertEqual(registrable_host_only("niche-records.ro"), "niche-records.ro")
        self.assertEqual(registrable_host_only("https://www.NicheRecords.ro/x"), "nicherecords.ro")


class IsValidStoreHostTests(unittest.TestCase):
    """Defence-in-depth filter for Tavily ``include_domains``.

    These are the strings most likely to leak in from the discovery LLM, manual
    SQL edits, or empty Pydantic defaults; any of them in ``include_domains``
    silently poisons a Tavily request — see ``run_tavily_for_store_domains``.
    """

    def test_rejects_known_placeholders(self) -> None:
        for raw in (
            "none",
            "None",
            "NONE",
            "null",
            "unknown",
            "n/a",
            "not provided",
            "Not Provided",
            "not_provided",
            "not-specified",
            "tbd",
            "missing",
            "example.com",
        ):
            with self.subTest(raw=raw):
                self.assertFalse(is_valid_store_host(raw))

    def test_rejects_empty_and_none(self) -> None:
        self.assertFalse(is_valid_store_host(None))
        self.assertFalse(is_valid_store_host(""))
        self.assertFalse(is_valid_store_host("   "))

    def test_rejects_single_label_hosts(self) -> None:
        self.assertFalse(is_valid_store_host("localhost"))
        self.assertFalse(is_valid_store_host("intranet"))

    def test_rejects_whitespace_and_illegal_chars(self) -> None:
        self.assertFalse(is_valid_store_host("foo bar.com"))
        self.assertFalse(is_valid_store_host("shop_name.com"))
        self.assertFalse(is_valid_store_host("шоп.рф"))  # Cyrillic — no DNS legal label here

    def test_accepts_real_eu_shop_hosts(self) -> None:
        for raw in (
            "mascom.rs",
            "hhv.de",
            "groovierecords.com",
            "https://www.misbits.ro/produs/x",
            "phono.cz",
            "rough-trade.com",
        ):
            with self.subTest(raw=raw):
                self.assertTrue(is_valid_store_host(raw))


if __name__ == "__main__":
    unittest.main()
