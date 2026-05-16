"""Canonical store hostname normalization."""

from __future__ import annotations

import unittest

from app.policies.store_domain import canonical_store_domain, registrable_host_only


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


if __name__ == "__main__":
    unittest.main()
