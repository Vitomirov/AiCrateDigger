"""Unit tests for ``app.policies.physical_local`` (local-first locality signals)."""

from __future__ import annotations

import unittest

from app.policies.eu_stores import StoreEntry
from app.policies.geo_scope import GeoIntent, NormalizedGeoIntent
from app.policies.physical_local import (
    curated_city_local_shop_domains,
    should_prioritize_physical_local_shops,
)


class TestPrioritizeLocals(unittest.TestCase):
    def test_search_scope_local_enables_prioritize(self) -> None:
        geo = GeoIntent(
            search_scope="local",
            raw_location=None,
            country_code="CZ",
            region=None,
        )
        norm = NormalizedGeoIntent(
            raw_location=None,
            resolved_city="Prague",
            resolved_country="CZ",
            confidence=0.9,
            granularity="city",
        )
        self.assertTrue(should_prioritize_physical_local_shops(geo, norm))

    def test_regional_scope_city_granularity_also_prioritizes_physical(self) -> None:
        geo = GeoIntent(
            search_scope="regional",
            raw_location=None,
            country_code="RO",
            region="central_europe",
        )
        norm = NormalizedGeoIntent(
            raw_location=None,
            resolved_city="Bucharest",
            resolved_country="RO",
            confidence=0.85,
            granularity="city",
        )
        self.assertTrue(should_prioritize_physical_local_shops(geo, norm))


class TestCuratedDomains(unittest.TestCase):
    def test_collects_only_matching_city_local_shop(self) -> None:
        stores = (
            StoreEntry(
                name="Indie Prague",
                domain="indy-prague-store.cz",
                country_code="CZ",
                region="central_europe",
                ships_to=("CZ", "EU"),
                priority=9,
                is_active=True,
                listing_quality=8,
                city="Prague",
                store_type="local_shop",
            ),
            StoreEntry(
                name="Giant EUR",
                domain="hhv.de",
                country_code="DE",
                region="central_europe",
                ships_to=("EU",),
                priority=10,
                is_active=True,
                listing_quality=9,
                store_type="regional_ecommerce",
            ),
        )
        norm = NormalizedGeoIntent(
            raw_location=None,
            resolved_city="Prague",
            resolved_country="CZ",
            confidence=0.82,
            granularity="city",
        )
        d = curated_city_local_shop_domains(stores, norm)
        self.assertIn("indy-prague-store.cz", d)


if __name__ == "__main__":
    unittest.main()
