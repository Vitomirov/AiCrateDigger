"""Regression tests for geo widening and proximity bonuses."""

from __future__ import annotations

import unittest

from app.policies.geo_proximity import cities_match, geo_proximity_bonus
from app.policies.geo_scope import (
    GeoIntent,
    NormalizedGeoIntent,
    expand_ships_to,
    tier_fallback_order,
)


class TestGeoProximity(unittest.TestCase):
    def test_city_typo_matches(self) -> None:
        self.assertTrue(cities_match("Barcelona", "barselona"))

    def test_same_city_country_100(self) -> None:
        b = geo_proximity_bonus(
            store_country="ES",
            store_city="Barcelona",
            store_commerce_region="southern_europe",
            target_country="ES",
            target_city="Barcelona",
            target_commerce_region="southern_europe",
            ships_expanded=expand_ships_to(("EU",)),
        )
        self.assertEqual(b, 100.0)

    def test_same_country_different_city_60(self) -> None:
        b = geo_proximity_bonus(
            store_country="ES",
            store_city="Madrid",
            store_commerce_region="southern_europe",
            target_country="ES",
            target_city="Barcelona",
            target_commerce_region="southern_europe",
            ships_expanded=expand_ships_to(("EU",)),
        )
        self.assertEqual(b, 60.0)

    def test_eu_shipper_abroad_is_weak_signal(self) -> None:
        b = geo_proximity_bonus(
            store_country="DE",
            store_city="Berlin",
            store_commerce_region="central_europe",
            target_country="ES",
            target_city="Barcelona",
            target_commerce_region="southern_europe",
            ships_expanded=expand_ships_to(("EU",)),
        )
        self.assertEqual(b, 5.0)


class TestTierFallback(unittest.TestCase):
    def test_city_query_includes_city_phase_first(self) -> None:
        geo = GeoIntent(
            search_scope="local",
            raw_location="Barcelona",
            country_code="ES",
            region="southern_europe",
        )
        norm = NormalizedGeoIntent(
            raw_location="Barcelona",
            resolved_city="Barcelona",
            resolved_country="ES",
            confidence=0.95,
            granularity="city",
        )
        self.assertEqual(tier_fallback_order(geo, norm), ("city", "country", "region", "continental"))

    def test_country_only_skips_city_tier(self) -> None:
        geo = GeoIntent(
            search_scope="local",
            raw_location="Spain",
            country_code="ES",
            region="southern_europe",
        )
        norm = NormalizedGeoIntent(
            raw_location="Spain",
            resolved_city=None,
            resolved_country="ES",
            confidence=0.9,
            granularity="country",
        )
        self.assertEqual(tier_fallback_order(geo, norm), ("country", "region", "continental"))


if __name__ == "__main__":
    unittest.main()
