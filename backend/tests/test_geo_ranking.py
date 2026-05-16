"""Regression tests for geo widening and proximity bonuses.

``geo_proximity_bonus`` applies a store-type locality factor (``local_shop``,
``regional_ecommerce``, ``marketplace``). Tests pass ``store_type`` explicitly
when asserting full undamped scores; omitting it asserts damped behavior for
the default regional class.
"""

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

    def test_same_city_country_100_for_local_shop(self) -> None:
        """Full 0..100 geo bonus only when locality factor is 1.0 (``local_shop``)."""
        b = geo_proximity_bonus(
            store_country="ES",
            store_city="Barcelona",
            store_commerce_region="southern_europe",
            target_country="ES",
            target_city="Barcelona",
            target_commerce_region="southern_europe",
            ships_expanded=expand_ships_to(("EU",)),
            store_type="local_shop",
        )
        self.assertEqual(b, 100.0)

    def test_same_city_country_regional_ecommerce_is_dampened(self) -> None:
        """Missing / generic store class defaults to regional ecommerce (×0.55)."""
        b = geo_proximity_bonus(
            store_country="ES",
            store_city="Barcelona",
            store_commerce_region="southern_europe",
            target_country="ES",
            target_city="Barcelona",
            target_commerce_region="southern_europe",
            ships_expanded=expand_ships_to(("EU",)),
        )
        self.assertAlmostEqual(b, 100.0 * 0.55)

    def test_same_country_different_city_50_for_local_shop(self) -> None:
        """Same-country non-city match uses a 50-point base (× locality factor)."""
        b = geo_proximity_bonus(
            store_country="ES",
            store_city="Madrid",
            store_commerce_region="southern_europe",
            target_country="ES",
            target_city="Barcelona",
            target_commerce_region="southern_europe",
            ships_expanded=expand_ships_to(("EU",)),
            store_type="local_shop",
        )
        self.assertEqual(b, 50.0)

    def test_eu_shipper_abroad_is_weak_signal_for_local_shop(self) -> None:
        """EU-only ship match uses a 4-point base (× locality factor)."""
        b = geo_proximity_bonus(
            store_country="DE",
            store_city="Berlin",
            store_commerce_region="central_europe",
            target_country="ES",
            target_city="Barcelona",
            target_commerce_region="southern_europe",
            ships_expanded=expand_ships_to(("EU",)),
            store_type="local_shop",
        )
        self.assertEqual(b, 4.0)


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
