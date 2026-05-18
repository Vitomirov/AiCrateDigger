"""Unit tests for geo widening early-stop policy."""

from __future__ import annotations

import unittest

from app.pipeline.tier_runner.evaluator import compute_early_stop


class TestComputeEarlyStop(unittest.TestCase):
    def test_localized_city_one_listing_stops_before_floor(self) -> None:
        stop, reason = compute_early_stop(
            tier="city",
            aggregated_validated_count=1,
            stop_floor_need=5,
        )
        self.assertTrue(stop)
        self.assertEqual(reason, "localized_satisfied_min_one")

    def test_localized_country_one_listing_stops_before_floor(self) -> None:
        stop, reason = compute_early_stop(
            tier="country",
            aggregated_validated_count=1,
            stop_floor_need=5,
        )
        self.assertTrue(stop)
        self.assertEqual(reason, "localized_satisfied_min_one")

    def test_region_needs_stop_floor_even_with_one(self) -> None:
        stop, reason = compute_early_stop(
            tier="region",
            aggregated_validated_count=1,
            stop_floor_need=5,
        )
        self.assertFalse(stop)
        self.assertIsNone(reason)

    def test_stop_floor_met_any_tier(self) -> None:
        stop, reason = compute_early_stop(
            tier="region",
            aggregated_validated_count=3,
            stop_floor_need=3,
        )
        self.assertTrue(stop)
        self.assertEqual(reason, "stop_floor_met")


if __name__ == "__main__":
    unittest.main()
