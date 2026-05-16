"""City matching exonyms used for local-shop catalogue pairing."""

from __future__ import annotations

import unittest

from app.policies.geo_proximity import cities_match


class TestCitySynonyms(unittest.TestCase):
    def test_prague_praha(self) -> None:
        self.assertTrue(cities_match("Prague", "Praha"))
        self.assertTrue(cities_match("praha", "Prague"))

    def test_bucharest_romanian_spelling(self) -> None:
        self.assertTrue(cities_match("Bucharest", "București"))
        self.assertTrue(cities_match("Bucuresti", "bucharest"))


if __name__ == "__main__":
    unittest.main()
