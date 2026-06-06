"""Tests for search intent derivation."""

from __future__ import annotations

import unittest

from app.domains.query_parser.parse_schema import ParsedQuery
from app.domains.search_pipeline.search_intent import (
    cache_album_segment,
    empty_reason_for_unresolved,
    resolve_search_intent,
)


def _parsed(**kwargs: object) -> ParsedQuery:
    base = {"original_query": "test", "language": "unknown"}
    base.update(kwargs)
    return ParsedQuery.model_validate(base)


class TestResolveSearchIntent(unittest.TestCase):
    def test_release_when_album_present(self) -> None:
        parsed = _parsed(artist="Pink Floyd", album="The Wall", country_code="PL")
        self.assertEqual(resolve_search_intent(parsed), "release")

    def test_release_when_resolved_album_present(self) -> None:
        parsed = _parsed(
            artist="Tool",
            album_index=3,
            resolved_album="Lateralus",
            resolution_confidence="high",
        )
        self.assertEqual(resolve_search_intent(parsed), "release")

    def test_artist_catalog_with_country(self) -> None:
        parsed = _parsed(artist="Mgła", country_code="PL", search_scope="local")
        self.assertEqual(resolve_search_intent(parsed), "artist_catalog")

    def test_artist_catalog_with_city(self) -> None:
        parsed = _parsed(
            artist="Iron Maiden",
            resolved_city="Oslo",
            country_code="NO",
            search_scope="local",
        )
        self.assertEqual(resolve_search_intent(parsed), "artist_catalog")

    def test_unresolved_artist_only(self) -> None:
        parsed = _parsed(artist="Miles Davis", search_scope="global")
        self.assertEqual(resolve_search_intent(parsed), "unresolved")

    def test_unresolved_no_artist(self) -> None:
        parsed = _parsed(location="Oslo", country_code="NO", search_scope="local")
        self.assertEqual(resolve_search_intent(parsed), "unresolved")

    def test_album_unresolved_reason_for_failed_ordinal(self) -> None:
        parsed = _parsed(artist="Tool", album_index=3, resolution_confidence="low")
        self.assertEqual(resolve_search_intent(parsed), "unresolved")
        self.assertEqual(empty_reason_for_unresolved(parsed), "album_unresolved")

    def test_cache_catalog_segment_for_artist_catalog(self) -> None:
        self.assertEqual(
            cache_album_segment(intent="artist_catalog", album=None),
            "catalog",
        )


if __name__ == "__main__":
    unittest.main()
