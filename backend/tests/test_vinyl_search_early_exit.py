"""Regression: structured empty search responses when no album anchor exists."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.domain.parse_schema import ParsedQuery
from app.pipeline.vinyl_search import run_vinyl_search
from app.services.discogs_service import Album, AlbumResolution


class TestVinylSearchAlbumUnresolved(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._settings_patcher = patch(
            "app.pipeline.vinyl_search.get_settings",
            return_value=SimpleNamespace(debug=False),
        )
        self._settings_patcher.start()

    def tearDown(self) -> None:
        self._settings_patcher.stop()
        super().tearDown()

    async def test_reason_when_no_album_and_no_ordinal(self) -> None:
        pq = ParsedQuery(
            artist="Someone",
            album=None,
            album_index=None,
            original_query="Someone vinyl",
        )
        with patch(
            "app.pipeline.vinyl_search.parse_user_query",
            new_callable=AsyncMock,
            return_value=pq,
        ):
            out = await run_vinyl_search("Someone vinyl")
        self.assertEqual(out["results"], [])
        self.assertEqual(out["reason"], "album_unresolved")
        self.assertIs(out["parsed"], pq)

    async def test_reason_when_album_is_whitespace_only(self) -> None:
        pq = ParsedQuery(
            artist="Someone",
            album="   ",
            album_index=None,
            original_query="Someone vinyl",
        )
        with patch(
            "app.pipeline.vinyl_search.parse_user_query",
            new_callable=AsyncMock,
            return_value=pq,
        ):
            out = await run_vinyl_search("Someone vinyl")
        self.assertEqual(out["reason"], "album_unresolved")

    async def test_reason_when_discogs_returns_no_album_for_ordinal(self) -> None:
        pq = ParsedQuery(
            artist="Someone",
            album=None,
            album_index=3,
            original_query="third Someone album vinyl",
        )
        with (
            patch(
                "app.pipeline.vinyl_search.parse_user_query",
                new_callable=AsyncMock,
                return_value=pq,
            ),
            patch(
                "app.pipeline.vinyl_search.resolve_album_by_index",
                new_callable=AsyncMock,
                return_value=AlbumResolution(album=None, index=None, confidence=0.0),
            ),
        ):
            out = await run_vinyl_search("third Someone album vinyl")
        self.assertEqual(out["reason"], "album_unresolved")

    async def test_reason_when_discogs_title_blank(self) -> None:
        pq = ParsedQuery(
            artist="Someone",
            album=None,
            album_index=1,
            original_query="debut Someone album vinyl",
        )
        bogus = Album(title="   ", year=None, discogs_id="1")
        with (
            patch(
                "app.pipeline.vinyl_search.parse_user_query",
                new_callable=AsyncMock,
                return_value=pq,
            ),
            patch(
                "app.pipeline.vinyl_search.resolve_album_by_index",
                new_callable=AsyncMock,
                return_value=AlbumResolution(album=bogus, index=1, confidence=1.0),
            ),
        ):
            out = await run_vinyl_search("debut Someone album vinyl")
        self.assertEqual(out["reason"], "album_unresolved")


if __name__ == "__main__":
    unittest.main()
