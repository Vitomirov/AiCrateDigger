"""Tests for deterministic gates in ``verify_album_match``."""

from __future__ import annotations

import unittest

from app.domain.listing_schema import Listing
from app.agents.extractor.verify_album_match import verify_album_match


class TestVerifyAlbumDeterministic(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_when_excerpt_has_no_target_album_or_artist(self) -> None:
        lst = Listing(
            title="Classicos GNR",
            price=22.0,
            currency="EUR",
            in_stock=True,
            url="https://shop.example/a",
            store="shop.example",
            source_snippet="Classicos GNR · Guns N Roses vinyl LP EU press",
        )
        out = await verify_album_match(
            [lst],
            artist="Rammstein",
            album_title="Mutter",
        )
        row = out.get("https://shop.example/a")
        self.assertIsNotNone(row)
        self.assertEqual(row.verdict, "reject")
        self.assertIn("deterministic", row.reason)


if __name__ == "__main__":
    unittest.main()
