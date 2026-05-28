"""Tests for fail-closed album field normalization in the parser."""

from __future__ import annotations

import unittest

from app.domains.query_parser.parse_user_query import _build_parsed_payload, _derive_album_fields


class TestAlbumFieldDerivation(unittest.TestCase):
    def test_explicit_album_clears_ordinal_fields(self) -> None:
        album, album_index, resolved, confidence = _derive_album_fields(
            album="OK Computer",
            album_index=2,
            resolved_album="The Bends",
        )
        self.assertEqual(album, "OK Computer")
        self.assertIsNone(album_index)
        self.assertIsNone(resolved)
        self.assertEqual(confidence, "unknown")

    def test_ordinal_with_resolved_album_is_high_confidence(self) -> None:
        album, album_index, resolved, confidence = _derive_album_fields(
            album=None,
            album_index=2,
            resolved_album="A Saucerful of Secrets",
        )
        self.assertIsNone(album)
        self.assertEqual(album_index, 2)
        self.assertEqual(resolved, "A Saucerful of Secrets")
        self.assertEqual(confidence, "high")

    def test_ordinal_without_title_is_low_confidence(self) -> None:
        album, album_index, resolved, confidence = _derive_album_fields(
            album=None,
            album_index=4,
            resolved_album=None,
        )
        self.assertIsNone(album)
        self.assertEqual(album_index, 4)
        self.assertIsNone(resolved)
        self.assertEqual(confidence, "low")

    def test_build_payload_applies_explicit_album_precedence(self) -> None:
        payload = _build_parsed_payload(
            {
                "artist": "Radiohead",
                "album": "OK Computer",
                "album_index": 2,
                "resolved_album": "The Bends",
                "search_scope": "global",
            },
            "Radiohead OK Computer vinyl",
        )
        self.assertEqual(payload["album"], "OK Computer")
        self.assertIsNone(payload["album_index"])
        self.assertIsNone(payload["resolved_album"])
        self.assertEqual(payload["resolution_confidence"], "unknown")


if __name__ == "__main__":
    unittest.main()
