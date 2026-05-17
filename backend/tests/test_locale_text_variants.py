from __future__ import annotations

import unittest

from app.agents.extractor.intent_match import intent_matches_snippet
from app.policies.locale_text_variants import expand_album_glyph_variants, expand_artist_glyph_variants


class TestArtistGlyphVariants(unittest.TestCase):
    def test_ascii_artist_maps_l_to_stroke_under_pl_country(self) -> None:
        names = expand_artist_glyph_variants("Mgla", country_code="PL", max_variants=20)
        self.assertTrue(any("ł" in n or "Ł" in n for n in names))

    def test_empty_artist(self) -> None:
        self.assertEqual(expand_artist_glyph_variants(None, country_code="DE"), [])
        self.assertEqual(expand_artist_glyph_variants("   ", country_code="DE"), [])


class TestAlbumGlyphVariants(unittest.TestCase):
    def test_polish_country_does_not_lstroke_common_english_l(self) -> None:
        albs = expand_album_glyph_variants("Exercises In Futility", country_code="PL")
        self.assertTrue(albs)
        self.assertFalse(any("futił" in x.lower() for x in albs), albs)


class TestSnippetIntent(unittest.TestCase):
    def test_editorial_hub_passes_when_artist_and_album_match_fuzzy(self) -> None:
        blob_lc = (
            "the vinyl factory feature mgła black metal vinyl pressing restock futility "
            "and exercises catalogue"
        )
        ok = intent_matches_snippet(
            url="https://www.thevinylfactory.com/mag/music-news/feature/",
            blob=blob_lc,
            artist_l="mgla",
            album_l="exercises in futility",
            relaxed_indie_candidate=False,
        )
        self.assertTrue(ok)


class TestGlobalFallbackGate(unittest.TestCase):
    def test_fnac_misc_sku_rejected(self) -> None:
        from app.validators.listings import global_fallback_matches_parsed_entity

        snippet = (
            "I Had Too Much To Dream Last Night - The Electric Prunes - Fnac. "
            "Buy on vinyl EUR 24"
        ).lower()
        ok = global_fallback_matches_parsed_entity(
            listing_title="I Had Too Much To Dream Last Night - The Electric Prunes - Fnac",
            source_snippet=snippet,
            validation_artist="Mgła",
            validation_album="Exercises In Futility",
        )
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
