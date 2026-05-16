"""Tests for ``app.agents.extractor`` (Agent 3).

No network calls: the LLM step is mocked. Pre-filter, normalizers, and host
parsing run with real RapidFuzz / logic.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from app.agents.extractor.constants import MIN_TITLE_LEN
from app.agents.extractor.field_normalizers import (
    clean_optional_string,
    normalize_availability,
    normalize_seller,
)
from app.agents.extractor.fuzzy_snippets import album_fuzzy_score, artist_fuzzy_score
from app.agents.extractor.hosts import normalize_domain
from app.agents.extractor.merch_gate import listing_looks_like_merch
from app.agents.extractor.pipeline import extract_and_score_results
from app.agents.extractor.pre_filter import run_pre_filter
from app.models.search_query import SearchResult
from app.pipeline_context import start_pipeline


class TestNormalizeDomain(unittest.TestCase):
    def test_strips_www_scheme_path(self) -> None:
        self.assertEqual(normalize_domain("https://WWW.Shop.Example.com/path?x=1"), "shop.example.com")

    def test_bare_hostname(self) -> None:
        self.assertEqual(normalize_domain("vinyl.example.rs"), "vinyl.example.rs")

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(normalize_domain(""))
        self.assertIsNone(normalize_domain("   "))


class TestMerchGate(unittest.TestCase):
    def test_detects_poster_in_title(self) -> None:
        self.assertTrue(listing_looks_like_merch("Tour poster", "Some artist"))

    def test_clean_music_title_false(self) -> None:
        self.assertFalse(
            listing_looks_like_merch("Artist Name — Album vinyl LP", "In stock vinyl record shop")
        )


class TestFieldNormalizers(unittest.TestCase):
    def test_clean_optional_string(self) -> None:
        self.assertIsNone(clean_optional_string(None))
        self.assertIsNone(clean_optional_string("   "))
        self.assertEqual(clean_optional_string("  12,90 €  "), "12,90 €")

    def test_normalize_availability(self) -> None:
        self.assertEqual(normalize_availability("SOLD_OUT"), "sold_out")
        self.assertEqual(normalize_availability("nope"), "unknown")

    def test_normalize_seller(self) -> None:
        self.assertEqual(normalize_seller("STORE"), "store")
        self.assertEqual(normalize_seller("x"), "unknown")


class TestFuzzySnippets(unittest.TestCase):
    def test_artist_score_high_when_substring_in_haystack(self) -> None:
        s = artist_fuzzy_score(
            "Electric Light Orchestra",
            "Electric Light Orchestra Out of the Blue vinyl",
            "Buy official release 180g",
        )
        self.assertGreaterEqual(s, 60.0)

    def test_album_zero_when_album_empty(self) -> None:
        self.assertEqual(album_fuzzy_score("", "Title", "content"), 0.0)


class TestRunPreFilter(unittest.TestCase):
    def _sr(self, *, title: str, url: str, content: str) -> SearchResult:
        return SearchResult(title=title, url=url, content=content, score=0.9)

    def test_rejects_short_title(self) -> None:
        short = "a" * (MIN_TITLE_LEN - 1)
        survivors, _, _ = run_pre_filter(
            batch=[self._sr(title=short, url="https://a.com/p", content="Artist X album")],
            artist="Artist X",
            album="Album",
        )
        self.assertEqual(survivors, [])

    def test_rejects_merch_keyword(self) -> None:
        survivors, _, _ = run_pre_filter(
            batch=[
                self._sr(
                    title="Official tote bag merch",
                    url="https://shop.com/m",
                    content="Artist Name",
                )
            ],
            artist="Artist Name",
            album="Album",
        )
        self.assertEqual(survivors, [])

    def test_keeps_matching_artist_and_album(self) -> None:
        survivors, a_scores, b_scores = run_pre_filter(
            batch=[
                self._sr(
                    title="Artist Name — Album Name vinyl LP new",
                    url="https://shop.example.com/product/1",
                    content="Artist Name Album Name vinyl record in stock",
                )
            ],
            artist="Artist Name",
            album="Album Name",
        )
        self.assertEqual(len(survivors), 1)
        self.assertEqual(len(a_scores), 1)
        self.assertEqual(len(b_scores), 1)
        self.assertGreater(a_scores[0], 59.0)
        self.assertGreater(b_scores[0], 0.0)

    def test_no_artist_uses_neutral_score_and_survives_if_other_gates_pass(self) -> None:
        survivors, a_scores, _ = run_pre_filter(
            batch=[
                self._sr(
                    title="Some Long Enough Vinyl Title Here",
                    url="https://shop.example.com/p",
                    content="Rare vinyl catalogue",
                )
            ],
            artist="",
            album="Album",
        )
        self.assertEqual(len(survivors), 1)
        self.assertEqual(a_scores[0], 50.0)


class TestExtractAndScoreResultsAsync(unittest.IsolatedAsyncioTestCase):
    """Integration-style tests with mocked LLM only."""

    async def test_empty_candidates_returns_empty(self) -> None:
        with start_pipeline(debug=False):
            out = await extract_and_score_results(
                [],
                artist="A",
                album="B",
                music_format="Vinyl",
                country="GB",
                city="London",
            )
        self.assertEqual(out, [])

    async def test_pre_filter_rejects_all_returns_empty(self) -> None:
        batch = [
            SearchResult(
                title="bad",
                url="https://x.com/y",
                content="Totally unrelated text with no artist",
                score=0.5,
            )
        ]
        with start_pipeline(debug=False):
            out = await extract_and_score_results(
                batch,
                artist="Obscure Artist Name",
                album="Album",
                music_format="Vinyl",
                country="US",
                city=None,
            )
        self.assertEqual(out, [])

    @patch("app.agents.extractor.pipeline.run_llm_extract", new_callable=AsyncMock)
    async def test_end_to_end_assembles_listing_results(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = [
            {
                "url": "https://shop.example.com/vinyl/123",
                "title": "Artist Name — Album Name vinyl LP",
                "score": 0.85,
                "price": "24.99 EUR",
                "location": None,
                "availability": "sold_out",
                "seller_type": "store",
                "match_reason": "accepted",
            }
        ]
        candidates = [
            SearchResult(
                title="Artist Name Album Name vinyl record",
                url="https://shop.example.com/vinyl/123",
                content="Artist Name Album Name official vinyl LP reissue shop.example.com",
                score=0.92,
            )
        ]
        with start_pipeline(debug=False):
            out = await extract_and_score_results(
                candidates,
                artist="Artist Name",
                album="Album Name",
                music_format="Vinyl",
                country="DE",
                city="Berlin",
            )

        self.assertEqual(len(out), 1)
        row = out[0]
        self.assertEqual(row.url, "https://shop.example.com/vinyl/123")
        self.assertEqual(row.availability, "sold_out")
        self.assertEqual(row.seller_type, "store")
        self.assertEqual(row.domain, "shop.example.com")
        self.assertAlmostEqual(row.score, 0.85, places=5)
        self.assertGreater(row.artist_match, 0.5)
        self.assertGreater(row.album_match, 0.0)

    @patch("app.agents.extractor.pipeline.run_llm_extract", new_callable=AsyncMock)
    async def test_llm_drops_url_yields_empty(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = []
        candidates = [
            SearchResult(
                title="Artist Name Album vinyl long title",
                url="https://only.this/url",
                content="Artist Name Album vinyl purchase here",
                score=0.9,
            )
        ]
        with start_pipeline(debug=False):
            out = await extract_and_score_results(
                candidates,
                artist="Artist Name",
                album="Album",
                music_format="Vinyl",
                country="FR",
                city="Paris",
            )
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
