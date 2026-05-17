"""Tests for Tavily snippet listing extraction — no real OpenAI calls.

Deterministic helpers use real RapidFuzz / regex. The LLM step is mocked at
``step_05_listings_orchestrator.llm_extract``.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from app.agents.extractor import ExtractListingsReport, extract_listings
from app.agents.extractor.evidence_alignment import (
    ascii_fold,
    evidence_blob_matches_target_release,
    url_path_evidence_text,
)
from app.agents.extractor.intent_match import intent_matches_snippet
from app.agents.extractor.listing_domains import (
    host_matches_whitelist,
    normalize_allowed_domains,
    normalize_domain,
)
from app.agents.extractor.steps.step_01_snippet_prefilter import collect_snippet_candidates
from app.agents.extractor.steps.step_02_listing_deterministic import deterministic_listings_from_candidates
from app.agents.extractor.steps.step_04_merge_llm_listings import merge_llm_rows_into_listings
from app.agents.extractor.utils.price_currency import coerce_price_currency, sniff_price_currency


class TestDomains(unittest.TestCase):
    def test_normalize_domain_strips_www_and_scheme(self) -> None:
        self.assertEqual(
            normalize_domain("https://WWW.shop.example.rs/p?q=1"),
            "shop.example.rs",
        )

    def test_normalize_domain_bare_host(self) -> None:
        self.assertEqual(normalize_domain("vinyl.example.com"), "vinyl.example.com")

    def test_normalize_domain_empty(self) -> None:
        self.assertIsNone(normalize_domain(""))
        self.assertIsNone(normalize_domain("   "))

    def test_normalize_allowed_domains_skips_empty(self) -> None:
        self.assertEqual(normalize_allowed_domains({"", "ok.example.com"}), {"ok.example.com"})

    def test_host_matches_whitelist_subdomain(self) -> None:
        allowed = {"example.com"}
        self.assertTrue(host_matches_whitelist("www.example.com", allowed))
        self.assertTrue(host_matches_whitelist("shop.example.com", allowed))
        self.assertFalse(host_matches_whitelist("other.com", allowed))


class TestPriceCurrency(unittest.TestCase):
    def test_coerce_defaults_invalid_currency_to_eur(self) -> None:
        p, c = coerce_price_currency({"price": 12.5, "currency": "XX"})
        self.assertEqual(p, 12.5)
        self.assertEqual(c, "EUR")

    def test_coerce_clamps_negative_price(self) -> None:
        p, _ = coerce_price_currency({"price": -5, "currency": "EUR"})
        self.assertEqual(p, 0.0)

    def test_sniff_eur_from_suffix(self) -> None:
        p, c = sniff_price_currency("Great LP for 29.99 EUR shipping")
        self.assertEqual(c, "EUR")
        self.assertGreater(p, 0.0)

    def test_sniff_empty_defaults(self) -> None:
        p, c = sniff_price_currency("no money here")
        self.assertEqual((p, c), (0.0, "EUR"))


class TestCollectSnippetCandidates(unittest.TestCase):
    def test_keeps_row_when_album_in_snippet_and_domain_allowed(self) -> None:
        raw = [
            {
                "url": "https://store.ok/catalog/moon",
                "title": "Pink Moon Vinyl LP",
                "content": "Nick Drake Pink Moon official reissue 24 EUR",
            }
        ]
        cands, dropped = collect_snippet_candidates(
            raw,
            artist_l="nick drake",
            album_l="pink moon",
            allowed_hosts={"store.ok"},
        )
        self.assertEqual(dropped, 0)
        self.assertEqual(len(cands), 1)
        self.assertTrue(cands[0]["url"].startswith("https://store.ok"))

    def test_drops_disallowed_domain(self) -> None:
        raw = [
            {
                "url": "https://evil.com/x",
                "title": "Pink Moon",
                "content": "Pink Moon vinyl Nick Drake",
            }
        ]
        cands, dropped = collect_snippet_candidates(
            raw,
            artist_l="nick drake",
            album_l="pink moon",
            allowed_hosts={"store.ok"},
        )
        self.assertEqual(cands, [])
        self.assertEqual(dropped, 0)


class TestDeterministicListings(unittest.TestCase):
    def test_builds_listing_from_snippet_price(self) -> None:
        candidates = [
            {
                "url": "https://a.example.com/p/1",
                "title": "Artist — Album Name LP",
                "content": "Buy now 19.50 EUR in stock",
            }
        ]
        rows = deterministic_listings_from_candidates(
            candidates,
            artist="Artist",
            album="Album Name",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].url, "https://a.example.com/p/1")
        self.assertTrue(rows[0].in_stock)
        self.assertEqual(rows[0].currency, "EUR")


class TestMergeLlmRows(unittest.TestCase):
    def test_drops_row_when_snippet_is_different_release_than_target(self) -> None:
        diagnostic: dict = {
            "drop_url_not_in_candidates": 0,
            "drop_title_gate": 0,
            "drop_pydantic": 0,
            "drop_evidence_target_miss_pdd": 0,
            "drop_llm_title_ungrounded": 0,
            "drop_query_echo_pick": 0,
        }
        candidates = [
            {
                "url": "https://groovierecords.com/p/abc",
                "title": "Classicos GNR — vinyl",
                "content": "Guns N Roses compilation remastered LP",
            },
        ]
        extracted = [
            {
                "url": "https://groovierecords.com/p/abc",
                "title": "Rammstein Mutter",
                "price": 24.0,
                "currency": "EUR",
                "in_stock": True,
                "store": "groovierecords.com",
            },
        ]
        out = merge_llm_rows_into_listings(
            extracted,
            candidates,
            artist="Rammstein",
            album="Mutter",
            artist_l="rammstein",
            album_l="mutter",
            diagnostic=diagnostic,
            snippet_relax_hosts=frozenset({"groovierecords.com"}),
        )
        self.assertEqual(out, [])
        self.assertGreater(
            diagnostic["drop_evidence_target_miss_pdd"] + diagnostic["drop_title_gate"],
            0,
        )

    def test_increments_drop_when_url_not_in_candidates(self) -> None:
        diagnostic: dict = {
            "drop_url_not_in_candidates": 0,
            "drop_title_gate": 0,
            "drop_pydantic": 0,
        }
        candidates = [
            {"url": "https://ok.com/a", "title": "Alpha Beta Album X", "content": "beta album x vinyl"},
        ]
        extracted = [
            {"url": "https://foreign.com/z", "title": "Beta Album X", "price": 1, "currency": "EUR", "in_stock": True, "store": None},
        ]
        out = merge_llm_rows_into_listings(
            extracted,
            candidates,
            artist="Alpha",
            album="Album X",
            artist_l="alpha",
            album_l="album x",
            diagnostic=diagnostic,
        )
        self.assertEqual(out, [])
        self.assertEqual(diagnostic["drop_url_not_in_candidates"], 1)


class TestExtractListingsPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_empty_input(self) -> None:
        r = await extract_listings([], artist="A", album="", allowed_domains={"x.com"})
        self.assertIsInstance(r, ExtractListingsReport)
        self.assertEqual(r.listings, [])
        self.assertEqual(r.diagnostic["empty_reason"], "no_raw_results_or_album")

    async def test_empty_allowed_domains_after_normalize(self) -> None:
        r = await extract_listings(
            [{"url": "https://a.com", "title": "t", "content": "album x"}],
            artist=None,
            album="Album X",
            allowed_domains={""},
        )
        self.assertEqual(r.diagnostic["empty_reason"], "empty_allowed_domains")

    async def test_small_batch_deterministic_no_llm(self) -> None:
        allowed = {"shop.example.org"}
        raw = [
            {
                "url": "https://shop.example.org/a",
                "title": "Gamma Band Album One LP",
                "content": "Gamma Band Album One vinyl 12.00 EUR available",
            },
            {
                "url": "https://shop.example.org/b",
                "title": "Gamma Band Album One CD",
                "content": "Gamma Band Album One 12 EUR",
            },
        ]
        r = await extract_listings(
            raw,
            artist="Gamma Band",
            album="Album One",
            allowed_domains=allowed,
        )
        self.assertEqual(r.diagnostic["extraction_mode"], "deterministic_small_batch")
        self.assertEqual(len(r.listings), 2)
        self.assertIsNone(r.diagnostic["empty_reason"])

    @patch(
        "app.agents.extractor.steps.step_05_listings_orchestrator.llm_extract",
        new_callable=AsyncMock,
    )
    async def test_llm_branch_merges_rows(self, mock_llm: AsyncMock) -> None:
        """More than ``SMALL_BATCH_NO_LLM`` candidates forces the async LLM path."""
        host = "four.example.net"
        album = "Deep Works"
        artist = "Deep Artist"
        raw = []
        for i in range(4):
            raw.append(
                {
                    "url": f"https://{host}/item/{i}",
                    "title": f"{artist} {album} vinyl LP",
                    "content": f"{artist} {album} — {18 + i}.99 EUR add to cart shop page",
                }
            )

        llm_rows = []
        for i in range(4):
            llm_rows.append(
                {
                    "url": f"https://{host}/item/{i}",
                    "title": f"{artist} — {album}",
                    "price": float(18 + i),
                    "currency": "EUR",
                    "in_stock": True,
                    "store": host,
                }
            )
        mock_llm.return_value = (llm_rows, '{"listings": [...]}')

        r = await extract_listings(
            raw,
            artist=artist,
            album=album,
            allowed_domains={host},
        )

        mock_llm.assert_awaited_once()
        self.assertEqual(r.diagnostic["extraction_mode"], "llm")
        self.assertEqual(r.diagnostic["llm_rows_returned"], 4)
        self.assertEqual(len(r.listings), 4)
        self.assertIsNone(r.diagnostic["empty_reason"])

    @patch(
        "app.agents.extractor.steps.step_05_listings_orchestrator.llm_extract",
        new_callable=AsyncMock,
    )
    async def test_llm_empty_array_sets_diagnostic(self, mock_llm: AsyncMock) -> None:
        host = "big.example.org"
        album = "Vol 9"
        artist = "Band Nine"
        raw = [
            {
                "url": f"https://{host}/p{n}",
                "title": f"{artist} {album} LP",
                "content": f"{artist} {album} euro price vinyl",
            }
            for n in range(4)
        ]
        mock_llm.return_value = ([], "{}")

        r = await extract_listings(
            raw,
            artist=artist,
            album=album,
            allowed_domains={host},
        )
        self.assertEqual(r.listings, [])
        self.assertEqual(r.diagnostic["llm_rows_returned"], 0)
        self.assertIn(r.diagnostic["empty_reason"], ("llm_empty_response", "llm_returned_empty_listings_array"))


class TestArtistArticleVariants(unittest.TestCase):
    """Leading-article bands (e.g. The Doors) often appear as \"Doors\" in snippets."""

    def test_intent_passes_when_album_present_and_artist_omits_the(self) -> None:
        blob = "strange days vinyl lp doors official reissue shop"
        self.assertTrue(
            intent_matches_snippet(
                url="https://example.com/product/strange-days",
                blob=blob,
                artist_l="the doors",
                album_l="strange days",
                relaxed_indie_candidate=False,
            )
        )

    def test_evidence_passes_for_the_doors_when_blob_says_doors(self) -> None:
        b = "doors strange days vinyl 180g reissue in stock".lower()
        self.assertTrue(
            evidence_blob_matches_target_release(
                b,
                artist="The Doors",
                album="Strange Days",
            )
        )


class TestAsciiFoldRecall(unittest.TestCase):
    """Diacritic-bearing queries (Polish, Scandinavian, Balkan…) must match
    plain-ASCII catalogue text and vice versa, language-agnostic."""

    def test_polish_l_with_stroke_folds_to_l(self) -> None:
        self.assertEqual(ascii_fold("Mgła"), "mgla")
        self.assertEqual(ascii_fold("MGŁA"), "mgla")

    def test_scandinavian_and_serbian_letters_fold(self) -> None:
        self.assertEqual(ascii_fold("Mørbid"), "morbid")
        self.assertEqual(ascii_fold("Đorđe"), "dorde")
        self.assertEqual(ascii_fold("straße"), "strasse")

    def test_evidence_matches_across_diacritic_boundary(self) -> None:
        # User typed plain ASCII artist; shop snippet uses the proper diacritic.
        diacritic_blob = "mgła – exercises in futility 12'' vinyl in stock".lower()
        self.assertTrue(
            evidence_blob_matches_target_release(
                diacritic_blob,
                artist="Mgla",
                album="Exercises in Futility",
            )
        )
        # And the reverse direction (user typed diacritic, shop wrote ASCII).
        ascii_blob = "mgla - exercises in futility lp pre-order".lower()
        self.assertTrue(
            evidence_blob_matches_target_release(
                ascii_blob,
                artist="Mgła",
                album="Exercises in Futility",
            )
        )


class TestUrlPathEvidenceText(unittest.TestCase):
    def test_decodes_and_splits_slug_separators(self) -> None:
        text = url_path_evidence_text(
            "https://store.example.pl/produkty/mgla-exercises-in-futility-12-vinyl"
        )
        self.assertIn("mgla", text)
        self.assertIn("exercises in futility", text)
        self.assertIn("vinyl", text)

    def test_percent_encoded_path_is_decoded(self) -> None:
        text = url_path_evidence_text(
            "https://store.example.pl/produkty/mg%C5%82a-exercises-in-futility"
        )
        # ``%C5%82`` is ``ł`` → folded to ``l``.
        self.assertIn("mgla", text)

    def test_empty_url_returns_empty_string(self) -> None:
        self.assertEqual(url_path_evidence_text(""), "")
        self.assertEqual(url_path_evidence_text("https://store.example.pl"), "")


class TestThinSnippetSurvivesViaUrlSlug(unittest.TestCase):
    """Indie PDP with bare-shop snippet + album-bearing URL slug should NOT be
    dropped at ``drop_title_gate`` — the regression that produced
    ``post_llm_all_dropped: drop_title_gate=15`` on real city-tier traces.

    No specific artist/shop is hard-coded; the test uses a synthetic indie
    storefront whose snippet only echoes the shop name (mirroring the real
    Tavily symptom).
    """

    def test_url_slug_carries_evidence_when_snippet_is_thin(self) -> None:
        host = "indie-shop.example.pl"
        url = f"https://{host}/produkty/mgla-exercises-in-futility-12-vinyl-lp"
        # Snippet barely has anything — just the shop's name + a vinyl token.
        thin_snippet = "INDIE SHOP — winyl, płyty, indie"
        candidates = [
            {
                "url": url,
                "title": thin_snippet,
                "content": "Indie record store. Płyty winylowe.",
            }
        ]
        llm_rows = [
            {
                "url": url,
                "title": "Mgła — Exercises in Futility 12'' Vinyl",
                "price": 119.0,
                "currency": "PLN",
                "in_stock": True,
                "store": host,
            }
        ]
        diagnostic: dict = {
            "drop_url_not_in_candidates": 0,
            "drop_title_gate": 0,
            "drop_pydantic": 0,
            "drop_evidence_target_miss_pdd": 0,
            "drop_llm_title_ungrounded": 0,
            "drop_query_echo_pick": 0,
        }
        out = merge_llm_rows_into_listings(
            llm_rows,
            candidates,
            artist="Mgła",
            album="Exercises in Futility",
            artist_l="mgła",
            album_l="exercises in futility",
            diagnostic=diagnostic,
            snippet_relax_hosts=frozenset({host}),
        )
        self.assertEqual(len(out), 1, f"expected 1 listing kept; diagnostic={diagnostic}")
        self.assertEqual(out[0].url, url)
        self.assertEqual(diagnostic["drop_title_gate"], 0)
        # Slug-derived evidence must be reported in the diagnostic for ops visibility.
        self.assertEqual(diagnostic["url_slug_evidence_used"], 1)


class TestPlaceholderDomainsDoNotPoisonTavily(unittest.TestCase):
    """``_dedupe_domains`` is the last line of defence — verify placeholder
    hosts emitted by the discovery LLM never reach ``include_domains``."""

    def test_invalid_hosts_filtered_from_dedupe(self) -> None:
        from app.services.tavily_service import _dedupe_domains  # noqa: PLC0415

        kept = _dedupe_domains([
            "groovierecords.com",
            "none",
            "unknown",
            "not provided",
            "",
            "https://www.misbits.ro/produs/x",
            "localhost",
            "foo bar.com",
        ])
        self.assertIn("groovierecords.com", kept)
        self.assertIn("misbits.ro", kept)
        for bad in ("none", "unknown", "not provided", "", "localhost", "foo bar.com"):
            self.assertNotIn(bad, kept)


if __name__ == "__main__":
    unittest.main()
