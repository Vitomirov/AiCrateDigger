"""HTTP end-to-end edge cases against the real FastAPI app.

External calls (OpenAI, Tavily, Postgres, Redis) are not exercised: lifespan skips
database work when ``DATABASE_URL`` is empty, and parser/search paths that would
hit the LLM pipeline are patched with ``AsyncMock``.

Run from ``backend``::

    poetry run python -m unittest tests.test_app_http_e2e -v
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import AsyncMock, patch

# -----------------------------------------------------------------------------
# Bootstrap env BEFORE importing Settings / app.main (required + infra opt-out).
# -----------------------------------------------------------------------------
os.environ.setdefault("OPENAI_API_KEY", "sk-e2e-dummy-unused")
os.environ.setdefault("TAVILY_API_KEY", "tv-e2e-dummy-unused")
os.environ["DATABASE_URL"] = ""
os.environ["REDIS_URL"] = ""
os.environ["DEBUG"] = "false"
os.environ["SEARCH_RATE_LIMIT_ENABLED"] = "false"
os.environ["INTERNAL_API_SECRET"] = ""

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from fastapi.testclient import TestClient  # noqa: E402

from app.domains.query_parser.parse_schema import ParsedQuery as StrictParsedQuery  # noqa: E402
from app.domains.search_pipeline.models.result import ListingResult  # noqa: E402
from app.main import app  # noqa: E402


class TestHTTPAppEdgeCases(unittest.TestCase):
    """Ten explicit boundary checks on routing, validation, and error mapping."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SEARCH_RATE_LIMIT_ENABLED"] = "false"
        os.environ["INTERNAL_API_SECRET"] = ""
        get_settings.cache_clear()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def test_health_returns_ok_payload(self) -> None:
        """Liveness probe: stable JSON contract for gateways / Compose."""
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["service"], "backend")
        self.assertIn("database_configured", body)
        self.assertFalse(body["database_configured"])

    def test_openapi_schema_exposed(self) -> None:
        """Regression: Swagger stack and codegen consumers expect bundled OpenAPI."""
        r = self.client.get("/openapi.json")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        # OpenAPI 3.0 and 3.1 both use a "3.x" version string.
        self.assertTrue(body.get("openapi").startswith("3."))
        paths = body.get("paths") or {}
        self.assertIn("/health", paths)
        self.assertIn("/search", paths)
        self.assertIn("/parse", paths)

    def test_parse_rejects_missing_query_field_with_422(self) -> None:
        """Request bodies must carry the singular ``query`` string field."""
        r = self.client.post("/parse", json={})
        self.assertEqual(r.status_code, 422)

    def test_parse_rejects_empty_query_string_with_422(self) -> None:
        """``query`` honours ``min_length=1`` — empty vinyl text is invalid."""
        r = self.client.post("/parse", json={"query": ""})
        self.assertEqual(r.status_code, 422)

    def test_parse_rejects_query_over_max_length_with_422(self) -> None:
        """Oversized prompts are rejected before any OpenAI call."""
        max_len = get_settings().search_query_max_length
        r = self.client.post("/parse", json={"query": "x" * (max_len + 1)})
        self.assertEqual(r.status_code, 422)

    def test_parse_rejects_wrong_query_type_with_422(self) -> None:
        """Structured field must remain a JSON string — no silent coercion."""
        r = self.client.post("/parse", json={"query": 12345})
        self.assertEqual(r.status_code, 422)

    def test_parse_rejects_malformed_json_body(self) -> None:
        """Garbage payload cannot be decoded as JSON (client error class)."""
        r = self.client.post(
            "/parse",
            content="not-valid-json-{",
            headers={"Content-Type": "application/json"},
        )
        self.assertIn(r.status_code, {400, 422})

    def test_parse_returns_strict_parsed_query_when_llm_stubbed_unicode(self) -> None:
        """Happy path wiring: Cyrillic artwork + mocked parser survives response_model."""
        q = 'Molchat Doma "Этажи" vinyl in Warszawa'
        stub = StrictParsedQuery(
            artist="Molchat Doma",
            album="Этажи",
            location="Warszawa",
            language="unknown",
            country_code="PL",
            search_scope="local",
            original_query=q,
        )
        with patch(
            "app.api.routers.search.parse_user_query",
            new_callable=AsyncMock,
            return_value=stub,
        ):
            r = self.client.post("/parse", json={"query": q})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["artist"], "Molchat Doma")
        self.assertEqual(data["album"], "Этажи")
        self.assertEqual(data["original_query"], q)

    def test_parse_maps_parser_exception_to_502(self) -> None:
        """Operator-facing failure surface when structured parsing blows up upstream."""
        with patch(
            "app.api.routers.search.parse_user_query",
            new_callable=AsyncMock,
            side_effect=RuntimeError("upstream parser unavailable"),
        ):
            r = self.client.post("/parse", json={"query": "anything"})
        self.assertEqual(r.status_code, 502)
        self.assertEqual(
            r.json().get("detail"),
            "Parse could not be completed. Please try again later.",
        )

    def test_search_structured_empty_when_album_unresolved_stubbed(self) -> None:
        """Pipeline short-circuit: zero Tavily listings + machine-readable ``reason``.

        ``SearchResponse.parsed`` expects the **strict** ``query_parser.parse_schema``
        model (same shape as ``/parse``), not the permissive DTO in ``search_query``.
        """
        parsed = StrictParsedQuery(
            artist="Someone",
            album=None,
            album_index=None,
            language="en",
            original_query="Someone vinyl",
            search_scope="global",
        )
        stub_payload = {"results": [], "parsed": parsed, "reason": "album_unresolved"}
        with patch(
            "app.api.routers.search.run_vinyl_search",
            new_callable=AsyncMock,
            return_value=stub_payload,
        ):
            r = self.client.post("/search", json={"query": "Someone vinyl"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["results"], [])
        self.assertEqual(data["reason"], "album_unresolved")
        self.assertEqual(data["parsed"]["artist"], "Someone")

    def test_search_listings_is_alias_of_search_under_same_stub(self) -> None:
        """Legacy route name stays wire-compatible — byte-for-byte parity on stubbed path."""
        listing = ListingResult(
            url="https://groovierecordshop.example/catalog/pink-floyd-wish-you-were-here-vinyl",
            title="Pink Floyd — Wish You Were Here (LP mint)",
            score=0.88,
            price="$42.00",
        )
        parsed = StrictParsedQuery(
            artist="Pink Floyd",
            album="Wish You Were Here",
            album_index=None,
            location="London",
            language="en",
            original_query="Wish you were here vinyl London",
            search_scope="local",
            country_code="GB",
        )
        stub_payload = {
            "results": [listing.model_dump(mode="json")],
            "parsed": parsed,
            "reason": None,
        }
        with patch(
            "app.api.routers.search.run_vinyl_search",
            new_callable=AsyncMock,
            return_value=stub_payload,
        ):
            primary = self.client.post("/search", json={"query": parsed.original_query})
            legacy = self.client.post("/search-listings", json={"query": parsed.original_query})
        self.assertEqual(primary.status_code, 200)
        self.assertEqual(legacy.status_code, 200)

        left, right = primary.json(), legacy.json()

        normalised_left = json.dumps(left, sort_keys=True)
        normalised_right = json.dumps(right, sort_keys=True)
        self.assertEqual(normalised_left, normalised_right)


if __name__ == "__main__":
    unittest.main()