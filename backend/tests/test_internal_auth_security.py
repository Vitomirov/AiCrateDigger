"""Phase 2: BFF secret and trusted client IP for rate limiting."""

from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
os.environ.setdefault("TAVILY_API_KEY", "tv-test-dummy")
os.environ["DATABASE_URL"] = ""
os.environ["REDIS_URL"] = "redis://127.0.0.1:6379/0"
os.environ["DEBUG"] = "false"
os.environ["SEARCH_RATE_LIMIT_ENABLED"] = "false"
os.environ["SEARCH_RATE_LIMIT_FAIL_CLOSED"] = "false"
os.environ["SEARCH_RATE_LIMIT_MAX_REQUESTS"] = "10"
os.environ["INTERNAL_API_SECRET"] = "test-secret-value"

from app.core.config import get_settings  # noqa: E402
from app.core.client_ip import resolve_client_ip  # noqa: E402
from app.core.internal_auth import INTERNAL_API_SECRET_HEADER  # noqa: E402

get_settings.cache_clear()

from fastapi.testclient import TestClient  # noqa: E402

from app.domains.query_parser.parse_schema import ParsedQuery as StrictParsedQuery  # noqa: E402
from app.main import app  # noqa: E402


def _auth_headers(*, secret: str | None = None, forwarded_for: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if secret is not None:
        headers[INTERNAL_API_SECRET_HEADER] = secret
    if forwarded_for is not None:
        headers["X-Forwarded-For"] = forwarded_for
    return headers


class TestInternalAuthSecurity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SEARCH_RATE_LIMIT_ENABLED"] = "false"
        os.environ["INTERNAL_API_SECRET"] = "test-secret-value"
        get_settings.cache_clear()
        cls.client = TestClient(app)
        cls.stub = StrictParsedQuery(
            artist="Test",
            album="Album",
            language="en",
            original_query="Test Album vinyl",
            search_scope="global",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def test_paid_route_rejects_missing_secret(self) -> None:
        r = self.client.post("/search", json={"query": "Test Album vinyl"})
        self.assertEqual(r.status_code, 401)

    def test_paid_route_rejects_wrong_secret(self) -> None:
        r = self.client.post(
            "/search",
            json={"query": "Test Album vinyl"},
            headers=_auth_headers(secret="wrong"),
        )
        self.assertEqual(r.status_code, 401)

    def test_paid_route_accepts_valid_secret_when_stubbed(self) -> None:
        with patch(
            "app.api.routers.search.run_vinyl_search",
            new_callable=AsyncMock,
            return_value={"results": [], "parsed": self.stub, "reason": None},
        ):
            r = self.client.post(
                "/search",
                json={"query": "Test Album vinyl"},
                headers=_auth_headers(secret="test-secret-value"),
            )
        self.assertEqual(r.status_code, 200)

    def test_health_stays_public_without_secret(self) -> None:
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)

    def test_resolve_client_ip_honors_xff_with_valid_secret(self) -> None:
        request = MagicMock()

        def _header_get(name: str) -> str | None:
            return {
                INTERNAL_API_SECRET_HEADER: "test-secret-value",
                "X-Forwarded-For": "203.0.113.50",
                "x-forwarded-for": "203.0.113.50",
            }.get(name)

        request.headers.get.side_effect = _header_get
        request.client.host = "198.51.100.9"
        self.assertEqual(resolve_client_ip(request), "203.0.113.50")

    def test_resolve_client_ip_ignores_xff_without_secret(self) -> None:
        request = MagicMock()
        request.headers.get.side_effect = lambda name: {
            "X-Forwarded-For": "203.0.113.50",
        }.get(name)
        request.client.host = "198.51.100.9"
        self.assertEqual(resolve_client_ip(request), "198.51.100.9")

    def test_rate_limit_uses_xff_when_bff_secret_valid(self) -> None:
        os.environ["SEARCH_RATE_LIMIT_ENABLED"] = "true"
        os.environ["SEARCH_RATE_LIMIT_FAIL_CLOSED"] = "false"
        get_settings.cache_clear()

        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_pipe.zremrangebyscore.return_value = mock_pipe
        mock_pipe.zcard.return_value = mock_pipe
        mock_pipe.zadd.return_value = mock_pipe
        mock_pipe.expire.return_value = mock_pipe
        captured_keys: list[str] = []

        async def fake_execute() -> list[object]:
            return [None, 0]

        mock_pipe.execute = fake_execute
        mock_pipe.zadd.side_effect = lambda key, mapping: captured_keys.append(key) or mock_pipe

        with (
            patch(
                "app.core.rate_limiter.get_redis_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
            patch(
                "app.api.routers.search.run_vinyl_search",
                new_callable=AsyncMock,
                return_value={"results": [], "parsed": self.stub, "reason": None},
            ),
        ):
            r = self.client.post(
                "/search",
                json={"query": "Test Album vinyl"},
                headers=_auth_headers(secret="test-secret-value", forwarded_for="203.0.113.50"),
            )

        self.assertEqual(r.status_code, 200)
        self.assertTrue(captured_keys)
        self.assertIn("203.0.113.50", captured_keys[0])

        os.environ["SEARCH_RATE_LIMIT_ENABLED"] = "false"
        get_settings.cache_clear()


if __name__ == "__main__":
    unittest.main()
