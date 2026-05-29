"""Rate-limit fail-closed and shared-bucket behaviour (no live Redis required)."""

from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
os.environ.setdefault("TAVILY_API_KEY", "tv-test-dummy")
os.environ["DATABASE_URL"] = ""
os.environ["REDIS_URL"] = "redis://127.0.0.1:6379/0"
os.environ["DEBUG"] = "false"
os.environ["SEARCH_RATE_LIMIT_ENABLED"] = "true"
os.environ["SEARCH_RATE_LIMIT_FAIL_CLOSED"] = "true"
os.environ["SEARCH_RATE_LIMIT_MAX_REQUESTS"] = "2"
os.environ["SEARCH_RATE_LIMIT_WINDOW_SECONDS"] = "3600"

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from fastapi.testclient import TestClient  # noqa: E402

from app.domains.query_parser.parse_schema import ParsedQuery as StrictParsedQuery  # noqa: E402
from app.main import app  # noqa: E402


class TestRateLimitSecurity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SEARCH_RATE_LIMIT_ENABLED"] = "true"
        os.environ["SEARCH_RATE_LIMIT_FAIL_CLOSED"] = "true"
        os.environ["SEARCH_RATE_LIMIT_MAX_REQUESTS"] = "2"
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

    def test_redis_unavailable_fail_closed_returns_503_on_search(self) -> None:
        with patch(
            "app.core.rate_limiter.get_redis_client",
            new_callable=AsyncMock,
            return_value=None,
        ):
            r = self.client.post("/search", json={"query": "Test Album vinyl"})
        self.assertEqual(r.status_code, 503)
        self.assertIn("temporarily unavailable", r.json().get("detail", "").lower())

    def test_redis_unavailable_fail_closed_returns_503_on_parse(self) -> None:
        with patch(
            "app.core.rate_limiter.get_redis_client",
            new_callable=AsyncMock,
            return_value=None,
        ):
            r = self.client.post("/parse", json={"query": "Test Album vinyl"})
        self.assertEqual(r.status_code, 503)

    def test_parse_and_search_share_rate_limit_bucket(self) -> None:
        """Third paid request across /parse and /search returns 429."""
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_pipe.zremrangebyscore.return_value = mock_pipe
        mock_pipe.zcard.return_value = mock_pipe
        mock_pipe.zadd.return_value = mock_pipe
        mock_pipe.expire.return_value = mock_pipe
        # Each allowed request: one read pipeline (zcard) then one write pipeline.
        mock_pipe.execute = AsyncMock(
            side_effect=[
                [None, 0],
                [],
                [None, 1],
                [],
                [None, 2],
            ]
        )

        with (
            patch(
                "app.core.rate_limiter.get_redis_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
            patch(
                "app.api.routers.search.parse_user_query",
                new_callable=AsyncMock,
                return_value=self.stub,
            ),
            patch(
                "app.api.routers.search.run_vinyl_search",
                new_callable=AsyncMock,
                return_value={"results": [], "parsed": self.stub, "reason": None},
            ),
        ):
            first = self.client.post("/parse", json={"query": "one"})
            second = self.client.post("/search", json={"query": "two"})
            third = self.client.post("/parse", json={"query": "three"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(third.status_code, 429)


if __name__ == "__main__":
    unittest.main()
