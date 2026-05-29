"""Phase 3: global daily quota counters."""

from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
os.environ.setdefault("TAVILY_API_KEY", "tv-test-dummy")
os.environ["DATABASE_URL"] = ""
os.environ["REDIS_URL"] = "redis://127.0.0.1:6379/0"
os.environ["DEBUG"] = "false"
os.environ["GLOBAL_DAILY_QUOTA_ENABLED"] = "true"
os.environ["GLOBAL_DAILY_QUOTA_FAIL_CLOSED"] = "true"
os.environ["GLOBAL_DAILY_QUOTA_PARSE_MAX"] = "2"
os.environ["GLOBAL_DAILY_QUOTA_TAVILY_MAX"] = "0"
os.environ["GLOBAL_DAILY_QUOTA_OPENAI_EXTRACT_MAX"] = "0"

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.quota import (  # noqa: E402
    QuotaExceededError,
    QuotaKind,
    QuotaUnavailableError,
    assert_quota_available,
    record_quota_usage,
)
from app.core.quota.limits import daily_limit_for_kind  # noqa: E402


class TestGlobalQuotaService(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        os.environ["GLOBAL_DAILY_QUOTA_PARSE_MAX"] = "2"
        os.environ["GLOBAL_DAILY_QUOTA_TAVILY_MAX"] = "0"
        get_settings.cache_clear()

    async def test_parse_quota_blocks_when_at_cap(self) -> None:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=b"2")

        with patch(
            "app.core.quota.service.get_redis_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            with self.assertRaises(QuotaExceededError) as ctx:
                await assert_quota_available(QuotaKind.PARSE)

        self.assertEqual(ctx.exception.limit, 2)
        self.assertEqual(ctx.exception.kind, QuotaKind.PARSE)

    async def test_record_increments_redis_key(self) -> None:
        mock_client = AsyncMock()
        mock_client.incrby = AsyncMock(return_value=1)

        with patch(
            "app.core.quota.service.get_redis_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            await record_quota_usage(QuotaKind.PARSE)

        mock_client.incrby.assert_awaited_once()
        key = mock_client.incrby.await_args.args[0]
        self.assertTrue(key.startswith("quota:parse:"))
        mock_client.expire.assert_awaited_once()

    async def test_unlimited_bucket_skips_redis(self) -> None:
        settings = get_settings()
        self.assertEqual(daily_limit_for_kind(settings, QuotaKind.TAVILY), 0)

        with patch(
            "app.core.quota.service.get_redis_client",
            new_callable=AsyncMock,
        ) as mock_redis:
            await assert_quota_available(QuotaKind.TAVILY)
            mock_redis.assert_not_awaited()

    async def test_fail_closed_when_redis_missing(self) -> None:
        with patch(
            "app.core.quota.service.get_redis_client",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with self.assertRaises(QuotaUnavailableError):
                await assert_quota_available(QuotaKind.PARSE)


class TestGlobalQuotaHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SEARCH_RATE_LIMIT_ENABLED"] = "false"
        os.environ["INTERNAL_API_SECRET"] = ""
        os.environ["GLOBAL_DAILY_QUOTA_ENABLED"] = "true"
        os.environ["GLOBAL_DAILY_QUOTA_FAIL_CLOSED"] = "true"
        os.environ["GLOBAL_DAILY_QUOTA_PARSE_MAX"] = "1"
        get_settings.cache_clear()

        from fastapi.testclient import TestClient  # noqa: E402

        from app.main import app  # noqa: E402

        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def test_search_returns_503_when_parse_quota_exceeded(self) -> None:
        """Parse runs before cache; quota blocks before OpenAI or Tavily."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=b"1")

        with patch(
            "app.core.quota.service.get_redis_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            r = self.client.post("/search", json={"query": "Tool Lateralus vinyl"})

        self.assertEqual(r.status_code, 503)
        self.assertIn("Daily usage limit", r.json().get("detail", ""))


if __name__ == "__main__":
    unittest.main()
