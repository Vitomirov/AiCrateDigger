"""Tests for canonical parsed-intent search cache keys."""

from __future__ import annotations

import unittest

from app.core.db.search_cache_key import (
    build_pipeline_search_cache_key,
    build_pipeline_search_cache_keys,
    build_postgres_search_cache_key,
)


class TestPipelineSearchCacheKey(unittest.TestCase):
    def test_paraphrases_share_redis_key_when_parsed_fields_match(self) -> None:
        common = {
            "format_token": "vinyl",
            "artist": "Pink Floyd",
            "album": "The Division Bell",
            "country_code": "ES",
            "resolved_city": "Barcelona",
            "geo_granularity": "city",
        }
        key_a = build_pipeline_search_cache_key(**common)
        key_b = build_pipeline_search_cache_key(
            **{
                **common,
                "artist": "  pink   floyd ",
                "album": "the division bell",
            }
        )
        self.assertEqual(key_a, key_b)
        self.assertIn(":es:barcelona", key_a)

    def test_country_only_queries_omit_city_segment(self) -> None:
        key = build_pipeline_search_cache_key(
            format_token="cd",
            artist="Radiohead",
            album="OK Computer",
            country_code="RO",
            resolved_city="Bucharest",
            geo_granularity="country",
        )
        self.assertEqual(
            key,
            "cratedigger:search:v4:cd:radiohead:ok_computer:ro",
        )

    def test_city_level_queries_include_city_segment(self) -> None:
        key = build_pipeline_search_cache_key(
            format_token="vinyl",
            artist="Radiohead",
            album="OK Computer",
            country_code="RO",
            resolved_city="Bucharest",
            geo_granularity="city",
        )
        self.assertTrue(key.endswith(":bucharest"))

    def test_different_cities_in_same_country_do_not_collide(self) -> None:
        base = {
            "format_token": "vinyl",
            "artist": "Pink Floyd",
            "album": "The Division Bell",
            "country_code": "ES",
            "geo_granularity": "city",
        }
        barcelona = build_pipeline_search_cache_key(**base, resolved_city="Barcelona")
        madrid = build_pipeline_search_cache_key(**base, resolved_city="Madrid")
        self.assertNotEqual(barcelona, madrid)

    def test_artist_catalog_uses_catalog_album_segment(self) -> None:
        key = build_pipeline_search_cache_key(
            format_token="vinyl",
            artist="Mgla",
            album="catalog",
            country_code="PL",
        )
        self.assertEqual(key, "cratedigger:search:v4:vinyl:mgla:catalog:pl")

    def test_postgres_key_is_sha256_of_redis_key(self) -> None:
        redis_key = build_pipeline_search_cache_key(
            format_token="vinyl",
            artist="Aphex Twin",
            album="Selected Ambient Works",
            country_code="GB",
        )
        pg_key = build_postgres_search_cache_key(redis_cache_key=redis_key)
        self.assertEqual(len(pg_key), 64)
        redis_key_2, pg_key_2 = build_pipeline_search_cache_keys(
            format_token="vinyl",
            artist="Aphex Twin",
            album="Selected Ambient Works",
            country_code="GB",
        )
        self.assertEqual(redis_key, redis_key_2)
        self.assertEqual(pg_key, pg_key_2)


if __name__ == "__main__":
    unittest.main()
