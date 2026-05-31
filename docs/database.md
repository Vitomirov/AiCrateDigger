# Database

AiCrateDigger uses **PostgreSQL 15** for persistent data and **Redis 7** for hot caching, rate limiting, and quota counters. Both are optional in local development but **required in production** (enforced by startup guard).

---

## PostgreSQL

### Connection

**Driver:** SQLAlchemy 2.0 async with `asyncpg`

**Configuration:**

| Method | Example |
|--------|---------|
| Full DSN (preferred) | `DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/aicratedigger` |
| Component vars | `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`, `POSTGRES_PORT` |

Resolved in `Settings.resolved_database_url` (`app/core/config.py`).

**Session factory:** `session_factory()` in `app/core/db/database.py` returns `async_sessionmaker[AsyncSession]`.

**When unset:** DB layer no-ops; store catalogue falls back to in-code policy; Postgres cache writes are skipped.

---

## Schema management

**No Alembic migrations.**

Schema lifecycle:

1. **`init_db()`** — `Base.metadata.create_all` on startup
2. **Inline `ALTER TABLE`** — additive columns and legacy column drops on startup
3. **Code seed** — `seed_whitelist_stores_if_empty()` from `ALLOWED_STORES`
4. **Catalogue sync** — `sync_whitelist_store_catalogue()`, `repair_whitelist_store_domains()`

This approach suits portfolio-stage iteration but is not recommended for production schema evolution at scale. Plan Alembic adoption before multi-environment deployments with divergent schemas.

---

## Tables

### `whitelist_stores`

Curated commerce domains used for Tavily prefilter signals and store discovery.

**ORM:** `WhitelistStoreORM` in `app/core/db/database.py`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | Integer | PK, autoincrement | |
| `name` | String(128) | | Display name |
| `domain` | String(160) | unique, indexed | Normalized host |
| `country` | String(8) | | Country label |
| `country_code` | String(2) | nullable | ISO-3166-1 alpha-2 |
| `region` | String(32) | nullable | Region label |
| `ships_to_json` | Text | default `'["EU"]'` | JSON array of ship-to regions |
| `priority` | Integer | default 5 | Sort/ranking weight |
| `is_active` | Boolean | default true | Soft disable |
| `city` | String(128) | nullable | Local shop city |
| `store_type` | String(32) | nullable | `local_shop`, `regional_ecommerce`, `marketplace` |

**Lifecycle:**

| Operation | Function | When |
|-----------|----------|------|
| Seed | `seed_whitelist_stores_if_empty()` | Startup, if table empty |
| Sync | `sync_whitelist_store_catalogue()` | Startup |
| Repair | `repair_whitelist_store_domains()` | Startup |
| Load | `load_active_stores()` | Each search request |

**Source of truth for seed:** `app/domains/engine/policies/eu_stores.py` → `ALLOWED_STORES`

**Discovery upserts:** `store_discovery.py` can insert new domains when Tavily surfaces unknown shop hosts (subject to LLM vetting).

---

### `search_response_cache`

Postgres-backed search response cache for audit and Redis miss fallback.

**ORM:** `SearchResponseCacheORM`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `cache_key` | String(64) | PK | SHA-256 hex of Redis key |
| `payload_json` | Text | | JSON-serialized search response |
| `created_at` | DateTime(tz) | server default now | |
| `expires_at` | DateTime(tz) | | TTL expiry |

**Module:** `app/core/db/cache.py`

| Function | Purpose |
|----------|---------|
| `get_cached_search_payload(key)` | Read by Postgres key |
| `set_cached_search_payload(key, payload, ttl)` | Write with expiry |
| `purge_expired_search_cache_rows()` | Startup cleanup |

**Read behavior:** Skipped when `DEBUG=true` (reads only; writes may still occur for operator audit).

---

## Redis

### Connection

**URL:** `REDIS_URL=redis://redis:6379/0`

**Client:** Async Redis client in `app/core/db/redis_cache.py`; disposed on app shutdown.

**When unset:** Redis cache, rate limiting, and quotas no-op or fail-closed depending on settings.

---

## Redis key namespaces

| Prefix / pattern | Module | Purpose | TTL |
|------------------|--------|---------|-----|
| `cratedigger:search:v3:...` | `redis_cache.py` | Search response cache | 7 days (configurable) |
| `rate_limit:api:{ip}` | `rate_limiter.py` | Per-IP sliding window | Window seconds |
| `quota:{kind}:{YYYY-MM-DD}` | `quota/service.py` | Daily provider caps | Until UTC midnight |

---

## Search cache key design

**Module:** `app/core/db/search_cache_key.py`

**Schema version:** `PIPELINE_CACHE_SCHEMA_VERSION = 3`

**Redis key format:**

```
cratedigger:search:v{version}:{format}:{artist}:{album}:{country}[:{city}]
```

Components are normalized (lowercase, trimmed) to improve hit rate on paraphrased queries.

**Postgres key:** SHA-256 hex digest of the Redis key via `build_postgres_search_cache_key()`.

**Builder:** `build_pipeline_search_cache_keys(parsed)` returns both keys from a `ParsedQuery`.

### Cache behavior

| Event | Redis | Postgres |
|-------|-------|----------|
| Cache hit | Return payload | Fallback if Redis miss |
| Cache miss | — | — |
| Pipeline success | Write with TTL | Write with expiry |
| Pipeline empty/failed | No write | No write |
| `DEBUG=true` | Skip read/write | Skip read only |
| Schema version change | `purge_stale_pipeline_cache_versions()` on startup | Old rows expire naturally |

**TTL:** `REDIS_SEARCH_CACHE_TTL_SECONDS` (default 604800 = 7 days)

**Toggle:** `SEARCH_CACHE_ENABLED` (default true)

---

## Entity relationship diagram

```mermaid
erDiagram
  whitelist_stores {
    int id PK
    string name
    string domain UK
    string country_code
    string city
    boolean is_active
  }

  search_response_cache {
    string cache_key PK
    text payload_json
    datetime created_at
    datetime expires_at
  }
```

No foreign keys between tables. Store catalogue and search cache are independent.

---

## Docker volumes

| Volume | Service | Mount |
|--------|---------|-------|
| `pgdata` | db | `/var/lib/postgresql/data` |
| `redisdata` | redis | `/data` |

Production Compose uses the same named volumes with `restart: unless-stopped`. No host port publishes for db/redis in production.

---

## Local access

Development Compose publishes:

| Service | Host port | Container port |
|---------|-----------|----------------|
| Postgres | `${POSTGRES_PORT:-5433}` | 5432 |
| Redis | `${REDIS_PORT:-6379}` | 6379 |

Connect from host for debugging:

```bash
psql postgresql://aicratedigger:aicratedigger@localhost:5433/aicratedigger
redis-cli -p 6379
```

---

## Operational queries

### Active store count

```sql
SELECT COUNT(*) FROM whitelist_stores WHERE is_active = true;
```

### Cache row count and oldest entry

```sql
SELECT COUNT(*), MIN(created_at), MAX(created_at)
FROM search_response_cache
WHERE expires_at > NOW();
```

### Stores by country

```sql
SELECT country_code, COUNT(*)
FROM whitelist_stores
WHERE is_active = true
GROUP BY country_code
ORDER BY COUNT(*) DESC;
```

### Purge expired cache (also runs on startup)

Handled by `purge_expired_search_cache_rows()` — manual purge not typically needed.

---

## Data retention

| Data | Retention |
|------|-----------|
| Search cache (Redis) | TTL-based (7 days default) |
| Search cache (Postgres) | `expires_at` column; purged on startup |
| Whitelist stores | Indefinite; soft-delete via `is_active` |
| Rate limit keys | Window TTL |
| Quota counters | UTC day boundary |

No user accounts or PII storage. Queries pass through the pipeline but are not persisted as user history (only cache keys derived from normalized parse fields).

---

## Failure modes

| Condition | Pipeline behavior |
|-----------|-------------------|
| Postgres down, Redis up | Stores from code fallback; cache from Redis; Postgres writes fail silently |
| Redis down, Postgres up | Rate limit/quota fail-closed (503) if configured; cache from Postgres only |
| Both down | Stores from code; no cache; rate limit/quota fail-closed |
| `DATABASE_URL` unset | Full no-op DB layer; dev-friendly |

See [Operations](./operations.md) for troubleshooting.
