# Operations

Guidance for running, monitoring, and troubleshooting AiCrateDigger in development and production.

**Production instance:** [https://aicratedigger.dejanvitomirov.com/](https://aicratedigger.dejanvitomirov.com/) — verify the home page shows “Backend online” and search returns results or explicit empty states.

---

## Health checks

### Backend

```
GET /health
```

**Response:**

```json
{
  "status": "ok",
  "service": "backend",
  "database_configured": true
}
```

- Does not verify Postgres connectivity (only whether DSN is configured)
- Does not verify Redis, OpenAI, or Tavily availability
- No authentication required

### Frontend

The home page (`app/page.tsx`) calls `fetchHealth()` server-side and exposes status in an `sr-only` `aria-live` region. No dedicated `/health` route on the frontend.

### Docker Compose healthchecks

| Service | Check |
|---------|-------|
| `db` | `pg_isready` |
| `redis` | `redis-cli ping` |
| `backend`, `frontend` | `depends_on` only (no HTTP healthcheck) |

---

## Logging

### Backend

**Module:** `app/core/logging_config.py`

| Variable | Values | Recommendation |
|----------|--------|----------------|
| `LOG_LEVEL` | DEBUG, INFO, WARNING, ERROR | INFO in production |
| `LOG_FORMAT` | `human`, `json` | `json` in production |

Structured logging uses `extra={}` dicts on log records. Example fields:

- `cache_key`, `path`, `required`, `stage`, `host`
- Database URLs masked via `mask_database_url()`

### Viewing logs

```bash
# Development
docker compose logs -f backend
docker compose logs -f frontend

# Production
docker compose -f docker-compose.prod.yml logs -f backend
docker compose -f docker-compose.prod.yml logs -f frontend
```

### Log events to monitor

| Event / pattern | Meaning |
|-----------------|---------|
| Startup guard failure | Misconfigured production env |
| `rate_limit_exceeded` | Client hit IP limit |
| `quota_exceeded` | Daily provider cap reached |
| `tavily_circuit_breaker` | Tavily throttling; request failed fast |
| `cache_hit` / `cache_miss` | Cache behavior |
| 502 in access logs | Pipeline or provider failure |

---

## Observability gaps

The following are **not** implemented in the current codebase:

| Capability | Status |
|------------|--------|
| Prometheus metrics | Not configured |
| OpenTelemetry tracing | Not configured |
| APM (Datadog, Sentry) | Not configured |
| Centralized log aggregation | Deployer responsibility |
| Uptime monitoring | Deployer responsibility |

Recommended additions for production:

- JSON log shipping to CloudWatch, Loki, or ELK
- Alert on 503 rate spike (Redis down or quota exhaustion)
- Alert on 502 rate spike (provider failures)
- External uptime check on frontend URL

---

## Debug mode

Enable pipeline stage traces in API responses:

```env
DEBUG=true
```

**Effects:**

- `SearchResponse.debug` includes full `PipelineContext` trace
- Redis cache reads bypassed (writes may still go to Postgres for audit)
- Frontend dev inspector shows parse, pipeline stages, and results JSON

**Never enable in production.** Blocked by production startup guard.

### Frontend dev inspector

Visible when:

- `NODE_ENV !== "production"`, or
- `NEXT_PUBLIC_DEV_INSPECTOR=true`

Shows three columns: Parse JSON, Pipeline stages, Listings JSON.

---

## Cost controls

AiCrateDigger incurs costs on three provider actions per search:

| Action | Provider | Typical frequency |
|--------|----------|-------------------|
| Parse | OpenAI | 1× per request |
| Search | Tavily | 0–1× (skipped on album_unresolved) |
| Extract | OpenAI | 0–N× (N = prefilter candidates, capped) |

### Built-in controls

| Control | Default | Config |
|---------|---------|--------|
| Redis search cache | 7-day TTL | `REDIS_SEARCH_CACHE_TTL_SECONDS` |
| Per-IP rate limit | 5/day | `SEARCH_RATE_LIMIT_*` |
| Daily parse quota | 500 | `GLOBAL_DAILY_QUOTA_PARSE_MAX` |
| Daily Tavily quota | 200 | `GLOBAL_DAILY_QUOTA_TAVILY_MAX` |
| Daily extract quota | 300 | `GLOBAL_DAILY_QUOTA_OPENAI_EXTRACT_MAX` |
| Prefilter cap | 6 candidates | `PIPELINE_PREFILTER_MAX_CANDIDATES` |
| Result cap | 4 listings | `PIPELINE_MAX_RESULTS` |
| Tavily circuit breaker | Fail fast on throttle | `TAVILY_CIRCUIT_BREAKER_FAILURE_THRESHOLD` |

### Monitoring spend

Check Redis quota keys:

```bash
redis-cli KEYS "quota:*"
redis-cli GET "quota:parse:2026-05-31"
redis-cli GET "quota:tavily:2026-05-31"
redis-cli GET "quota:openai_extract:2026-05-31"
```

Counters reset at UTC midnight.

---

## Troubleshooting

### Search returns empty results (no error)

| Cause | Check |
|-------|-------|
| Album not resolved | Response `reason: "album_unresolved"` |
| Tavily returned no relevant snippets | Enable DEBUG; inspect `tavily` stage |
| Prefilter removed all candidates | DEBUG trace `prefilter` stage |
| Extract found no valid listings | DEBUG trace `extract` stage |
| Cache hit with stale empty result | Set `DEBUG=true` or purge cache |

### HTTP 502 Bad Gateway

| Layer | Cause |
|-------|-------|
| BFF | Backend unreachable; wrong `BACKEND_URL` |
| Backend | OpenAI/Tavily failure; unhandled pipeline exception |

Check backend logs for the specific exception. Verify API keys.

### HTTP 503 Service Unavailable

| Cause | Resolution |
|-------|------------|
| Rate limit fail-closed | Redis down or unreachable |
| Quota exceeded | Wait for UTC reset or raise caps |
| Quota fail-closed | Redis down |

```bash
docker compose ps redis
docker compose exec redis redis-cli ping
```

### HTTP 401 Unauthorized

Direct backend call without valid `X-Internal-Api-Secret` when secret is configured.

### HTTP 429 Too Many Requests

Client exceeded per-IP rate limit. Default: 5 requests per 24 hours per IP.

### Startup failure: RuntimeError (production guard)

Read the error message. Common fixes:

- Set `INTERNAL_API_SECRET`
- Set `DATABASE_URL` and `REDIS_URL`
- Set `DEBUG=false`
- Enable rate limit and quota with fail-closed

### Frontend: "Could not reach backend"

- Verify backend container is running: `docker compose ps backend`
- Verify `BACKEND_URL=http://backend:8000` in frontend env
- Check Docker network connectivity

### Postgres connection errors

- Verify `DATABASE_URL` uses `@db:5432` inside Compose (not `localhost`)
- Wait for db healthcheck: `docker compose ps db`
- Host dev: use `@localhost:5433`

### Store catalogue empty

- Check `whitelist_stores` row count
- Restart backend to trigger seed/sync
- Verify `DATABASE_URL` is configured

---

## Cache management

### Purge stale Redis schema versions

Automatic on backend startup via `purge_stale_pipeline_cache_versions()`.

### Purge expired Postgres cache

Automatic on startup via `purge_expired_search_cache_rows()`.

### Manual Redis flush (development only)

```bash
docker compose exec redis redis-cli FLUSHDB
```

**Warning:** Also clears rate limits and quota counters.

### Invalidate specific search

Cache keys follow pattern:

```
cratedigger:search:v3:{format}:{artist}:{album}:{country}[:{city}]
```

Use `DEBUG=true` for one-off bypass without key deletion.

---

## Database operations

### Connect to Postgres (dev)

```bash
docker compose exec db psql -U aicratedigger -d aicratedigger
```

### Useful queries

```sql
-- Active stores
SELECT domain, country_code, city, store_type
FROM whitelist_stores WHERE is_active = true
ORDER BY priority DESC LIMIT 20;

-- Cache stats
SELECT COUNT(*), MIN(created_at), MAX(expires_at)
FROM search_response_cache;
```

### Backup

```bash
docker compose exec db pg_dump -U aicratedigger aicratedigger > backup.sql
```

See [Database](./database.md).

---

## Service restart

```bash
# Development
docker compose restart backend frontend

# Production
docker compose -f docker-compose.prod.yml restart backend frontend
```

Backend startup runs: DB init, store seed/sync, cache purge, production guard validation.

---

## Performance expectations

Typical full search latency (cache miss):

| Stage | Approximate time |
|-------|------------------|
| Parse (OpenAI) | 1–3 s |
| Tavily search | 2–8 s |
| Extract (OpenAI) | 1–5 s per candidate |
| **Total** | 5–20 s typical |

Cache hit: sub-second (Redis) or low seconds (Postgres fallback).

The frontend shows an animated progress bar during wait — not SSE streaming.

---

## Maintenance tasks

| Task | Frequency | Command / action |
|------|-----------|------------------|
| Review quota usage | Daily (if public) | Redis `quota:*` keys |
| Purge expired cache | Automatic | Backend startup |
| Sync store catalogue | Automatic | Backend startup |
| Run eval harness | After pipeline changes | `docker compose --profile eval run --rm eval` |
| Run unit tests | Before deploy | `poetry run python -m unittest discover -s tests -v` |
| Rotate secrets | Periodic | Update `.env`, restart services |
| Postgres backup | Weekly (production) | `pg_dump` |

---

## Support information

When reporting issues, include:

1. `APP_ENV`, `DEBUG` settings (not secrets)
2. Request query (sanitized)
3. Response status and `reason` field
4. Relevant backend log lines (JSON format preferred)
5. Whether issue reproduces with `DEBUG=true`
6. Docker Compose file used (dev vs prod)
