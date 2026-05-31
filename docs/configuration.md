# Configuration

All services load environment variables from the repo-root `.env` file via Docker Compose `env_file`. The backend additionally discovers `backend/.env` or repo `.env` through `_discover_env_files()` in `app/core/config.py`. OS environment always overrides file values.

Override the env file path with `ENV_FILE`.

Template: [.env.example](../.env.example)

---

## Required variables

These must be set for the full search pipeline to function:

| Variable | Consumer | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Backend | OpenAI API key for parse and extract LLM calls |
| `TAVILY_API_KEY` | Backend | Tavily API key for web search |

Missing keys cause pipeline failures at runtime. Eval full-mode cases fail at CLI startup with an explicit error.

---

## Runtime mode

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `development` | `development` or `production`. Triggers startup guard when `production` |
| `DEBUG` | `false` | When true, API responses include `debug` pipeline traces; cache reads bypassed |

**Production requirements** (enforced by `validate_production_settings()`):

- `DEBUG=false`
- `INTERNAL_API_SECRET` set
- `SEARCH_RATE_LIMIT_ENABLED=true`
- `SEARCH_RATE_LIMIT_FAIL_CLOSED=true`
- `GLOBAL_DAILY_QUOTA_ENABLED=true`
- `GLOBAL_DAILY_QUOTA_FAIL_CLOSED=true`
- `DATABASE_URL` (or `POSTGRES_*`) configured
- `REDIS_URL` configured

See [Security](./security.md).

---

## Database

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | Primary DSN. Format: `postgresql+asyncpg://user:pass@host:port/db` |
| `POSTGRES_USER` | `aicratedigger` | Postgres user (Compose db service) |
| `POSTGRES_PASSWORD` | `aicratedigger` | Postgres password |
| `POSTGRES_DB` | `aicratedigger` | Database name |
| `POSTGRES_HOST` | — | Alternative to full URL (with other POSTGRES_*) |
| `POSTGRES_PORT` | `5433` (host) / `5432` (container) | Port mapping |

**Docker Compose DSN:**

```
DATABASE_URL=postgresql+asyncpg://aicratedigger:aicratedigger@db:5432/aicratedigger
```

**Host development (Poetry + Compose db):**

```
DATABASE_URL=postgresql+asyncpg://aicratedigger:aicratedigger@localhost:5433/aicratedigger
```

When unset, database layer no-ops. Suitable for unit tests and parse-only experiments.

---

## Redis

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection URL |
| `REDIS_PORT` | `6379` | Host port for Compose dev |

When unset: search cache no-ops; rate limit and quota behavior depends on fail-closed settings.

---

## Security

| Variable | Default | Description |
|----------|---------|-------------|
| `INTERNAL_API_SECRET` | `dev-change-me-before-production` | Shared BFF secret. Must match on backend and frontend. Unset = dev bypass |
| `SEARCH_RATE_LIMIT_ENABLED` | `true` | Enable per-IP rate limiting |
| `SEARCH_RATE_LIMIT_FAIL_CLOSED` | `true` | Return 503 when Redis unavailable |
| `SEARCH_RATE_LIMIT_MAX_REQUESTS` | `5` | Max requests per IP per window |
| `SEARCH_RATE_LIMIT_WINDOW_SECONDS` | `86400` | Rate limit window (24 hours) |
| `SEARCH_QUERY_MAX_LENGTH` | `512` | Maximum query string length |

Generate a production secret:

```bash
openssl rand -hex 32
```

---

## Global daily quotas

| Variable | Default | Description |
|----------|---------|-------------|
| `GLOBAL_DAILY_QUOTA_ENABLED` | `true` | Enable daily provider caps |
| `GLOBAL_DAILY_QUOTA_FAIL_CLOSED` | `true` | Block calls when Redis unavailable |
| `GLOBAL_DAILY_QUOTA_PARSE_MAX` | `500` | Daily parse LLM calls; `0` = unlimited |
| `GLOBAL_DAILY_QUOTA_TAVILY_MAX` | `200` | Daily Tavily HTTP calls |
| `GLOBAL_DAILY_QUOTA_OPENAI_EXTRACT_MAX` | `300` | Daily extract + discovery LLM calls |

Counters reset at UTC midnight. Stored in Redis keys `quota:{kind}:{YYYY-MM-DD}`.

---

## Search cache

| Variable | Default | Description |
|----------|---------|-------------|
| `SEARCH_CACHE_ENABLED` | `true` | Enable search response caching |
| `REDIS_SEARCH_CACHE_TTL_SECONDS` | `604800` | Cache TTL (7 days) |

Cache schema version is code-defined (`PIPELINE_CACHE_SCHEMA_VERSION = 3`). Changing extraction or response shape requires bumping the version in `search_cache_key.py`.

---

## Pipeline knobs

These are defined in `Settings` (`app/core/config.py`) and can be set via environment:

| Variable | Default | Description |
|----------|---------|-------------|
| `PIPELINE_MAX_RESULTS` | `4` | Max listings returned to client |
| `PIPELINE_PREFILTER_MAX_CANDIDATES` | `6` | Max Tavily rows after prefilter |
| `PIPELINE_PREFILTER_MAX_PER_HOST` | `1` | Max candidates per hostname |
| `PIPELINE_OPPORTUNISTIC_STORE_DISCOVERY_ENABLED` | `true` | Background store discovery from snippets |
| `PIPELINE_OPPORTUNISTIC_DISCOVERY_MIN_UNKNOWN_HOSTS` | `2` | Threshold to trigger opportunistic discovery |
| `TAVILY_SINGLE_CALL_MAX_RESULTS` | `10` | Tavily result cap per request |
| `TAVILY_SINGLE_CALL_DEPTH` | `advanced` | Tavily search depth |
| `TAVILY_HTTP_RETRY_ATTEMPTS` | `5` | Tavily HTTP retry count |
| `TAVILY_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `2` | Fail-fast threshold for Tavily throttling |

---

## Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Python log level |
| `LOG_FORMAT` | `human` | `human` or `json`. Production recommends `json` |

---

## Frontend

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_URL` | `http://backend:8000` | Server-side proxy target (Docker network) |
| `NEXT_PUBLIC_BACKEND_URL` | `http://localhost:8000` | SSR health checks only — not for browser search |
| `FRONTEND_PUBLIC_URL` | `http://localhost:3000` | Public UI URL (backend redirect target) |
| `FRONTEND_PORT` | `3000` | Host port in production Compose |
| `NEXT_PUBLIC_DEV_INSPECTOR` | — | Force show/hide pipeline JSON inspector |

**Important:** Search traffic from the browser always goes to `/api/search` (same origin). Never expose backend URL or secrets via `NEXT_PUBLIC_*` for paid routes.

---

## Settings resolution order

```
1. OS environment variables (highest priority)
2. .env file(s) discovered by _discover_env_files()
3. Pydantic field defaults in Settings class
```

Backend loads from:

- `ENV_FILE` if set
- `backend/.env`
- Repo root `.env`

Frontend reads env at build time for `NEXT_PUBLIC_*` and at runtime for server-only vars.

---

## Environment profiles

### Local development

```env
APP_ENV=development
DEBUG=false
INTERNAL_API_SECRET=dev-change-me-before-production
DATABASE_URL=postgresql+asyncpg://aicratedigger:aicratedigger@db:5432/aicratedigger
REDIS_URL=redis://redis:6379/0
BACKEND_URL=http://backend:8000
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
FRONTEND_PUBLIC_URL=http://localhost:3000
LOG_FORMAT=human
```

Swagger available at `http://localhost:8000/docs` (if backend port published).

### Production

```env
APP_ENV=production
DEBUG=false
INTERNAL_API_SECRET=<openssl rand -hex 32>
DATABASE_URL=postgresql+asyncpg://...
REDIS_URL=redis://redis:6379/0
FRONTEND_PUBLIC_URL=https://aicratedigger.dejanvitomirov.com
LOG_FORMAT=json
# Do not set NEXT_PUBLIC_DEV_INSPECTOR
```

Deploy with:

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

See [Deployment](./deployment.md).

---

## Configuration validation

| Check | When | Behavior |
|-------|------|----------|
| Production guard | Backend import | `RuntimeError` on misconfiguration |
| BFF guard | Next.js route handler | Throws if production without secret |
| API key presence | Pipeline runtime | 502 on provider failure |
| Query length | Request validation | 422 |

No configuration hot-reload. Changes require service restart.
