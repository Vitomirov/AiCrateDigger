# Backend

The backend is a Python 3.11 **FastAPI** application organized by domain. Business logic lives under `app/domains/`; cross-cutting infrastructure under `app/core/`.

---

## Entry point

**File:** `backend/app/main.py`

| Responsibility | Detail |
|----------------|--------|
| App factory | FastAPI title `"AiCrateDigger API"`, version `0.1.0` |
| Lifespan | DB init, store seed/sync, cache purge on startup; engine disposal on shutdown |
| Production guard | `validate_production_settings()` at import time |
| OpenAPI | Disabled when `APP_ENV=production` |
| Global routes | `GET /` → redirect to `FRONTEND_PUBLIC_URL`; `GET /health` |
| Exception handlers | `QuotaExceededError`, `QuotaUnavailableError` → 503 |

---

## Project structure

```
backend/app/
├── main.py
├── api/routers/
│   └── search.py              # POST /parse, /search, /search-listings
├── core/
│   ├── config.py              # Settings (Pydantic BaseSettings)
│   ├── production_guard.py    # Production startup validation
│   ├── logging_config.py      # human vs JSON formatters
│   ├── internal_auth.py       # X-Internal-Api-Secret gate
│   ├── client_ip.py           # IP resolution for rate limiting
│   ├── rate_limiter.py        # Per-IP sliding window (Redis)
│   ├── quota/                 # Global daily spend caps
│   └── db/
│       ├── database.py        # ORM models, init_db, sessions
│       ├── cache.py           # Postgres search cache
│       ├── redis_cache.py     # Redis search cache
│       ├── search_cache_key.py
│       └── store_loader.py    # Whitelist seed/sync/load
└── domains/
    ├── query_parser/
    │   ├── parse_user_query.py
    │   └── parse_schema.py    # ParsedQuery model
    ├── search_pipeline/
    │   ├── vinyl_search.py    # Production hot path
    │   ├── pipeline_context.py
    │   └── models/            # API DTOs
    └── engine/
        ├── listing_schema.py
        ├── search/            # Tavily, prefilter, store_discovery
        ├── extraction/steps/  # step_01 … step_05
        ├── policies/          # eu_stores, geo_scope, format_detect
        ├── validators/listings.py
        └── llm/
```

**Packaging:** Poetry with `package-mode = false` — dependencies only; `app/` is copied into the Docker image.

---

## Configuration

Settings class: `app.core.config.Settings`, accessor `get_settings()` (LRU cached).

Env file discovery loads `backend/.env` and/or repo-root `.env`. OS environment always wins. Override path with `ENV_FILE`.

Full variable reference: [Configuration](./configuration.md).

---

## API router

**File:** `app/api/routers/search.py`

| Route | Handler | Behavior |
|-------|---------|----------|
| `POST /parse` | `parse_search_query()` | Parse-only; returns `ParsedQuery` |
| `POST /search` | `search()` | Full pipeline via `_execute_search()` |
| `POST /search-listings` | `search_listings()` | Legacy alias of `/search` |

**Router dependencies (all routes):**

- `Depends(require_internal_api_secret)` — BFF shared secret
- `Depends(ip_rate_limiter)` — per-IP rate limit

---

## Query parser

**Module:** `app/domains/query_parser/`

### `parse_user_query(query: str) -> ParsedQuery`

- Single `AsyncOpenAI` call with `response_format={"type": "json_object"}`, temperature 0
- Model: `gpt-4o-mini`
- Quota kind: `QuotaKind.PARSE`
- Helpers: `_derive_album_fields()`, `_build_parsed_payload()`

### `ParsedQuery` fields

| Field | Type | Description |
|-------|------|-------------|
| `artist` | `str \| None` | Artist name |
| `album` | `str \| None` | Literal album title from query |
| `album_index` | `int \| None` | 1-based ordinal; `-1` = latest |
| `resolved_album` | `str \| None` | LLM-resolved canonical title |
| `resolution_confidence` | enum | `high`, `medium`, `low`, `unknown` |
| `location` | `str \| None` | Verbatim geo substring |
| `country_code` | `str \| None` | ISO-3166-1 alpha-2 |
| `search_scope` | enum | `local`, `regional`, `global` |
| `resolved_city` | `str \| None` | Normalized city |
| `geo_confidence` | `float \| None` | 0–1 |
| `geo_granularity` | enum | `city`, `country`, `region`, `none` |
| `language` | `str` | Default `"unknown"` |
| `original_query` | `str` | Echo of input |

**Property:** `effective_album` → `resolved_album or album`

---

## Search pipeline

**Module:** `app/domains/search_pipeline/vinyl_search.py`

### `run_vinyl_search(query: str, *, debug: bool = False) -> SearchResponse`

Production orchestrator. Stages are timed and recorded in `PipelineContext` when debug is enabled.

**Early exit:** If `effective_album` is empty after parse, returns empty results with `reason: "album_unresolved"` — Tavily is never called.

**Cache hit:** Returns hydrated `ListingResult[]` with zero provider calls.

**Dedupe:** `_dedupe_listings_by_host()` keeps best-scoring listing per domain.

### Debug tracing

`pipeline_context.py` provides:

- `start_pipeline(debug=...)` — request-scoped context
- `stage_timer(name)` — context manager for stage timing/status
- `ctx.as_debug_payload()` — serialized trace for API response

Enabled when backend `DEBUG=true`.

---

## Engine: Tavily search

**Module:** `app/domains/engine/search/`

| File | Role |
|------|------|
| `single_call.py` | `build_consolidated_query()`, `run_consolidated_tavily_search()` |
| `client.py` | Tavily HTTP client, retries, quota hooks |
| `prefilter.py` | `prefilter_tavily_results()` — blacklist, PDP heuristics, caps |
| `store_discovery.py` | `discover_new_stores()`, `discover_stores_from_snippets()` |
| `circuit_breaker.py` | `tavily_circuit_breaker_scope()` — per-request throttling protection |

**Query construction:** Tavily query text is built in `build_consolidated_query()` — not a separate LLM planning step.

**Prefilter knobs** (via Settings):

- `pipeline_prefilter_max_candidates` (default 6)
- `pipeline_prefilter_max_per_host` (default 1)

**Tavily knobs:**

- `tavily_single_call_max_results` (default 10)
- `tavily_single_call_depth` (default `"advanced"`)
- `tavily_http_retry_attempts` (default 5)

---

## Engine: Extraction

**Module:** `app/domains/engine/extraction/`

Orchestrator: `step_05_listings_orchestrator.py` → `extract_listings()`

| Step | File | Function |
|------|------|----------|
| 01 | `step_01_snippet_prefilter.py` | `collect_snippet_candidates()` |
| 02 | `step_02_listing_deterministic.py` | `deterministic_listings_from_candidates()` |
| 03 | `step_03_listing_llm_extract.py` | `llm_extract()` |
| 04 | `step_04_merge_llm_listings.py` | `merge_llm_rows_into_listings()` |

Returns `ExtractListingsReport` with `listings: list[Listing]`.

**Validation:** `engine/validators/listings.py` exposes PDP URL heuristics (`url_suggests_product_detail_page`) used consistently in prefilter and extraction.

**Fuzzy matching:** RapidFuzz used for evidence alignment and geo proximity where applicable.

---

## Engine: Policies

**Module:** `app/domains/engine/policies/`

| Module | Purpose |
|--------|---------|
| `eu_stores.py` | `ALLOWED_STORES` seed catalogue |
| `store_domain.py` | Domain normalization helpers |
| `geo_scope.py` | Geographic scope rules |
| `geo_proximity.py` | City matching (RapidFuzz) |
| `format_detect.py` | Physical format detection (vinyl, CD, cassette) |

Store catalogue is DB-backed (`whitelist_stores`) with code seed from `ALLOWED_STORES`.

---

## Store loader

**Module:** `app/core/db/store_loader.py`

| Function | Purpose |
|----------|---------|
| `seed_whitelist_stores_if_empty()` | Initial population from policy |
| `sync_whitelist_store_catalogue()` | Upsert from policy on startup |
| `repair_whitelist_store_domains()` | Fix/normalize domain values |
| `load_active_stores()` | Runtime store list for pipeline |

When `DATABASE_URL` is unset, pipeline falls back to in-code `get_active_stores()`.

---

## Quota system

**Module:** `app/core/quota/`

Global daily caps stored in Redis with UTC midnight reset.

| Kind | Enforced at | Default cap |
|------|-------------|-------------|
| `PARSE` | `parse_user_query` | 500/day |
| `TAVILY` | Tavily client | 200/day |
| `OPENAI_EXTRACT` | OpenAI extract + store discovery LLM | 300/day |

`0` = unlimited for that bucket. Fail-closed mode returns 503 when Redis is unavailable.

---

## Rate limiting

**Module:** `app/core/rate_limiter.py`

- Redis sorted-set sliding window per IP
- Key: `rate_limit:api:{ip}`
- Shared bucket across `/parse`, `/search`, `/search-listings`
- Default: 5 requests per 86400 seconds (24 hours)
- Fail-closed → 503 when Redis unavailable (configurable)

Client IP from `client_ip.resolve_client_ip()` — honors `X-Forwarded-For` only when BFF secret is valid.

---

## Dependencies

From `backend/pyproject.toml`:

| Package | Version constraint | Purpose |
|---------|-------------------|---------|
| fastapi | * | Web framework |
| uvicorn[standard] | * | ASGI server |
| pydantic-settings | * | Settings loading |
| sqlalchemy[asyncio] | ^2.0 | Async ORM |
| asyncpg | * | Postgres driver |
| openai | * | AsyncOpenAI |
| httpx | * | Tavily HTTP |
| rapidfuzz | ^3.9 | Fuzzy matching |
| redis | ^5.0 | Async Redis client |

**Python:** `^3.11`

---

## Running locally

### With Docker Compose (recommended)

From repo root:

```bash
cp .env.example .env
# Set OPENAI_API_KEY and TAVILY_API_KEY
docker compose up --build
```

Backend runs inside the `backend` service; not published to host by default.

### With Poetry (host)

```bash
cd backend
poetry install
export OPENAI_API_KEY=... TAVILY_API_KEY=...
export DATABASE_URL=postgresql+asyncpg://...@localhost:5433/aicratedigger
export REDIS_URL=redis://localhost:6379/0
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Ensure Postgres and Redis are reachable (Compose db/redis services or local installs).

---

## Docker targets

**File:** `backend/Dockerfile`

| Target | Command |
|--------|---------|
| `dev` | `uvicorn app.main:app --host 0.0.0.0 --port 8000` (1 worker) |
| `production` | Same with `--workers 2` |

Copies `app/` and `eval/` after `poetry install`.

---

## Logging

Configured in `app/core/logging_config.py`:

| Setting | Values |
|---------|--------|
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FORMAT` | `human` (dev) or `json` (production recommended) |

Structured fields use `extra={}` dicts on log records. Database URLs are masked in logs via `mask_database_url()`.

See [Operations](./operations.md) for observability guidance.
