# Overview

## Live demo

**[https://aicratedigger.dejanvitomirov.com/](https://aicratedigger.dejanvitomirov.com/)** — production deployment behind HTTPS with the Next.js BFF as the only public entry point. See [Deployment](./deployment.md) for running the stack locally or on your own host.

---

## Purpose

AiCrateDigger addresses a practical problem: finding a **specific album** in a **specific geography** usually requires visiting many regional record-shop websites individually. The system combines:

1. **Intent parsing** — extract artist, album (including ordinal references like "their third album"), and location from natural language
2. **Constrained web retrieval** — one Tavily search call scoped to whitelisted commerce domains
3. **Structured extraction** — LLM-assisted parsing of search snippets into listing-shaped rows (URL, title, price hints, availability signals)

The output is actionable commerce links, not a generic search results page.

---

## Core capabilities

| Capability | Description |
|------------|-------------|
| **Parse** | Single LLM call produces structured `ParsedQuery`: artist, album, ordinal resolution, geo inference (`country_code`, `search_scope`, `resolved_city`) |
| **Store catalogue** | Postgres-backed `whitelist_stores`, seeded from policy modules; optional inline store discovery when coverage is thin |
| **Search** | One consolidated Tavily call per request with configurable depth and result cap |
| **Prefilter** | Python gate on raw Tavily rows: blacklist, PDP URL heuristics, per-host caps |
| **Extract** | Snippet-first pipeline: deterministic paths + `gpt-4o-mini` JSON extraction |
| **Cache** | Redis (7-day TTL) with Postgres fallback for operator visibility |
| **Explicit empty states** | `reason: album_unresolved` when parse cannot anchor an album title |

---

## Design principles

### Tavily + snippets, not site crawling

Full-page HTML scraping would increase latency, operational burden, and compliance risk. Snippets bound token usage and keep the problem in the **search → extract** domain rather than **browse → render**.

### Whitelist-first retrieval

Open web search drifts toward megamarket SEO. Curated store domains (Postgres + policy seed) plus opportunistic discovery keep results commerce-shaped and easier to validate.

### One Tavily call per search

Location influences parse output, store discovery, and prefilter whitelist signals — but the hot path issues **one** consolidated Tavily request rather than widening geo tiers with multiple HTTP calls.

### Single parser contract

`parse_user_query` produces one `ParsedQuery`. Ordinal album titles resolve inside the same LLM call (`resolved_album`), not via a separate metadata service.

### Structured failure over silent zeros

When no album anchor exists after parse, the API returns `reason: album_unresolved` so clients do not appear broken.

### Cost-aware defaults

Redis TTL cache, Tavily HTTP retries, per-request circuit breaker for hard throttling, global daily quotas on parse/Tavily/extract calls, and a capped prefilter candidate pool.

### BFF security boundary

The browser never calls the FastAPI backend directly for paid routes. Next.js route handlers proxy requests with a shared secret and forwarded client IP.

---

## Technology summary

| Layer | Stack |
|-------|-------|
| Backend runtime | Python 3.11, FastAPI, Uvicorn, Pydantic v2 |
| Database | PostgreSQL 15, SQLAlchemy 2.0 async, asyncpg |
| Cache / limits | Redis 7 |
| External APIs | OpenAI (`gpt-4o-mini`), Tavily |
| Fuzzy matching | RapidFuzz |
| Frontend | Next.js 14 App Router, React 18, TypeScript 5 strict, Tailwind CSS 3 |
| Packaging | Poetry (backend), npm (frontend) |
| Infrastructure | Docker Compose |

---

## Repository layout

```
AiCrateDigger/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI entry, lifespan, health
│   │   ├── api/routers/            # HTTP routes
│   │   ├── core/                   # Config, DB, cache, security, quotas
│   │   └── domains/                # Business logic
│   │       ├── query_parser/       # LLM parse
│   │       ├── search_pipeline/    # Orchestration (vinyl_search)
│   │       └── engine/             # Tavily, prefilter, extraction, policies
│   ├── eval/                       # Offline pipeline evaluation CLI
│   ├── tests/                      # unittest suite
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/
│   ├── app/                        # Pages + API route handlers (BFF)
│   ├── components/                 # React UI
│   ├── lib/                        # API client, config, production guard
│   ├── Dockerfile
│   └── package.json
├── docs/                           # This documentation set
├── docker-compose.yml              # Development stack
├── docker-compose.prod.yml         # Production stack
├── .env.example                    # Environment template
└── README.md                       # Project README
```

---

## What is explicitly out of scope

| Item | Note |
|------|------|
| User authentication | No login, sessions, or RBAC |
| Alembic migrations | Schema managed via `create_all` and inline startup alters |
| HTML crawling | No Crawl4AI or first-party page fetcher |
| Vector store / Chroma | Legacy `chroma_db/` directory is gitignored and unused |
| Frontend automated tests | No Jest/Playwright suite in repository |
| CI/CD pipelines | No GitHub Actions or equivalent configured |

---

## Naming

The project may appear as **AiCrateDigger** or **AiCrateDigg** in code (e.g. API title `"AiCrateDigg API"`). Both refer to the same system.
