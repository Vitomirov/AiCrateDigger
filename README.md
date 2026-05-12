# AiCrateDigger

**AiCrateDigger** (also referenced internally as *AiCrateDigg*) is an AI-assisted search application for **physical music releases**—primarily **vinyl**, plus **CD** and **cassette**. Users describe what they want in natural language (artist, album, format, and optional geography); the system parses the intent, queries the web through curated store domains, and returns **structured listings** (URLs, titles, prices when visible, availability signals, and relevance scoring).

---

## What problem it solves

Finding buyable copies of a specific release often means hunting across many regional shops and marketplaces. AiCrateDigger combines:

1. **Structured parsing** of messy user queries (including ordinals like “second album” and location hints).
2. **Music metadata** from **Discogs** where the pipeline needs canonical album alignment.
3. **Geo-aware domain targeting** so search is biased toward relevant regions and whitelisted retailers—not generic global noise.
4. **LLM-assisted extraction** from search snippets to turn raw hits into normalized listing objects.
5. **Validation and ranking** (e.g. fuzzy matching, PDP heuristics, configurable thresholds).

---

## Project stage

| Aspect | Status |
|--------|--------|
| **Version** | `0.1.0` (backend and frontend package metadata) |
| **Maturity** | **Early / alpha** — core search pipeline is implemented end-to-end; behavior and tuning evolve quickly |
| **Persistence** | PostgreSQL optional in dev (`DATABASE_URL`); startup skips DB features when unset |
| **Migrations** | SQLAlchemy `create_all` only (**no Alembic** yet) |
| **Tests** | Small **unittest** suite (`backend/tests/`) focused on helpers and regressions |

Treat production deployment as requiring your own hardening (secrets, rate limits, observability, and legal/compliance review for scraping and third-party APIs).

---

## Languages & major frameworks

| Layer | Stack |
|-------|--------|
| **Backend** | Python **3.11**, **FastAPI**, **Pydantic v2** (`pydantic-settings`), **SQLAlchemy 2.0** (async) + **asyncpg**, **httpx**, **Poetry** |
| **Frontend** | **Next.js 14** (App Router), **React 18**, **TypeScript 5** (strict), **Tailwind CSS 3** |
| **Infra** | **Docker** / **Docker Compose**, **PostgreSQL 15** (Compose image) |

---

## External services & APIs

| Service | Role |
|---------|------|
| [**OpenAI**](https://platform.openai.com/) | All LLM calls use **`gpt-4o-mini`** via **`AsyncOpenAI`**, JSON-mode responses (`response_format={"type": "json_object"}`), temperature **0** for extraction-style tasks |
| [**Tavily**](https://tavily.com/) | Web search with **`include_domains`** batching, configurable depth/score thresholds (defaults tuned for cost vs. quality in `Settings`) |
| [**Discogs API**](https://www.discogs.com/developers/) | Artist/discography and release alignment (`DISCOGS_TOKEN` optional but recommended for enrichment) |

---

## LLM usage (high level)

The **live HTTP pipeline** (`POST /search`) roughly follows:

1. **Parse** (`app.llm.parse_user_query`) — one structured JSON object per query: artist, album, album index, location, ISO country code, search scope.
2. **Search** (`app.services.tavily_service`) — Tavily calls constrained by geo-tiered **whitelist stores** loaded from DB or policy fallbacks.
3. **Extract** (`app.llm.extract_listings`) — maps search snippets to listing rows (price, currency, stock hints, titles).
4. **Validate & rank** (`app.validators.listings`, RapidFuzz, URL normalization) — filters and scores before response.

The repository also contains **`app.agents.parser`** and **`app.agents.extractor`**—richer, alternative agent-style modules. The **documented primary path** for the API is the `llm/` + `pipeline/vinyl_search.py` stack above.

> **Note:** Project documentation in `.cursorrules` mentions **Crawl4AI** for full-page markdown extraction. That is **not** present in `pyproject.toml` today; retrieval is **Tavily-centric** with snippet-based LLM extraction.

---

## Repository layout

```
AiCrateDigger/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + lifespan (DB seed/cache purge)
│   │   ├── config.py            # Pydantic settings (env-driven)
│   │   ├── routers/search.py    # /parse, /search, /search-listings
│   │   ├── pipeline/vinyl_search.py
│   │   ├── llm/                 # parse_user_query, extract_listings, coercion helpers
│   │   ├── agents/              # parser.py, extractor.py (alternate/heavier paths)
│   │   ├── services/          # tavily_service, discogs_service, domain batching
│   │   ├── policies/           # geo_scope, eu_stores, store domains, query DSL
│   │   ├── db/                 # async SQLAlchemy, cache, store loader
│   │   ├── domain/             # parse/listing schemas
│   │   ├── models/             # API-facing Pydantic models
│   │   └── validators/
│   └── tests/
├── frontend/
│   ├── app/                    # Next.js App Router pages & API route proxy
│   ├── components/
│   └── lib/api.ts              # Client fetch helpers
├── docker-compose.yml
└── README.md
```

---

## Configuration

Backend settings are loaded from **environment variables** (and optional `.env`). Important keys:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required for parsing and extraction |
| `TAVILY_API_KEY` | Required for web search |
| `DISCOGS_TOKEN` | Optional; improves music resolution |
| `DATABASE_URL` | Optional `postgresql+asyncpg://…` or `postgresql://…` (coerced to async driver) |

Additional knobs (TTLs, Tavily chunk sizes, fuzzy validation thresholds, geo tier caps) live in **`backend/app/config.py`** as `Settings` fields.

---

## Running locally

### Docker Compose (recommended)

From the repo root:

```bash
docker compose up --build
```

- **Frontend:** http://localhost:3000  
- **Backend API:** http://localhost:8000  
- **PostgreSQL:** exposed on host port **5433** by default (see `docker-compose.yml`)

Ensure `.env` or shell exports supply API keys for OpenAI and Tavily.

### Backend only (Poetry)

```bash
cd backend
poetry install
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend only (npm)

```bash
cd frontend
npm install
npm run dev
```

Point the UI at the API with `NEXT_PUBLIC_BACKEND_URL` (browser) and/or `BACKEND_URL` (Next.js server-side proxy in `app/api/search/route.ts`).

---

## API overview

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness |
| `POST` | `/parse` | Parse query only (debugging / introspection) |
| `POST` | `/search` | Full pipeline: parse → Tavily → extract → validate |
| `POST` | `/search-listings` | Alias of `/search` |

Request body for parse/search: JSON `{"query": "<natural language>"}`.

---

## Tests

```bash
cd backend
poetry run python -m unittest discover -s tests -p 'test_*.py' -v
```

---

## License / naming

Verify licensing before redistribution. Naming varies slightly (**AiCrateDigger** vs **AiCrateDigg**) across files; both refer to this codebase.

---

## Contributing mindset

When extending the project:

- Prefer **async** end-to-end for I/O-bound paths.
- Keep **secrets out of logs**; use structured logging already wired in `logging_config`.
- Align new retrieval or extraction steps with existing **whitelist + geo tier** policies so results stay locally relevant and maintainable.
