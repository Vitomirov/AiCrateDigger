# Testing

AiCrateDigger has a **backend unit test suite** using Python's stdlib `unittest`. There are **no frontend automated tests** and **no CI pipeline** configured in the repository.

Pipeline quality is additionally validated by an offline **evaluation harness** — see [Evaluation](./evaluation.md).

---

## Backend test suite

**Location:** `backend/tests/`

**Runner:** stdlib `unittest` (no pytest)

**Framework:** Tests mock external dependencies (OpenAI, Tavily, Redis, pipeline) to run without API keys or live services.

---

## Running tests

### All tests

```bash
cd backend
poetry run python -m unittest discover -s tests -v
```

### Single module

```bash
poetry run python -m unittest tests.test_app_http_e2e -v
poetry run python -m unittest tests.test_rate_limit_security -v
poetry run python -m unittest tests.test_internal_auth_security -v
poetry run python -m unittest tests.test_global_quota -v
poetry run python -m unittest tests.test_search_cache_key -v
poetry run python -m unittest tests.test_parse_album_resolution -v
poetry run python -m unittest tests.test_eval_dataset -v
```

### Via Docker (if backend container is running)

```bash
docker compose exec backend python -m unittest discover -s tests -v
```

---

## Test modules

| File | Focus | Key assertions |
|------|-------|----------------|
| `test_app_http_e2e.py` | FastAPI HTTP layer | Health, OpenAPI availability, 422 validation, 502 error mapping, `/search-listings` alias |
| `test_rate_limit_security.py` | Rate limiting | Fail-closed behavior, shared bucket across routes |
| `test_internal_auth_security.py` | BFF auth | Secret validation, constant-time compare, XFF trust rules |
| `test_global_quota.py` | Daily quotas | `assert_quota_available()`, `record_quota_usage()` with Redis mock |
| `test_search_cache_key.py` | Cache keys | Normalization, paraphrase collision, city segment |
| `test_parse_album_resolution.py` | Parse helpers | `_derive_album_fields()`, `_build_parsed_payload()` unit tests |
| `test_eval_dataset.py` | Eval dataset | JSON schema smoke test for edge case dataset |

---

## Test environment pattern

Tests configure a safe environment **before importing** `app.main`:

```python
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["TAVILY_API_KEY"] = "test-key"
os.environ["DATABASE_URL"] = ""
os.environ["SEARCH_RATE_LIMIT_ENABLED"] = "false"
os.environ["GLOBAL_DAILY_QUOTA_ENABLED"] = "false"
```

This prevents accidental provider calls and Redis dependency during unit tests.

HTTP e2e tests use `TestClient` from FastAPI with mocked pipeline functions.

---

## What is tested

| Area | Coverage |
|------|----------|
| HTTP routing and status codes | Yes |
| Request validation | Yes |
| Internal auth gate | Yes |
| Rate limit fail-closed | Yes |
| Quota logic | Yes (mocked Redis) |
| Cache key generation | Yes |
| Parse album resolution helpers | Yes |
| Eval dataset schema | Yes |
| Full pipeline with live APIs | No (use eval harness) |
| Tavily integration | No |
| OpenAI integration | No |
| Postgres integration | No |
| Frontend components | No |

---

## What is not tested

- End-to-end pipeline with real OpenAI/Tavily calls (covered by eval harness, not CI)
- Database migrations or ORM integration
- Redis cache read/write integration
- Frontend React components or API routes
- Docker image builds
- Production Compose deployment

---

## Writing new tests

### Conventions

1. Place tests in `backend/tests/test_<area>.py`
2. Use `unittest.TestCase` or plain test functions with `unittest.main()`
3. Mock external I/O at module boundaries (`AsyncMock`, `patch`)
4. Set env vars before importing application modules
5. Use descriptive test method names: `test_<behavior>_<expected_outcome>`

### Example: testing a pure helper

```python
import unittest
from app.domains.query_parser.parse_user_query import _derive_album_fields

class TestAlbumFields(unittest.TestCase):
    def test_ordinal_index_sets_resolved_flag(self):
        result = _derive_album_fields({"album_index": 3, "album": None})
        self.assertEqual(result["album_index"], 3)
```

### Example: testing HTTP with mocks

Follow the pattern in `test_app_http_e2e.py`:

```python
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

with patch("app.api.routers.search.run_vinyl_search", new_callable=AsyncMock) as mock:
    mock.return_value = SearchResponse(results=[], parsed=..., reason=None)
    client = TestClient(app)
    response = client.post("/search", json={"query": "test"})
    self.assertEqual(response.status_code, 200)
```

---

## Frontend testing

No test framework is configured. Recommended additions for future work:

| Tool | Purpose |
|------|---------|
| Vitest or Jest | Unit tests for `lib/` helpers |
| React Testing Library | Component tests for SearchExperience |
| Playwright | E2E search flow against mocked backend |

---

## CI recommendations

The repository has no `.github/workflows/`. A minimal CI pipeline would:

```yaml
# Suggested workflow (not present in repo)
jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install poetry && cd backend && poetry install
      - run: cd backend && poetry run python -m unittest discover -s tests -v
        env:
          OPENAI_API_KEY: test
          TAVILY_API_KEY: test
          DATABASE_URL: ""
          SEARCH_RATE_LIMIT_ENABLED: "false"
          GLOBAL_DAILY_QUOTA_ENABLED: "false"

  frontend-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
      - run: cd frontend && npm ci && npm run build
```

Optional: run eval harness on schedule with secrets (costs API credits).

---

## Test vs evaluation

| Concern | Unit tests | Eval harness |
|---------|------------|--------------|
| Speed | Fast (mocked) | Slow (live APIs for full mode) |
| Cost | Free | OpenAI + Tavily charges |
| Scope | HTTP, security, helpers | End-to-end pipeline behavior |
| CI-friendly | Yes | Requires secrets; run manually or nightly |
| Dataset-driven | No | Yes (`eval/dataset/edge_cases.json`) |

Use unit tests for regression on code changes. Use eval for pipeline quality after parser, prefilter, or extraction changes.

---

## Manual smoke test

After deployment or significant changes:

1. Open http://localhost:3000
2. Search: `"Pink Floyd Dark Side of the Moon vinyl"`
3. Verify results or explicit empty state (not a generic error)
4. Search: `"Radiohead"` (no album) → expect `album_unresolved` messaging
5. With `DEBUG=true`: verify dev inspector shows pipeline stages

```bash
curl -s -X POST http://localhost:3000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Pink Floyd Dark Side vinyl"}' | jq '{count: (.results | length), reason}'
```
