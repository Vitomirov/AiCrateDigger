# API Reference

| Environment | Base URL |
|-------------|----------|
| **Production (live)** | [https://aicratedigger.dejanvitomirov.com](https://aicratedigger.dejanvitomirov.com) — BFF at `/api/search` and `/api/parse` only |
| **Development (direct backend)** | `http://localhost:8000` |
| **Development (via BFF)** | `http://localhost:3000/api/*` |

In production, the backend is not publicly exposed. All client traffic goes through the Next.js BFF.

OpenAPI/Swagger is available at `/docs` when `APP_ENV=development`. Disabled in production.

---

## Authentication

When `INTERNAL_API_SECRET` is configured on the backend, all paid routes require:

```
X-Internal-Api-Secret: <shared-secret>
```

The Next.js BFF attaches this header automatically. Direct API calls must include it.

When the secret is **unset**, validation is bypassed (local development only).

**Client IP forwarding:** When secret is valid, the backend trusts `X-Forwarded-For` for rate limiting. See [Security](./security.md).

---

## Global routes

### GET /health

Health probe. No authentication required.

**Response 200:**

```json
{
  "status": "ok",
  "service": "backend",
  "database_configured": true
}
```

### GET /

Redirects (307) to `FRONTEND_PUBLIC_URL`.

---

## Search routes

All search routes share:

- Request body: `ParseRequest`
- Dependencies: internal auth + IP rate limiter
- Query length cap: `SEARCH_QUERY_MAX_LENGTH` (default 512)

### POST /parse

Parse natural language into structured query fields. Does not run Tavily or extraction.

**Request:**

```json
{
  "query": "The Doors Strange Days vinyl in Belgrade"
}
```

**Response 200:** `ParsedQuery`

```json
{
  "artist": "The Doors",
  "album": "Strange Days",
  "album_index": null,
  "resolved_album": null,
  "resolution_confidence": "unknown",
  "location": "Belgrade",
  "country_code": "RS",
  "search_scope": "local",
  "resolved_city": "Belgrade",
  "geo_confidence": 0.9,
  "geo_granularity": "city",
  "language": "en",
  "original_query": "The Doors Strange Days vinyl in Belgrade"
}
```

**Errors:**

| Status | Condition |
|--------|-----------|
| 401 | Missing or invalid internal secret |
| 422 | Validation error (empty query, exceeds max length) |
| 429 | Rate limit exceeded |
| 502 | Parse upstream failure |
| 503 | Quota exceeded; rate limiter fail-closed |

---

### POST /search

Full search pipeline: parse → cache → Tavily → prefilter → extract → dedupe.

**Request:** Same as `/parse`

**Response 200:** `SearchResponse`

```json
{
  "results": [
    {
      "url": "https://example-shop.com/product/strange-days-doors-vinyl",
      "title": "The Doors — Strange Days (LP)",
      "score": 0.92,
      "price": "€24.99",
      "location": "Belgrade",
      "availability": "available",
      "seller_type": "store",
      "domain": "example-shop.com",
      "artist_match": 1.0,
      "album_match": 0.95,
      "match_reason": "Snippet mentions artist and album title"
    }
  ],
  "parsed": { },
  "reason": null,
  "debug": null
}
```

**Empty response with reason:**

```json
{
  "results": [],
  "parsed": {
    "artist": "Radiohead",
    "album": null,
    "resolved_album": null,
    "original_query": "Radiohead third album vinyl"
  },
  "reason": "album_unresolved",
  "debug": null
}
```

**`reason` values:**

| Value | Meaning |
|-------|---------|
| `album_unresolved` | Parse could not anchor an album title; Tavily was not called |
| `null` | Pipeline ran; zero results is a valid outcome |

**`debug` object:** Present only when backend `DEBUG=true`. Contains pipeline stage traces from `PipelineContext`.

---

### POST /search-listings

Legacy alias for `POST /search`. Identical request, response, and behavior.

---

## Data models

### ParseRequest

| Field | Type | Constraints |
|-------|------|-------------|
| `query` | string | min 1 char, max `search_query_max_length` |

### ParsedQuery

| Field | Type | Description |
|-------|------|-------------|
| `artist` | string \| null | |
| `album` | string \| null | Literal title from query |
| `album_index` | int \| null | 1-based ordinal; -1 = latest |
| `resolved_album` | string \| null | LLM-resolved canonical title |
| `resolution_confidence` | enum | high, medium, low, unknown |
| `location` | string \| null | Verbatim geo substring |
| `country_code` | string \| null | ISO-3166-1 alpha-2 |
| `search_scope` | enum | local, regional, global |
| `resolved_city` | string \| null | |
| `geo_confidence` | float \| null | 0–1 |
| `geo_granularity` | enum | city, country, region, none |
| `language` | string | Default "unknown" |
| `original_query` | string | Echo of input |

### ListingResult

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | yes | min 8 chars |
| `title` | string | yes | min 5 chars |
| `score` | float | yes | 0–1 composite relevance |
| `price` | string \| null | | Price hint from snippet |
| `location` | string \| null | | Seller location hint |
| `availability` | enum | | available, sold_out, unknown (default unknown) |
| `seller_type` | enum | | store, private, unknown (default unknown) |
| `domain` | string \| null | | Hostname |
| `artist_match` | float | | 0–1 (default 0) |
| `album_match` | float | | 0–1 (default 0) |
| `match_reason` | string \| null | | Human-readable match explanation |

### SearchResponse

| Field | Type | Description |
|-------|------|-------------|
| `results` | ListingResult[] | Ranked listing rows |
| `parsed` | ParsedQuery | Parser output from this request |
| `reason` | string \| null | Empty-state reason code |
| `debug` | object \| null | Pipeline trace (DEBUG mode only) |

---

## BFF proxy routes

The frontend exposes same-origin proxies. Request/response shapes are identical to the backend routes above.

| Frontend route | Method | Proxies to |
|----------------|--------|------------|
| `/api/search` | POST | `{BACKEND_URL}/search` |
| `/api/parse` | POST | `{BACKEND_URL}/parse` |

**Additional BFF behavior:**

- Returns 400 on invalid JSON body
- Returns 502 when backend is unreachable
- Passes through all backend status codes and bodies unchanged

**Browser client usage:**

```typescript
import { postSearch } from "@/lib/api";

const response = await postSearch("Pink Floyd Dark Side vinyl London");
// response.results, response.parsed, response.reason
```

---

## Error responses

FastAPI standard error shape:

```json
{
  "detail": "Human-readable message"
}
```

Validation errors (422):

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "query"],
      "msg": "String should have at least 1 character"
    }
  ]
}
```

### Status code summary

| Code | Meaning |
|------|---------|
| 200 | Success |
| 307 | Redirect (GET /) |
| 400 | Invalid JSON (BFF only) |
| 401 | Invalid internal API secret |
| 422 | Request validation failure |
| 429 | Rate limit exceeded |
| 502 | Pipeline or upstream failure; BFF cannot reach backend |
| 503 | Quota exceeded/unavailable; rate limiter fail-closed (Redis down) |

---

## Rate limiting

Shared bucket across `/parse`, `/search`, and `/search-listings`.

Default: **5 requests per IP per 24 hours**.

429 response when exceeded. Configure via `SEARCH_RATE_LIMIT_*` variables. See [Configuration](./configuration.md).

---

## Quotas

Global daily caps (UTC reset) on provider calls:

| Provider action | Quota kind | Default cap |
|-----------------|------------|-------------|
| Parse LLM call | PARSE | 500/day |
| Tavily HTTP call | TAVILY | 200/day |
| Extract + discovery LLM | OPENAI_EXTRACT | 300/day |

503 when exceeded or when quota Redis is unavailable (fail-closed mode).

---

## Example curl commands

### Health check

```bash
curl -s http://localhost:8000/health | jq
```

### Parse (direct backend, dev without secret)

```bash
curl -s -X POST http://localhost:8000/parse \
  -H "Content-Type: application/json" \
  -d '{"query": "Daft Punk Discovery vinyl Paris"}' | jq
```

### Search via BFF

```bash
curl -s -X POST http://localhost:3000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Daft Punk Discovery vinyl Paris"}' | jq
```

### With internal secret

```bash
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -H "X-Internal-Api-Secret: your-secret" \
  -d '{"query": "Daft Punk Discovery vinyl Paris"}' | jq
```

---

## Versioning

API version is embedded in the FastAPI app metadata (`0.1.0`). No URL prefix versioning (`/v1/`) is implemented. Breaking changes should be coordinated with frontend DTO updates in `frontend/lib/api-types.ts`.
