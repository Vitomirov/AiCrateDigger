# Evaluation

The evaluation harness runs the search pipeline against a **JSON dataset of edge cases** and reports pass/fail per case. It is designed for offline quality assurance — not continuous integration by default (requires live API keys and incurs provider costs).

**Location:** `backend/eval/`

---

## Purpose

| Goal | Description |
|------|-------------|
| Regression detection | Catch parse and pipeline behavior changes against known queries |
| Edge case coverage | Ordinal albums, unresolved titles, geo inference, empty results |
| Stage tracing | Assert pipeline stages appear with expected status in DEBUG mode |
| Reproducibility | Cache bypass by default for consistent stage traces |

Unit tests mock providers. Eval exercises real OpenAI and Tavily calls in `full` mode.

---

## Running eval

### Docker Compose (recommended)

```bash
docker compose --profile eval run --rm eval
```

Uses the same `.env` as the dev stack. Requires `OPENAI_API_KEY` and `TAVILY_API_KEY` for full-mode cases.

### Local (Poetry)

```bash
cd backend
export OPENAI_API_KEY=...
export TAVILY_API_KEY=...
poetry run python -m eval.cli
```

---

## CLI reference

**Entry point:** `python -m eval.cli`

| Flag | Description |
|------|-------------|
| `--dataset PATH` | Path to JSON dataset (default: `eval/dataset/edge_cases.json`) |
| `--case ID` | Run only specific case id(s); repeatable |
| `--mode {all,parse,full}` | Filter cases by mode (default: `all`) |
| `--json-out PATH` | Write machine-readable report to file |
| `--use-cache` | Allow Redis cache hits (default: bypass cache) |
| `--list` | List case ids and exit |

### Examples

```bash
# List all cases
poetry run python -m eval.cli --list

# Parse-only cases (no Tavily)
poetry run python -m eval.cli --mode parse

# Single case
poetry run python -m eval.cli --case ordinal_third_album

# Full report to JSON
poetry run python -m eval.cli --json-out /tmp/eval-report.json
```

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | All cases passed |
| 1 | One or more cases failed |
| 2 | Dataset not found or missing API keys for full mode |

---

## Dataset format

**Default file:** `backend/eval/dataset/edge_cases.json`

**Schema:** Defined in `backend/eval/schema.py`

```json
{
  "version": "1",
  "description": "Edge cases for pipeline evaluation",
  "cases": [
    {
      "id": "example_case",
      "query": "The Beatles Abbey Road vinyl London",
      "description": "Basic artist + album + city",
      "tags": ["happy-path", "geo"],
      "mode": "full",
      "expect": {
        "parse": {
          "artist": "The Beatles",
          "album": "Abbey Road",
          "country_code": "GB",
          "search_scope": "local",
          "fields_present": ["resolved_city"]
        },
        "pipeline": {
          "reason": null,
          "min_results": 0,
          "max_results": 4,
          "stages": {
            "parse": { "required": true, "status_in": ["success"] },
            "tavily": { "required": true, "status_in": ["success", "empty"] }
          }
        }
      }
    }
  ]
}
```

### Case fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique case identifier |
| `query` | string | Natural language input |
| `description` | string | Human-readable case purpose |
| `tags` | string[] | Optional categorization |
| `mode` | `"parse"` \| `"full"` | `parse` = parse only; `full` = complete pipeline |
| `expect.parse` | ParseExpectation | Subset match on ParsedQuery fields |
| `expect.pipeline` | PipelineExpectation | Full pipeline outcome assertions |

### ParseExpectation

Subset match on `ParsedQuery` fields. Special lists:

| Field | Purpose |
|-------|---------|
| `fields_present` | Keys that must be non-null after parse |
| `fields_absent` | Keys that must be null after parse |

All other fields are matched exactly when specified.

### PipelineExpectation

| Field | Purpose |
|-------|---------|
| `reason` | Expected `SearchResponse.reason` (e.g. `"album_unresolved"`) |
| `min_results` / `max_results` | Bounds on result count |
| `stages` | Map of stage name → `StageExpectation` |

### StageExpectation

| Field | Default | Purpose |
|-------|---------|---------|
| `required` | true | Stage must appear in debug trace |
| `status_in` | success, empty | Allowed stage status values |

---

## Execution flow

```
eval.cli
  → load_dataset(path)
  → filter by --case / --mode
  → bootstrap_eval_env()     # DB/Redis setup for eval
  → run_dataset()
      → for each case:
          parse mode: parse_user_query() + check parse expectations
          full mode:  run_vinyl_search(debug=True) + check parse + pipeline
  → shutdown_eval_env()
  → print_report() / report_to_json()
```

**Cache:** Bypassed by default (`bypass_cache=True`) unless `--use-cache` is passed. Ensures reproducible stage traces and fresh provider calls.

---

## Report output

Human-readable report printed to stdout via `eval/report.py`:

```
Dataset: edge_cases.json (version 1)
Mode: all | Total: N | Passed: X | Failed: Y

[PASS] case_id — query snippet
[FAIL] case_id — query snippet
  - parse.artist: expected "X", got "Y"
  - pipeline.stages.tavily: missing required stage
```

JSON report structure (`EvalReport`):

```json
{
  "dataset_version": "1",
  "mode": "all",
  "total": 10,
  "passed": 9,
  "failed": 1,
  "cases": [
    {
      "case_id": "example",
      "query": "...",
      "mode": "full",
      "passed": false,
      "stages": [],
      "errors": ["parse.country_code: expected RS, got null"],
      "result_count": 0,
      "reason": null
    }
  ]
}
```

---

## Adding cases

1. Edit `backend/eval/dataset/edge_cases.json`
2. Add a case with unique `id`, descriptive `tags`, and appropriate `mode`
3. Start with `mode: "parse"` for parse-only regression (no API cost beyond OpenAI parse)
4. Add `pipeline` expectations when testing full flow
5. Run: `poetry run python -m eval.cli --case your_new_id`
6. Add schema smoke coverage in `tests/test_eval_dataset.py` if structure changes

### Guidelines for good cases

| Category | Example query | What to assert |
|----------|---------------|----------------|
| Ordinal album | "Radiohead third album vinyl" | `resolved_album`, `album_index: 3` |
| Unresolved album | "Radiohead vinyl" | `reason: album_unresolved` |
| Geo local | "... in Belgrade" | `country_code: RS`, `search_scope: local` |
| Geo global | "... anywhere" | `search_scope: global` |
| Format hint | "... CD" | Format detection in parse or pipeline |
| Empty Tavily | Obscure query | `max_results: 0`, stage status `empty` |

---

## Cost considerations

Full-mode cases incur:

- 1× OpenAI parse call per case
- 0–1× Tavily call per case (skipped on `album_unresolved`)
- 0–N× OpenAI extract calls depending on prefilter candidates

Run parse-only mode during development:

```bash
poetry run python -m eval.cli --mode parse
```

Global daily quotas (`GLOBAL_DAILY_QUOTA_*`) apply during eval unless disabled.

---

## Relationship to unit tests

| | Unit tests | Eval |
|---|-----------|------|
| Location | `backend/tests/` | `backend/eval/` |
| Provider calls | Mocked | Live (full mode) |
| Speed | Milliseconds | Seconds to minutes |
| Dataset | Inline assertions | JSON edge case file |
| CI default | Yes | No (needs secrets) |

`tests/test_eval_dataset.py` validates the JSON dataset schema — not pipeline outcomes.

---

## Troubleshooting

| Issue | Resolution |
|-------|------------|
| Exit code 2, missing keys | Set `OPENAI_API_KEY` and `TAVILY_API_KEY` |
| All full cases fail at Tavily | Check Tavily key, quota, circuit breaker logs |
| Parse failures after prompt change | Update expectations or fix parser |
| Stage not found in trace | Ensure `DEBUG` effective during eval; check stage name spelling |
| Stale results | Remove `--use-cache` flag (default bypasses cache) |
