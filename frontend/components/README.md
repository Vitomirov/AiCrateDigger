# Frontend components

## Layout

```
components/
  ui/           Reusable visuals (vinyl backdrop, modals) — no search domain logic
  search/       Search page UI + thin client container (SearchExperience)
  listing/      Result cards and list wrapper
  dev/          DEBUG pipeline JSON inspector (non-production by default)
hooks/          Client state and side effects (useDigSearch, useRampProgress)
lib/            API client, DTO types, display/format helpers
```

## Data flow

```
app/page.tsx (RSC)
  └── SearchExperience (client container)
        ├── useDigSearch → lib/api postSearch
        ├── DigSearchForm, SearchHero, SearchExampleHints, …
        └── SearchResultsList → ListingResultCard → lib/listing-display
```

**Smart:** `SearchExperience` + `hooks/useDigSearch.ts` — query state, API call, errors, rate limit, fake progress.

**Dumb:** Everything under `ui/`, `listing/`, and presentational files in `search/` — props in, JSX out.

## Dev inspector

`components/dev/SearchDevInspector` renders when `isDevInspectorEnabled()` is true (`NODE_ENV !== production`, or `NEXT_PUBLIC_DEV_INSPECTOR=true`). Requires backend `DEBUG=true` for rich `debug.stages` payloads.
