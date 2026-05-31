# AiCrateDigger Documentation

Technical documentation for **AiCrateDigger** — an AI-assisted search system for physical music (vinyl, CD, cassette). Users submit natural-language queries; the system parses intent, searches the web via Tavily within a curated store whitelist, and extracts structured listing rows from search snippets.

This documentation is organized for engineers onboarding to the codebase, operating a deployment, or extending the pipeline.

**Live demo:** [https://aicratedigger.dejanvitomirov.com/](https://aicratedigger.dejanvitomirov.com/)

---

## Documentation index

| Document | Audience | Contents |
|----------|----------|----------|
| [Overview](./overview.md) | All | Product purpose, capabilities, design principles, repository layout |
| [Architecture](./architecture.md) | Backend, full-stack | System diagram, request lifecycle, pipeline stages, data flow |
| [Backend](./backend.md) | Backend | FastAPI structure, domains, search pipeline, engine modules |
| [Frontend](./frontend.md) | Frontend | Next.js App Router, BFF pattern, components, styling |
| [Database](./database.md) | Backend, ops | PostgreSQL schema, Redis usage, cache key design |
| [API Reference](./api.md) | Integrators | Endpoints, request/response schemas, error codes |
| [Configuration](./configuration.md) | Dev, ops | Environment variables, settings hierarchy, pipeline knobs |
| [Deployment](./deployment.md) | Dev, ops | Docker Compose (dev & prod), local development without Docker |
| [Testing](./testing.md) | Backend | Unit test suite, patterns, CI recommendations |
| [Evaluation](./evaluation.md) | Backend, QA | Offline pipeline eval harness, dataset format, CLI usage |
| [Security](./security.md) | Backend, ops | Production guard, BFF auth, rate limits, quotas |
| [Operations](./operations.md) | Ops | Logging, health checks, troubleshooting, cost controls |

---

## Quick links

- **Live app:** [https://aicratedigger.dejanvitomirov.com/](https://aicratedigger.dejanvitomirov.com/)
- **Run locally:** [Deployment → Development](./deployment.md#development-environment)
- **Production checklist:** [Deployment → Production](./deployment.md#production-environment)
- **API contract:** [API Reference](./api.md)
- **All env vars:** [Configuration](./configuration.md)

---

## Related material

- Root [README.md](../README.md) — portfolio-oriented project summary and quick start
- [.env.example](../.env.example) — annotated environment template
- [.cursorrules](../.cursorrules) — agent and contributor conventions for this repository

---

## Version & status

| Component | Version / note |
|-----------|----------------|
| Backend API | `0.1.0` (see `app/main.py`) |
| Python | 3.11+ |
| Node.js | 20 (Docker); 18+ supported for local Next.js |
| Database migrations | None (Alembic not adopted; schema via `create_all` + inline alters) |
| Frontend tests | Not implemented |
| CI/CD | Not configured in repository |

This is a portfolio-stage project with a [public deployment](https://aicratedigger.dejanvitomirov.com/). Expect evolving APIs and operational hardening rather than semver guarantees on internal modules.
