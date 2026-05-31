# Deployment

AiCrateDigger ships with two Docker Compose configurations: **development** (hot reload, exposed ports) and **production** (hardened, frontend-only public port).

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Docker | 20+ |
| Docker Compose | v2 |
| API keys | OpenAI + Tavily |

---

## Development environment

### Quick start

```bash
git clone <repository-url>
cd AiCrateDigger
cp .env.example .env
# Edit .env: set OPENAI_API_KEY and TAVILY_API_KEY
docker compose up --build
```

Open **http://localhost:3000** in the browser. The backend is not published to the host by default (internal Docker network only).

### Services

| Service | Image / build | Ports | Notes |
|---------|---------------|-------|-------|
| `db` | postgres:15-alpine | 5433→5432 | Volume `pgdata` |
| `redis` | redis:7-alpine | 6379 | Volume `redisdata` |
| `backend` | `./backend` target `dev` | expose 8000 | Bind mount `./backend/app:/app/app` |
| `frontend` | `./frontend` target `dev` | 3000→3000 | Bind mount `./frontend:/app` |

All services load repo-root `.env`.

### Optional: expose backend to host

Uncomment in `docker-compose.yml`:

```yaml
ports:
  - "127.0.0.1:8000:8000"
```

Then access Swagger at http://localhost:8000/docs.

### Rebuild after dependency changes

```bash
docker compose up --build
```

### Stop and remove containers

```bash
docker compose down
```

Remove volumes (destructive — deletes DB and Redis data):

```bash
docker compose down -v
```

---

## Production environment

### Checklist

Before deploying, verify [Configuration → Production profile](./configuration.md#production):

- [ ] `APP_ENV=production`
- [ ] `DEBUG=false`
- [ ] Strong `INTERNAL_API_SECRET` (same on backend and frontend)
- [ ] `DATABASE_URL` and `REDIS_URL` set
- [ ] Real `FRONTEND_PUBLIC_URL` (HTTPS)
- [ ] `LOG_FORMAT=json`
- [ ] Rate limits and quotas fail-closed
- [ ] No `NEXT_PUBLIC_DEV_INSPECTOR`

### Deploy

```bash
cp .env.example .env
# Configure production values
docker compose -f docker-compose.prod.yml up --build -d
```

### Production topology

| Service | Differences from dev |
|---------|---------------------|
| `db`, `redis` | No host ports; `restart: unless-stopped` |
| `backend` | Target `production`; 2 Uvicorn workers; `APP_ENV=production` |
| `frontend` | Target `production`; baked Next.js build; only public port |
| `eval` | Not included |

Only `${FRONTEND_PORT:-3000}` is published. Backend communicates over internal Docker network.

### Verify deployment

```bash
# Frontend health (via SSR)
curl -s http://localhost:3000 | head

# Search via BFF
curl -s -X POST http://localhost:3000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}' | jq .reason
```

Backend `/health` is not publicly reachable in production Compose — probe via frontend or Docker exec.

### Logs

```bash
docker compose -f docker-compose.prod.yml logs -f backend
docker compose -f docker-compose.prod.yml logs -f frontend
```

---

## Local development without Docker

For faster iteration on backend or frontend individually.

### Backend (Poetry)

**Prerequisites:** Python 3.11, Poetry, running Postgres and Redis.

```bash
cd backend
poetry install

export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...
export DATABASE_URL=postgresql+asyncpg://aicratedigger:aicratedigger@localhost:5433/aicratedigger
export REDIS_URL=redis://localhost:6379/0

poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Start Postgres/Redis via Compose (db + redis services only):

```bash
docker compose up db redis -d
```

### Frontend (npm)

**Prerequisites:** Node.js 18+.

```bash
cd frontend
npm ci

export BACKEND_URL=http://localhost:8000
export NEXT_PUBLIC_BACKEND_URL=http://localhost:8000

npm run dev
```

Open http://localhost:3000.

---

## Docker image details

### Backend Dockerfile

| Stage | Base | Workers | Use |
|-------|------|---------|-----|
| `dev` | python:3.11-slim | 1 | Local Compose |
| `production` | python:3.11-slim | 2 | Production Compose |

Poetry 1.8.4, `POETRY_VIRTUALENVS_CREATE=false`. Copies `app/` and `eval/`.

### Frontend Dockerfile

| Stage | Base | Use |
|-------|------|-----|
| `dev` | node:20-alpine | Hot reload |
| `production` | node:20-alpine | `next build` + `next start` |

---

## Evaluation profile

Run offline pipeline evaluation inside Compose:

```bash
docker compose --profile eval run --rm eval
```

See [Evaluation](./evaluation.md).

---

## Reverse proxy (production)

The Compose production file exposes the frontend directly. For HTTPS and domain routing, place a reverse proxy in front:

```nginx
# Example nginx snippet
server {
    listen 443 ssl;
    server_name your-domain.example;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Set `FRONTEND_PUBLIC_URL=https://your-domain.example`.

The BFF forwards `X-Forwarded-For` to the backend for rate limiting. Ensure your proxy sets these headers correctly.

---

## Scaling considerations

Current architecture assumptions:

| Component | Scaling note |
|-----------|--------------|
| Backend | Stateless; multiple Uvicorn workers supported. Horizontal scaling requires shared Redis and Postgres |
| Frontend | Stateless Next.js; scale behind load balancer |
| Redis | Single instance; required for cache, rate limits, quotas |
| Postgres | Single instance; no read replicas configured |
| Tavily / OpenAI | External rate limits; global daily quotas protect spend |

No Kubernetes manifests or cloud-specific IaC are included in the repository.

---

## Backup and recovery

| Data | Location | Backup approach |
|------|----------|-----------------|
| Store catalogue | Postgres `pgdata` volume | `pg_dump` |
| Search cache | Redis + Postgres | Regenerates on miss; backup optional |
| Application code | Git repository | Standard git workflow |

```bash
# Postgres backup (dev Compose)
docker compose exec db pg_dump -U aicratedigger aicratedigger > backup.sql
```

---

## Upgrade procedure

1. Pull latest code
2. Review `.env.example` for new variables
3. Rebuild images: `docker compose -f docker-compose.prod.yml up --build -d`
4. Verify health and run eval profile if pipeline changed
5. Monitor logs for startup guard or migration errors

Schema changes apply automatically via `create_all` and inline alters on backend startup. Review backend startup logs after upgrades.
