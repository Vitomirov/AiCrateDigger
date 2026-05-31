FROM python:3.11-slim AS base

# Runtime: Postgres + Redis URLs come from the repo-root `.env` via docker-compose `env_file`.

ENV POETRY_VERSION=1.8.4 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

COPY pyproject.toml ./
RUN poetry install --no-interaction --no-ansi --no-root

COPY app ./app
COPY eval ./eval

EXPOSE 8000

# Local Compose: app code bind-mounted for live edits (see docker-compose.yml).
FROM base AS dev
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
