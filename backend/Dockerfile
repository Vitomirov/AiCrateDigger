FROM python:3.11-slim

# Runtime: `asyncpg` wheels are prebuilt; SQLAlchemy talks to Postgres over the Docker network.
# Set `DATABASE_URL=postgresql+asyncpg://USER:PASS@db:5432/DB` (see docker-compose).

ENV POETRY_VERSION=1.8.4 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

COPY pyproject.toml ./
RUN poetry install --no-interaction --no-ansi --no-root

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
