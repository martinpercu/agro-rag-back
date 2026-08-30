FROM python:3.13-slim

# uv from official image (requires BuildKit)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# System deps for healthcheck (curl) and for pdfplumber/pypdf (no extra)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Deps first for layer cache
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# App source (no data/vector, no .venv, no .env)
COPY src/ src/
COPY scripts/ scripts/
COPY data/raw/ data/raw/
COPY README.md ./
COPY alembic/ alembic/
COPY alembic.ini ./

EXPOSE 8002

# Railway injects $PORT and DATABASE_URL; run migrations then start
CMD ["sh", "-c", "uv run alembic upgrade head 2>&1 | head -20; uv run uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8002} --app-dir src"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8002}/ || exit 1
