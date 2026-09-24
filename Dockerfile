# API and worker image. Training runs elsewhere (see CLAUDE.md), so this uses
# CPU-only PyTorch from the lockfile.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first so code changes don't reinstall them.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
COPY migrations ./migrations
COPY alembic.ini ./
RUN uv sync --locked --no-dev && useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
CMD ["wildinbox", "api", "--host", "0.0.0.0", "--port", "8000"]
