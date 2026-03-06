FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.10.8@sha256:88234bc9e09c2b2f6d176a3daf411419eb0370d450a08129257410de9cfafd2a /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src/ ./src/

RUN uv sync --package data-pipeline --no-dev --no-editable


FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

RUN useradd -r -s /bin/false appuser && chown -R appuser /app
USER appuser

EXPOSE 80

CMD ["/app/.venv/bin/uvicorn", "data_pipeline.webhook:app", "--host", "0.0.0.0", "--port", "80"]
