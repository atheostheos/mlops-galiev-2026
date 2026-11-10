# ---------- стадия сборки ----------
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535 AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-service.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-service.txt

# ---------- финальный образ ----------
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535 AS runtime

RUN useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY --from=builder /install /usr/local

COPY src/ ./src/
COPY params.yaml ./params.yaml
COPY models/ ./models/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health').json()['model_loaded'] else 1)"

CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "src/uvicorn_logging.json"]
