# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.8 AS uv

FROM python:3.12-slim AS builder
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /build
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable \
    --extra embeddings \
    --extra embeddings-cloud \
    --extra llm \
    --extra public-demo

FROM python:3.12-slim AS runtime
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /build/.venv /app/.venv
COPY --chown=appuser:appuser deployment/assets/models/embeddinggemma-300m /app/models/embeddinggemma-300m
COPY --chown=appuser:appuser data/uni_rag.sqlite /app/seed-data/uni_rag.sqlite
COPY --chown=appuser:appuser data/indexes/vector /data/indexes/vector
COPY --chown=appuser:appuser deployment/entrypoint.sh /app/entrypoint.sh
RUN chmod 0555 /app/entrypoint.sh \
    && mkdir -p /app/Courses /data/extracted /data/runs \
    && chown -R appuser:appuser /data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/app/models/huggingface-cache \
    UNI_RAG_HOSTED_MODE=true \
    UNI_RAG_COURSES_ROOT=/app/Courses \
    UNI_RAG_DATA_DIR=/data \
    UNI_RAG_SQLITE_PATH=/data/uni_rag.sqlite \
    UNI_RAG_CHROMA_DIR=/data/indexes/vector \
    UNI_RAG_RUNS_DIR=/data/runs \
    UNI_RAG_EMBEDDINGGEMMA_MODEL_PATH=/app/models/embeddinggemma-300m

USER appuser
EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
