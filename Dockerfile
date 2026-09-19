# Single host-agnostic image: the same artefact runs on a laptop, the GPU box,
# or any public ingress. No host-specific assumptions.
#
# CPU-only by default so the image works anywhere. For GPU (local embeddings and
# cross-encoder reranking) install the CUDA build of torch and pass
# --gpus all plus CODEMEM_EMBED_DEVICE=cuda; see deploy/README.md.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CODEMEM_DATA_DIR=/data \
    CODEMEM_HOST=0.0.0.0 \
    CODEMEM_PORT=8080

WORKDIR /app

# Build toolchain only where needed; removed in the same layer.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Dependency layer first so source edits do not invalidate the wheel cache.
COPY pyproject.toml README.md ./
COPY src/codemem ./src/codemem
RUN pip install --upgrade pip \
 && pip install .

RUN mkdir -p /data && useradd --create-home --uid 10001 codemem \
 && chown -R codemem:codemem /app /data
USER codemem

VOLUME ["/data"]
EXPOSE 8080

# The contract requires an unauthenticated GET returning 2xx.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["python", "-m", "codemem"]
