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
 && pip install ".[dense]"

# Pre-download the optional models at build time.
#
# This must happen during the build, not at first request: model loading does a
# network check against huggingface.co, and a deployment host that cannot reach
# it stalls for minutes on retries, which would block Add and blow the contract's
# latency budget. Baking the weights in also means runtime works offline.
# Failures are non-fatal: without the models the service still runs on its
# deterministic lexical and identifier channels.
ARG CODEMEM_EMBED_MODEL=BAAI/bge-small-en-v1.5
ARG CODEMEM_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ENV HF_HOME=/opt/hf
RUN python - <<'PY' || echo "model pre-download skipped; dense/rerank will be unavailable"
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer(os.environ.get("CODEMEM_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
CrossEncoder(os.environ.get("CODEMEM_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
print("models cached")
PY

RUN mkdir -p /data && useradd --create-home --uid 10001 codemem \
 && chown -R codemem:codemem /app /data /opt/hf
USER codemem

VOLUME ["/data"]
EXPOSE 8080

# The contract requires an unauthenticated GET returning 2xx.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["python", "-m", "codemem"]
