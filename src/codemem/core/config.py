"""Runtime configuration.

Every tunable is settable through an environment variable so the Docker image
stays host-agnostic: the same image runs locally on CPU, on the GPU box, or
behind any public ingress.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


def _env_str(name: str, default: str | None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    # ---- storage -------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    db_filename: str = "codemem.sqlite3"

    # ---- transport -----------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8080
    # When unset, authentication is disabled (allowed for public smoke only).
    api_key: str | None = None
    log_level: str = "INFO"
    # Do not log raw memory text; see docs/DESIGN.md data-handling section.
    log_request_bodies: bool = False

    # ---- contract limits ----------------------------------------------
    # Platform sends top_k=100. We accept more but always clamp the response.
    max_top_k: int = 1000
    max_messages_per_add: int = 5000
    max_content_chars: int = 4_000_000
    add_deadline_seconds: float = 1500.0  # 25 min internal guard (< 30 min)

    # ---- chunking ------------------------------------------------------
    target_chunk_tokens: int = 320
    max_chunk_tokens: int = 1400
    # Code fences are never cut mid-line; this is the hard cap that forces a
    # split at line boundaries for pathological blobs.
    hard_chunk_chars: int = 24_000
    chunk_overlap_tokens: int = 32

    # ---- evidence assembly --------------------------------------------
    evidence_full_count: int = 8  # items rendered in full form
    evidence_item_tokens: int = 380  # cap for a full-form item
    evidence_ptr_tokens: int = 110  # cap for a pointer-form item
    evidence_excerpt_chars: int = 900
    # Cap on items returned from one session. A session yields many chunks, and
    # without a cap they monopolise the ranked list: measured on the proxy
    # benchmark, ~100 returned chunks collapsed to ~23 distinct sessions, so
    # other relevant prior work never got a slot. Coverage depends on session
    # diversity, because a task is answered from a session, not from one chunk.
    max_evidence_per_session: int = 3
    # Total budget for the whole `data` payload (platform input window minus
    # answer/safety reservation; kept well under 117_760 on purpose).
    evidence_budget_tokens: int = 60_000
    # Noise gate: drop candidates below this normalised score. Prevents
    # filling the token prefix with same-repo distractors. Calibrated with the
    # local benchmark (see eval/), not by intuition.
    min_evidence_score: float = 0.15
    # Always return at least this many if any candidate clears the gate.
    min_evidence_count: int = 1

    # ---- retrieval -----------------------------------------------------
    rrf_k: int = 60
    recall_per_channel: int = 120
    candidate_pool: int = 300

    # ---- llm (enrichment / query understanding; Add+Search share it) ----
    llm_enabled: bool = False
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 60.0
    enrich_ratio: float = 0.25  # fraction of chunks eligible for enrichment

    # ---- dense retrieval ------------------------------------------------
    dense_enabled: bool = False
    embed_backend: str = "local"  # local | openai | none
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024
    embed_device: str = "cpu"
    embed_batch_size: int = 16

    # ---- rerank ---------------------------------------------------------
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_n: int = 120

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Settings":
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}

        def put(name: str, value: Any) -> None:
            if name in known:
                kwargs[name] = value

        put("data_dir", Path(_env_str("CODEMEM_DATA_DIR", "./data") or "./data"))
        put("db_filename", _env_str("CODEMEM_DB_FILENAME", "codemem.sqlite3"))
        put("host", _env_str("CODEMEM_HOST", "0.0.0.0"))
        put("port", _env_int("CODEMEM_PORT", 8080))
        put("api_key", _env_str("CODEMEM_API_KEY", None))
        put("log_level", (_env_str("CODEMEM_LOG_LEVEL", "INFO") or "INFO").upper())
        put("log_request_bodies", _env_bool("CODEMEM_LOG_REQUEST_BODIES", False))
        put("max_top_k", _env_int("CODEMEM_MAX_TOP_K", 1000))
        put("add_deadline_seconds", _env_float("CODEMEM_ADD_DEADLINE_SECONDS", 1500.0))
        put("target_chunk_tokens", _env_int("CODEMEM_TARGET_CHUNK_TOKENS", 320))
        put("max_chunk_tokens", _env_int("CODEMEM_MAX_CHUNK_TOKENS", 1400))
        put("hard_chunk_chars", _env_int("CODEMEM_HARD_CHUNK_CHARS", 24_000))
        put("evidence_full_count", _env_int("CODEMEM_EVIDENCE_FULL_COUNT", 8))
        put(
            "max_evidence_per_session",
            _env_int("CODEMEM_MAX_EVIDENCE_PER_SESSION", 3),
        )
        put("evidence_item_tokens", _env_int("CODEMEM_EVIDENCE_ITEM_TOKENS", 380))
        put("evidence_ptr_tokens", _env_int("CODEMEM_EVIDENCE_PTR_TOKENS", 110))
        put("evidence_budget_tokens", _env_int("CODEMEM_EVIDENCE_BUDGET_TOKENS", 60_000))
        put("min_evidence_score", _env_float("CODEMEM_MIN_EVIDENCE_SCORE", 0.15))
        put("min_evidence_count", _env_int("CODEMEM_MIN_EVIDENCE_COUNT", 1))
        put("rrf_k", _env_int("CODEMEM_RRF_K", 60))
        put("recall_per_channel", _env_int("CODEMEM_RECALL_PER_CHANNEL", 120))
        put("candidate_pool", _env_int("CODEMEM_CANDIDATE_POOL", 300))
        put("llm_enabled", _env_bool("CODEMEM_LLM_ENABLED", False))
        put("llm_base_url", _env_str("CODEMEM_LLM_BASE_URL", None))
        put("llm_api_key", _env_str("CODEMEM_LLM_API_KEY", None))
        put("llm_model", _env_str("CODEMEM_LLM_MODEL", "gpt-4o-mini"))
        put("enrich_ratio", _env_float("CODEMEM_ENRICH_RATIO", 0.25))
        put("dense_enabled", _env_bool("CODEMEM_DENSE_ENABLED", False))
        put("embed_backend", _env_str("CODEMEM_EMBED_BACKEND", "local"))
        put("embed_model", _env_str("CODEMEM_EMBED_MODEL", "BAAI/bge-m3"))
        put("embed_device", _env_str("CODEMEM_EMBED_DEVICE", "cpu"))
        put("rerank_enabled", _env_bool("CODEMEM_RERANK_ENABLED", False))
        put("rerank_model", _env_str("CODEMEM_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
        return cls(**kwargs)
