"""Embedding adapter (plan.md section 3, dataset.md §11).

Independently configured from the chat model on purpose: plan.md §2.2 flags
whether the gpt-4o-mini rule also binds embeddings as *unverified*, so the
vector channel must be able to exist, fail, or be absent without touching
retrieval correctness.

Two invariants this module enforces:

* **Never raise into the request path.** Any provider problem returns ``None``;
  the caller degrades to lexical (plan.md §9 stage 1: "向量服务失效不破坏基础
  检索能力").
* **Cache by (content hash, model, dim).** dataset.md §11 requires this so
  B0/B1/B2 ablations replaying the same corpus do not re-bill the provider.

The cache is content-addressed and stores only vectors, never benchmark text.
"""

from __future__ import annotations

import hashlib
import json
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings, get_settings

EMBEDDING_VERSION = "embed-v1"


class ModelCallLimiter:
    """Bounds concurrent model calls across the process (FIXES.md P1.2).

    The service runs inside FastAPI's threadpool. Without a bound, N concurrent
    Add requests each doing a blocking embedding call occupy N workers, and a
    health check can find none free. This caps *model* concurrency while leaving
    the rest of the pool for cheap requests, and refuses to queue indefinitely:
    a caller that cannot get a slot within its timeout proceeds without the
    model, which is always safe because embeddings are an enhancement.
    """

    def __init__(self, limit: int, acquire_timeout_s: float) -> None:
        self._semaphore = threading.BoundedSemaphore(max(1, limit))
        self.acquire_timeout_s = acquire_timeout_s
        self._in_flight = 0
        self._lock = threading.Lock()
        self.rejected = 0

    def acquire(self, timeout_s: float | None = None) -> bool:
        timeout = self.acquire_timeout_s if timeout_s is None else timeout_s
        got = self._semaphore.acquire(timeout=max(0.0, timeout))
        with self._lock:
            if got:
                self._in_flight += 1
            else:
                self.rejected += 1
        return got

    def release(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
        self._semaphore.release()

    def stats(self) -> dict:
        with self._lock:
            return {"in_flight": self._in_flight, "rejected": self.rejected}


_limiter: ModelCallLimiter | None = None
_limiter_lock = threading.Lock()


def get_model_limiter(settings: Settings | None = None) -> ModelCallLimiter:
    global _limiter
    with _limiter_lock:
        if _limiter is None:
            cfg = settings or get_settings()
            _limiter = ModelCallLimiter(cfg.model_concurrency, cfg.model_acquire_timeout_s)
        return _limiter


def reset_model_limiter_for_tests() -> None:
    global _limiter
    with _limiter_lock:
        _limiter = None


class EmbeddingUnavailable(Exception):
    """Provider not configured -- callers must degrade, not fail."""


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    model: str
    dim: int
    usage: dict = field(default_factory=dict)
    elapsed_ms: int = 0
    cached: bool = False


def content_key(text: str, model: str, dim: int) -> str:
    hasher = hashlib.sha256()
    hasher.update(model.encode("utf-8"))
    hasher.update(b"\x1f")
    hasher.update(str(dim).encode("ascii"))
    hasher.update(b"\x1f")
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def pack_vector(vector: list[float]) -> bytes:
    """float32 little-endian blob, matching the storage column."""
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(blob: bytes, dim: int) -> list[float]:
    if len(blob) != dim * 4:
        raise ValueError(f"vector blob is {len(blob)} bytes, expected {dim * 4}")
    return list(struct.unpack(f"<{dim}f", blob))


class EmbeddingCache:
    """On-disk content-addressed cache (dataset.md §11).

    Kept as a single JSONL file keyed by the content hash so ablation reruns
    are cheap and the cache is trivially inspectable. Vectors are stored as
    plain lists; the file holds no source text.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._memory: dict[str, list[float]] = {}
        self._loaded = False
        self.hits = 0
        self.misses = 0

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                self._memory[row["k"]] = row["v"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    def get(self, key: str) -> list[float] | None:
        self._load()
        found = self._memory.get(key)
        if found is None:
            self.misses += 1
        else:
            self.hits += 1
        return found

    def put(self, key: str, vector: list[float]) -> None:
        self._load()
        if key in self._memory:
            return
        self._memory[key] = vector
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"k": key, "v": vector}, separators=(",", ":")) + "\n")

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._memory)}


class EmbeddingClient:
    """[OI]-compatible embeddings client with a bounded, degrading interface."""

    def __init__(self, settings: Settings | None = None, cache: EmbeddingCache | None = None) -> None:
        self.settings = settings or get_settings()
        self.cache = cache
        self._client = None
        # Call accounting, so callers can report actual activity instead of
        # asserting it (FIXES.md P2.7). Distinguishing these matters: a run
        # where every call timed out attempted work but completed none.
        self.calls_attempted = 0  # network attempts, including failures
        self.calls_made = 0       # successful provider responses
        self.calls_failed = 0     # attempts that raised
        self.cache_hits = 0       # texts served from cache without a call

    @property
    def configured(self) -> bool:
        return bool(self.settings.embed_base_url and self.settings.embed_api_key and self.settings.embed_model)

    @property
    def available(self) -> bool:
        return self.configured

    @property
    def dim(self) -> int:
        return self.settings.embed_dim

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.configured:
            raise EmbeddingUnavailable("embedding endpoint is not configured")
        import openai

        self._client = openai.OpenAI(
            base_url=self.settings.embed_base_url,
            api_key=self.settings.embed_api_key,
            timeout=self.settings.embed_timeout_s,
            max_retries=self.settings.embed_max_retries,
        )
        return self._client

    def embed(self, texts: list[str], *, timeout_s: float | None = None) -> EmbeddingResult | None:
        """Embed a batch, returning ``None`` on any failure.

        Cached entries are served without a call. Uncached texts go out in one
        request; a failure returns ``None`` rather than raising.
        """
        if not texts:
            return EmbeddingResult(vectors=[], model=self.settings.embed_model, dim=self.dim)

        keys = [content_key(t, self.settings.embed_model, self.dim) for t in texts]
        vectors: list[list[float] | None] = [None] * len(texts)
        pending: list[int] = []
        if self.cache is not None:
            for i, key in enumerate(keys):
                cached = self.cache.get(key)
                if cached is not None:
                    vectors[i] = cached
                    self.cache_hits += 1
                else:
                    pending.append(i)
        else:
            pending = list(range(len(texts)))

        if not pending:
            return EmbeddingResult(
                vectors=[v for v in vectors if v is not None],
                model=self.settings.embed_model,
                dim=self.dim,
                cached=True,
            )

        try:
            client = self._ensure_client()
        except EmbeddingUnavailable:
            return None

        # Bound concurrent model calls so a burst of slow Adds cannot occupy
        # every threadpool worker (FIXES.md P1.2). Failing to acquire a slot is
        # not an error: we simply return None and the caller stays lex-only.
        limiter = get_model_limiter(self.settings)
        if not limiter.acquire():
            return None

        # Attempted counts every network attempt, including ones that then
        # fail. A failed call still consumed a request, so a manifest that
        # reports "0 calls" during a provider outage would be wrong
        # (FIXES.md P2.7 follow-up).
        started = time.monotonic()
        self.calls_attempted += 1
        try:
            response = client.embeddings.create(
                model=self.settings.embed_model,
                input=[texts[i] for i in pending],
                timeout=timeout_s or self.settings.embed_timeout_s,
            )
        except Exception:
            self.calls_failed += 1
            return None
        finally:
            limiter.release()
        self.calls_made += 1
        elapsed = int((time.monotonic() - started) * 1000)

        for position, datum in enumerate(response.data):
            target = pending[position]
            vec = list(datum.embedding)
            vectors[target] = vec
            if self.cache is not None:
                self.cache.put(keys[target], vec)

        if any(v is None for v in vectors):
            return None

        usage = {}
        if getattr(response, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
        return EmbeddingResult(
            vectors=[v for v in vectors if v is not None],
            model=getattr(response, "model", self.settings.embed_model),
            dim=self.dim,
            usage=usage,
            elapsed_ms=elapsed,
        )

    def stats(self) -> dict:
        """Call accounting for reporting (FIXES.md P2.7).

        ``attempted`` is the honest answer to "did this run touch the model":
        it counts network attempts including failures, so a provider outage
        still reports non-zero activity.
        """
        return {
            "attempted": self.calls_attempted,
            "succeeded": self.calls_made,
            "failed": self.calls_failed,
            "cache_hits": self.cache_hits,
        }

    def probe(self, *, live: bool = False) -> dict:
        """Verify configuration; ``live=False`` sends nothing (zero cost)."""
        report = {
            "configured": self.configured,
            "base_url_set": bool(self.settings.embed_base_url),
            "api_key_set": bool(self.settings.embed_api_key),
            "model": self.settings.embed_model or None,
            "dim": self.dim,
            "live": live,
        }
        if not self.configured:
            report["status"] = "not_configured"
            report["detail"] = "vector channel disabled; retrieval runs lexical-only"
            return report
        if not live:
            report["status"] = "configured"
            report["detail"] = "local checks passed; pass live=True to send one minimal request"
            return report
        result = self.embed(["connectivity probe"])
        if result is None:
            report["status"] = "live_failed"
            report["detail"] = "embedding call failed; see server logs"
            return report
        report["status"] = "live_ok"
        report["elapsed_ms"] = result.elapsed_ms
        report["returned_dim"] = len(result.vectors[0]) if result.vectors else 0
        if result.vectors and len(result.vectors[0]) != self.dim:
            report["dim_warning"] = (
                f"provider returned {len(result.vectors[0])} dims, "
                f"CODEMEM_EMBED_DIM is {self.dim}"
            )
        return report


_client: EmbeddingClient | None = None


def get_embedder(settings: Settings | None = None, cache_path: Path | None = None) -> EmbeddingClient:
    global _client
    if _client is None or settings is not None:
        cache = EmbeddingCache(cache_path) if cache_path else None
        _client = EmbeddingClient(settings, cache)
    return _client


def peek_embedder() -> EmbeddingClient | None:
    """Return the current embedder without creating one.

    Lets callers report actual call counts without side effects.
    """
    return _client


def reset_embedder_for_tests() -> None:
    global _client
    _client = None