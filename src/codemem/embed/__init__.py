"""Dense retrieval: embeddings and a per-user vector index.

Design constraints that shape this module
-----------------------------------------
**Isolation.** Vectors are stored in SQLite keyed by ``user_id`` and searched
within a single user only. There is no global index to accidentally search
across users, which matters because a dense channel is otherwise the easiest
place to leak: nearest-neighbour over a shared index would return other users'
memory regardless of any WHERE clause applied afterwards.

**Degradation.** The service must run with no model at all — embeddings are an
optional channel, and any load or encode failure disables the channel rather
than failing a request.

**Backends.** ``local`` uses sentence-transformers, which needs no network and
no API key (the recommended path, and the only one that satisfies the Coding
track's "same model" reproduction requirement without depending on a relay).
``openai`` is a drop-in for deployments that prefer a hosted encoder.

Vectors are stored L2-normalized, so inner product equals cosine similarity.
"""

from __future__ import annotations

import array
import math
import os
import threading
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..core.config import Settings
from ..core.logging import get_logger

log = get_logger("codemem.embed")


def prepare_offline_env(offline: bool = True) -> None:
    """Force HuggingFace libraries into offline mode, robustly.

    Setting the environment variables alone is not enough: ``huggingface_hub``
    reads them into module-level constants *at import time*, so if anything has
    already imported it (transformers, another library, a host application) the
    later env change is ignored and every model load attempts a network check.
    Where the Hub is unreachable that turns a fail-fast error into a stall of
    tens of seconds — measured at 19 s for a missing model — which would block
    Add and blow the contract's latency budget.

    So: set the variables, and patch the already-imported constants too.
    """
    if not offline:
        return
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["USE_TF"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:  # patch constants that were captured before the variables were set
        import huggingface_hub.constants as hub_constants

        hub_constants.HF_HUB_OFFLINE = True
    except Exception:
        pass


def _to_blob(values: Sequence[float]) -> bytes:
    return array.array("f", values).tobytes()


def _from_blob(blob: bytes) -> list[float]:
    arr = array.array("f")
    arr.frombytes(blob)
    return list(arr)


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-12:
        return values
    return [v / norm for v in values]


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` to an actual torch device.

    ``auto`` means "use the GPU if this host has one, otherwise CPU". It is the
    default because the same image must be correct on a CPU-only deployment host
    and fast on a GPU one; measured cost of getting this wrong is ~8x on both
    models. An explicit device is honoured as given, so a misconfiguration is
    visible rather than silently downgraded.
    """
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def model_in_cache(model_name: str) -> bool:
    """Fast local-cache probe for a model.

    Needed because ``SentenceTransformer(..., local_files_only=True)`` does not
    honour the flag for every lookup: on a missing model it still attempts
    network calls and takes ~20 s to fail, versus 0.07 s for a direct
    ``snapshot_download`` probe. Twenty seconds of stall on a misconfigured
    deployment is worth avoiding, so we check first and fail fast.

    Returns True when the probe cannot run at all, so a missing
    huggingface_hub never blocks a model that would otherwise load.
    """
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_name, local_files_only=True)
        return True
    except Exception as exc:
        name = type(exc).__name__
        if name in ("LocalEntryNotFoundError", "EntryNotFoundError", "RepositoryNotFoundError"):
            return False
        # Unknown failure (offline library, permissions): let the loader decide.
        return True


@dataclass(slots=True)
class Encoder:
    """A lazily-loaded text encoder.

    Loading is deferred and guarded so an unavailable model degrades the system
    to its lexical channels instead of breaking startup.
    """

    backend: str
    model_name: str
    device: str
    batch_size: int
    offline: bool = True
    _model: object | None = None
    _dim: int = 0
    _failed: bool = False
    _lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._lock is None:
            self._lock = threading.Lock()

    # ------------------------------------------------------------ loading --

    @property
    def available(self) -> bool:
        if self.backend == "none" or self._failed:
            return False
        if self._model is not None:
            return True
        return self._load()

    def _load(self) -> bool:
        with self._lock:
            if self._model is not None:
                return True
            if self._failed:
                return False
            try:
                if self.backend == "local":
                    self._model = self._load_local()
                elif self.backend == "openai":
                    self._model = self._load_openai()
                else:
                    log.warning("unknown embed backend", extra={"ctx": {"backend": self.backend}})
                    self._failed = True
                    return False
                log.info(
                    "encoder ready",
                    extra={
                        "ctx": {
                            "backend": self.backend,
                            "model": self.model_name,
                            "dim": self._dim,
                            "device": self.device,
                        }
                    },
                )
                return True
            except Exception as exc:
                log.warning(
                    "encoder unavailable; dense channel disabled",
                    extra={"ctx": {"backend": self.backend, "error": str(exc)[:300]}},
                )
                self._failed = True
                return False

    def _load_local(self):
        # Prepare the environment BEFORE importing sentence-transformers. Two
        # problems are fixed here, both of which otherwise hang or crash the
        # service on a deployment host:
        #
        # 1. TensorFlow interop in transformers raises on a Keras version
        #    conflict. This model is pure PyTorch, so disable the TF path.
        # 2. Model loading performs a network HEAD check against
        #    huggingface.co. Where that is unreachable it retries with backoff
        #    and stalls for tens of seconds, blocking Add. Offline mode uses the
        #    local cache and fails fast when the model is genuinely absent.
        prepare_offline_env(self.offline)
        if not self.offline:
            os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
            os.environ.setdefault("USE_TF", "0")
        if self.offline and not model_in_cache(self.model_name):
            raise FileNotFoundError(
                f"{self.model_name} is not in the local cache and offline mode is on"
            )

        self.device = resolve_device(self.device)

        from sentence_transformers import SentenceTransformer

        try:
            model = SentenceTransformer(
                self.model_name,
                device=self.device,
                local_files_only=self.offline,
            )
        except TypeError:
            # Older sentence-transformers without the local_files_only kwarg.
            model = SentenceTransformer(self.model_name, device=self.device)
        getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension"
        )
        self._dim = int(getter() or 0)
        return model

    def _load_openai(self):
        import os

        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEMEM_LLM_API_KEY"),
            base_url=os.environ.get("CODEMEM_EMBED_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL"),
        )
        return client

    # ------------------------------------------------------------ encoding --

    @property
    def dim(self) -> int:
        if self._dim:
            return self._dim
        # OpenAI dimensions are not discoverable before the first call.
        return 1536 if self.backend == "openai" else 0

    def encode(self, texts: Sequence[str]) -> list[list[float]] | None:
        """Encode texts to L2-normalized vectors, or None if unavailable."""
        if not texts:
            return []
        if not self.available:
            return None
        try:
            if self.backend == "local":
                vectors = self._model.encode(  # type: ignore[union-attr]
                    list(texts),
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return [[float(x) for x in row] for row in vectors]
            if self.backend == "openai":
                response = self._model.embeddings.create(  # type: ignore[union-attr]
                    model=self.model_name, input=list(texts)
                )
                out = []
                for item in sorted(response.data, key=lambda d: d.index):
                    out.append(_normalize([float(x) for x in item.embedding]))
                return out
        except Exception as exc:
            log.warning(
                "encode failed",
                extra={"ctx": {"backend": self.backend, "error": str(exc)[:300]}},
            )
            return None
        return None


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; inputs are already normalized, so this is a dot product."""
    return sum(x * y for x, y in zip(a, b))


class Instance:
    """An embedding service, cached per configuration.

    The cache is keyed on the settings that affect behaviour, not on the call
    order. An unkeyed process-wide singleton silently returns a stale instance
    when configuration changes — which made an A/B measurement report identical
    metrics for two different configurations, because the second run reused the
    first run's disabled encoder.
    """

    _cache: dict[tuple, "Instance"] = {}
    _cache_lock = threading.Lock()

    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.dense_enabled
        self.encoder = Encoder(
            backend=settings.embed_backend,
            model_name=settings.embed_model,
            device=settings.embed_device,
            batch_size=settings.embed_batch_size,
            offline=settings.embed_offline,
        )

    @classmethod
    def _key(cls, settings: Settings) -> tuple:
        return (
            settings.dense_enabled,
            settings.embed_backend,
            settings.embed_model,
            settings.embed_device,
            settings.embed_batch_size,
            settings.embed_offline,
        )

    @classmethod
    def get(cls, settings: Settings) -> "Instance":
        key = cls._key(settings)
        with cls._cache_lock:
            instance = cls._cache.get(key)
            if instance is None:
                instance = cls(settings)
                cls._cache[key] = instance
            return instance

    @classmethod
    def reset(cls) -> None:
        """Drop cached instances (used by tests and between configurations)."""
        with cls._cache_lock:
            cls._cache.clear()

    @property
    def available(self) -> bool:
        return self.enabled and self.encoder.available

    def embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        if not self.enabled:
            return None
        return self.encoder.encode(texts)


def pack(vec: Sequence[float]) -> bytes:
    return _to_blob(vec)


def unpack(blob: bytes) -> list[float]:
    return _from_blob(blob)


def batched(items: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
