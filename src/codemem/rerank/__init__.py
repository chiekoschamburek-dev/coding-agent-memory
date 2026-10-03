"""Cross-encoder reranking.

The recall channels score query and memory independently (BM25, embeddings) or
by exact identifier overlap. A cross-encoder reads the pair *jointly*, which is
what lets it distinguish the two cases that matter most in this track:

    "IndexError in tokenizer.py, guarded the empty buffer"   -> relevant
    "reformatted tokenizer.py for the linter"                -> not relevant

Both share every identifier, so the recall channels cannot separate them; only
joint reading can. That is the whole reason this stage exists.

Reranking is a *scoring* operation over memory already stored during Add. It
produces numbers, never text, so it does not violate the rule that Search must
not generate content.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Sequence

from ..core.config import Settings
from ..core.logging import get_logger

log = get_logger("codemem.rerank")


@dataclass
class Reranker:
    """A lazily-loaded cross-encoder.

    Loading is deferred and failures disable reranking rather than breaking the
    request, matching the dense channel's degradation contract.
    """

    model_name: str
    device: str = "cpu"
    offline: bool = True
    max_length: int = 512
    # Characters are a poor proxy for cost: measured on MiniLM, 2 048 characters
    # can be 630 tokens while 40 characters is 21, and latency scales with real
    # token count (4 ms/doc at 21 tokens, 48 ms/doc at 1 034). So the document is
    # clipped with the model's own tokenizer, to this budget.
    doc_token_budget: int = 200
    _model: object | None = None
    _failed: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def available(self) -> bool:
        if self._failed:
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
                from ..embed import model_in_cache, prepare_offline_env

                prepare_offline_env(self.offline)
                if not self.offline:
                    import os

                    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
                    os.environ.setdefault("USE_TF", "0")
                # Same fast-path probe as the embedder: the loader's own
                # local_files_only does not short-circuit every lookup, so a
                # missing model would otherwise stall for ~20 s.
                if self.offline and not model_in_cache(self.model_name):
                    raise FileNotFoundError(
                        f"{self.model_name} is not in the local cache and "
                        "offline mode is on"
                    )

                from ..embed import resolve_device

                self.device = resolve_device(self.device)

                from sentence_transformers import CrossEncoder

                try:
                    self._model = CrossEncoder(
                        self.model_name,
                        device=self.device,
                        local_files_only=self.offline,
                        max_length=self.max_length,
                    )
                except TypeError:
                    self._model = CrossEncoder(self.model_name, device=self.device)
                log.info(
                    "reranker ready",
                    extra={
                        "ctx": {
                            "model": self.model_name,
                            "device": self.device,
                            # The window the stage will really read. A mismatch
                            # between this and the model's capacity is invisible
                            # in every metric (it shows up as "no change"), so it
                            # has to be visible in the log.
                            "pair_tokens": self.max_length,
                            "doc_tokens": self.doc_token_budget,
                        }
                    },
                )
                return True
            except Exception as exc:
                log.warning(
                    "reranker unavailable; ranking left to recall channels",
                    extra={"ctx": {"model": self.model_name, "error": str(exc)[:300]}},
                )
                self._failed = True
                return False

    def _clip_many(self, documents: Sequence[str]) -> list[str]:
        """Clip many documents, tokenizing as a batch.

        One tokenizer call for the batch rather than one per document: with the
        rerank pool at ~120 items, per-document tokenization was a measurable
        share of the stage's cost.
        """
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is None:
            return [d[: self.doc_token_budget * 4] for d in documents]
        try:
            encoded = tokenizer(
                list(documents),
                add_special_tokens=False,
                truncation=True,
                max_length=self.doc_token_budget,
            )
            return tokenizer.batch_decode(
                encoded["input_ids"], skip_special_tokens=True
            )
        except Exception:
            return [self._clip(d) for d in documents]

    def score(self, query: str, documents: Sequence[str]) -> list[float] | None:
        """Score (query, document) pairs jointly. Higher means more relevant.

        Documents are clipped to a token budget *before* scoring. This is both
        the latency knob and a quality protection: the cross-encoder's 512-token
        window is shared between the question and the document, so a memory that
        fills the window leaves the model unable to attend to the question at
        all. Measured: 48 ms/doc unclipped versus ~14 ms/doc clipped on long
        memories, and 11 s versus sub-second per search in the real pipeline.
        """
        if not documents:
            return []
        if not self.available:
            return None
        try:
            clipped = self._clip_many(documents)
            pairs = [(query, doc) for doc in clipped]
            raw = self._model.predict(pairs, show_progress_bar=False)  # type: ignore[union-attr]
            return [float(x) for x in raw]
        except Exception as exc:
            log.warning(
                "rerank failed",
                extra={"ctx": {"error": str(exc)[:300]}},
            )
            return None

    def _clip(self, document: str) -> str:
        """Clip a document to ``doc_token_budget`` tokens.

        Falls back to a character heuristic only when the tokenizer is
        unavailable, so scoring never fails purely because clipping could not be
        measured.
        """
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is None:
            return document[: self.doc_token_budget * 4]
        try:
            ids = tokenizer.encode(document, add_special_tokens=False)
            if len(ids) <= self.doc_token_budget:
                return document
            return tokenizer.decode(
                ids[: self.doc_token_budget], skip_special_tokens=True
            )
        except Exception:
            return document[: self.doc_token_budget * 4]


def sigmoid(x: float) -> float:
    """Map an unbounded cross-encoder logit into (0, 1).

    Keeps rerank scores on the same scale as the other components so the blend
    weights stay meaningful.
    """
    if x >= 0:
        z = 2.718281828459045 ** (-x)
        return 1.0 / (1.0 + z)
    z = 2.718281828459045 ** x
    return z / (1.0 + z)


@dataclass
class RerankState:
    """Shared reranker, cached per configuration.

    Keyed like the embedding instance: an unkeyed cache would silently serve a
    stale model across configuration changes and make A/B runs incomparable.
    """

    _cache: dict[tuple, Reranker] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    _instance: "RerankState | None" = None

    @classmethod
    def get_state(cls) -> "RerankState":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get(self, settings: Settings) -> Reranker:
        key = (
            settings.rerank_model,
            settings.rerank_device,
            settings.embed_offline,
            settings.rerank_max_length,
            settings.rerank_doc_tokens,
        )
        with self._lock:
            reranker = self._cache.get(key)
            if reranker is None:
                reranker = Reranker(
                    model_name=settings.rerank_model,
                    device=settings.rerank_device,
                    offline=settings.embed_offline,
                    max_length=settings.rerank_max_length,
                    doc_token_budget=settings.rerank_doc_tokens,
                )
                self._cache[key] = reranker
            return reranker

    def reset(self) -> None:
        with self._lock:
            self._cache.clear()
        RerankState._instance = None
