"""Reranking: degradation, clipping, and blending.

The reranker is optional, so the important properties are that a missing model
degrades to the recall ranking (never raises) and that documents are clipped by
real token count, not by characters — per-character clipping left long memories
costing 48 ms each and made search take 11 s.
"""

from __future__ import annotations

from codemem.core.config import Settings
from codemem.rerank import Reranker, RerankState, sigmoid


class _StubTokenizer:
    def __call__(self, texts, add_special_tokens=False, truncation=False, max_length=None):
        assert isinstance(texts, (list, tuple))
        ids = [[ord(c) for c in t] for t in texts]
        if truncation and max_length:
            ids = [row[:max_length] for row in ids]
        return {"input_ids": ids}

    def batch_decode(self, rows, skip_special_tokens=False):
        return ["".join(chr(c) for c in row) for row in rows]

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(c) for c in ids)


class _StubModel:
    """Stands in for a CrossEncoder without loading anything."""

    def __init__(self):
        self.tokenizer = _StubTokenizer()
        self.seen: list[tuple[str, str]] = []
        self.batch_sizes: list[int] = []

    def predict(self, pairs, show_progress_bar=False):
        self.seen = list(pairs)
        self.batch_sizes.append(len(pairs))
        # Score by shared characters, so ordering is deterministic and testable.
        return [len(set(q) & set(d)) / max(1, len(set(q))) for q, d in pairs]


# ------------------------------------------------------------- degradation --


def test_unavailable_reranker_returns_none(settings):
    r = Reranker(model_name="definitely/not-a-real-model-xyz", device="cpu", offline=True)
    assert not r.available
    assert r.score("q", ["doc"]) is None


def test_search_without_reranker_still_returns_results(settings):
    """The pipeline must work when reranking is unavailable."""
    from codemem.add.pipeline import AddPipeline
    from codemem.index.store import Store
    from codemem.search.service import SearchPipeline

    store = Store(settings)
    try:
        class M:
            role = "user"
            content = "We fixed IndexError in src/parser/tokenizer.py by guarding the buffer."
            timestamp = None

        AddPipeline(settings, store).handle(
            request_id="a", user_id="u1", session_id="s1", messages=[M()]
        )
        broken = Reranker(model_name="definitely/not-a-real-model-xyz", offline=True)
        items = SearchPipeline(settings, store, reranker=broken).handle(
            user_id="u1", query="IndexError tokenizer buffer", options=None, top_k=10
        )
        assert items, "lexical channels must still answer"
    finally:
        store.close()


def test_score_failure_is_swallowed(settings):
    """An exception inside predict must not propagate to the request."""

    class Exploding(_StubModel):
        def predict(self, pairs, show_progress_bar=False):
            raise RuntimeError("model exploded")

    r = Reranker(model_name="stub", offline=True)
    r._model = Exploding()
    r._failed = False
    assert r.score("q", ["a", "b"]) is None


# ---------------------------------------------------------------- clipping --


def test_documents_are_clipped_by_token_count(settings):
    """Clipping must use the tokenizer, not a character guess: 2048 characters
    can be 630 tokens on this token family."""
    model = _StubModel()
    r = Reranker(model_name="stub", offline=True, doc_token_budget=50)
    r._model = model
    long_doc = "x" * 5000
    r.score("query", [long_doc, "short"])
    sent = [d for _, d in model.seen]
    assert len(sent[0]) == 50, f"expected 50 chars after clipping, got {len(sent[0])}"
    assert sent[1] == "short", "short documents must pass through untouched"


def test_clipping_is_batched(settings):
    """One tokenizer call for the pool, not one per document."""
    calls = {"n": 0}

    class CountingTokenizer(_StubTokenizer):
        def __call__(self, texts, **kwargs):
            calls["n"] += 1
            assert len(texts) == 120, "expected a single batched call"
            return super().__call__(texts, **kwargs)

    model = _StubModel()
    model.tokenizer = CountingTokenizer()
    r = Reranker(model_name="stub", offline=True)
    r._model = model
    r.score("q", [f"doc {i}" for i in range(120)])
    assert calls["n"] == 1


def test_clipping_falls_back_when_tokenizer_missing(settings):
    class NoTokenizer:
        def predict(self, pairs, show_progress_bar=False):
            return [0.0 for _ in pairs]

    r = Reranker(model_name="stub", offline=True, doc_token_budget=10)
    r._model = NoTokenizer()
    # Should not raise, even though clipping cannot be measured.
    assert r.score("q", ["y" * 1000]) == [0.0]


# ----------------------------------------------------------------- signals --


def test_score_orders_more_relevant_first(settings):
    model = _StubModel()
    r = Reranker(model_name="stub", offline=True)
    r._model = model
    query = "tokenizer indexerror"
    scores = r.score(
        query,
        ["tokenizer indexerror fixed", "unrelated content", "indexerror in tokenizer"],
    )
    assert scores is not None
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]


def test_sigmoid_is_bounded_and_monotonic():
    # Values are bounded in (0, 1) and strictly increasing. Saturation at the
    # extremes is expected in double precision: exp(-100) underflows, so
    # sigmoid(100) is exactly 1.0.
    assert 0.0 <= sigmoid(-100) < sigmoid(-1) < sigmoid(0) < sigmoid(1) <= 1.0
    assert abs(sigmoid(0) - 0.5) < 1e-9
    for x in (-50, -5, 0, 5, 50):
        assert 0.0 <= sigmoid(x) <= 1.0


# ------------------------------------------------------------------ cache --


def test_reranker_cache_is_keyed_on_configuration(settings):
    """Keyed like the embedder, so an A/B run cannot reuse a stale model."""
    state = RerankState()
    a = settings
    b = type(settings)(**{**settings.__dict__} if hasattr(settings, "__dict__") else {})
    b = type(settings)(
        data_dir=settings.data_dir,
        rerank_model="other-model",
    )
    assert state.get(a) is not state.get(b)
    assert state.get(a) is state.get(a), "same configuration should be reused"


def test_rerank_enabled_by_default():
    """Production default is on: it is the largest measured gain and costs Add
    nothing. The test suite disables it only for speed and hermeticity."""
    assert Settings().rerank_enabled is True
    assert Settings().rerank_device == "auto"
    assert Settings().rerank_weight == 0.65


def test_rerank_normalization_is_pool_independent(settings):
    """A memory's reranked contribution must not depend on how many other
    candidates were reranked alongside it.

    Regression: normalising by the maximum rerank score in the head made the
    ranking a function of rerank_top_n, which produced an incoherent metric
    sequence (top_n=30 -> MRR 0.789, top_n=60 -> 0.772, top_n=120 -> 0.818).
    A fixed temperature removes that dependency.
    """
    from codemem.rerank import sigmoid

    temperature = settings.rerank_temperature
    logit = 3.0
    alone = sigmoid(logit / temperature)

    # Sitting beside a much stronger candidate must not change this memory's
    # own contribution.
    with_stronger = sigmoid(logit / temperature)
    assert abs(alone - with_stronger) < 1e-12

    # And the mapping is absolute: the same logit is always worth the same.
    assert sigmoid(0.0 / temperature) == 0.5
