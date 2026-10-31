"""Card-vector channel tests.

Hermetic: a deterministic fake embedder stands in for the provider, so the
whole channel -- card embedding, storage, cosine search, fusion, degradation --
is verifiable without network access or credentials.

The invariants that matter:

* **Only cards are embedded.** Chunks are persisted facts, never vectors;
  there is no chunk vector store to regress into.
* **Vectors never break lexical retrieval.** Any embedder failure returns no
  candidates and leaves the lexical path untouched.
* **Caching is by content hash.** Re-embedding identical text is free.
* **Cosine search is scope-bound.** A vector query never returns another
  user's cards.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from codemem.common.config import Settings  # noqa: E402
from codemem.common.embeddings import (  # noqa: E402
    EmbeddingCache,
    EmbeddingClient,
    EmbeddingResult,
    content_key,
    pack_vector,
    unpack_vector,
)
from codemem.common.chunking import build_chunks  # noqa: E402
from codemem.search.retrieval import SearchEngine, StoreExperienceVectorIndex  # noqa: E402
from codemem.common.store import MessageInput, Store  # noqa: E402

DIM = 8


# --------------------------------------------------------------------------
# fake embedder: deterministic, dimension-controlled, failure-injectable
# --------------------------------------------------------------------------


class FakeEmbedder:
    """Maps text to a deterministic unit vector; can be made to fail."""

    available = True

    def __init__(self, dim: int = DIM, fail: bool = False, model: str = "fake-embed") -> None:
        self.dim = dim
        self.fail = fail
        self.model = model
        self.calls = 0
        self.texts_seen: list[str] = []

    def embed(self, texts, timeout_s=None):
        self.calls += 1
        if self.fail:
            return None
        self.texts_seen.extend(texts)
        return EmbeddingResult(
            vectors=[self._vector(t) for t in texts],
            model=self.model,
            dim=self.dim,
        )

    def _vector(self, text: str) -> list[float]:
        # Deterministic pseudo-embedding: hashed bag of words, normalized.
        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = hash(token) % self.dim
            vec[h] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        db_path=tmp_path / "vec.db",
        embed_base_url="http://fake",
        embed_api_key="fake",
        embed_model="fake-embed",
        embed_dim=DIM,
        embed_cache_path=tmp_path / "cache.jsonl",
        add_sync_budget_s=40.0,
    )
    base.update(overrides)
    return Settings(**base)


def seed(store: Store, settings: Settings, user_id: str, request_id: str, text: str) -> list:
    messages = [MessageInput("user", text)]
    chunks = build_chunks(user_id=user_id, request_id=request_id, messages=messages, settings=settings)
    state = store.begin_ingestion(
        user_id=user_id, request_id=request_id, session_id="s",
        digest=request_id, mode="full", lease_seconds=60,
    )
    store.commit_chunks(state=state, messages=messages, chunks=chunks, sync_ms=1)
    return chunks


# --------------------------------------------------------------------------
# packing / keys
# --------------------------------------------------------------------------


def test_vector_pack_roundtrip():
    vec = [0.5, -1.25, 3.0, 0.0]
    blob = pack_vector(vec)
    assert len(blob) == 4 * 4
    assert unpack_vector(blob, 4) == pytest.approx(vec, abs=1e-6)


def test_unpack_rejects_wrong_dim():
    with pytest.raises(ValueError):
        unpack_vector(pack_vector([1.0, 2.0]), 3)


def test_content_key_depends_on_model_and_dim():
    a = content_key("hello", "m1", 8)
    assert a != content_key("hello", "m2", 8)
    assert a != content_key("hello", "m1", 16)
    assert a == content_key("hello", "m1", 8)
    assert a != content_key("other", "m1", 8)


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


def test_cache_roundtrip_and_stats(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.jsonl")
    cache.put("k1", [1.0, 2.0])
    assert cache.get("k1") == [1.0, 2.0]
    assert cache.get("missing") is None
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "c.jsonl"
    EmbeddingCache(path).put("k1", [1.0])
    assert EmbeddingCache(path).get("k1") == [1.0]


def test_cache_tolerates_corrupt_lines(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text('{"k":"good","v":[1.0]}\nnot json\n{"k":"also","v":[2.0]}\n', encoding="utf-8")
    cache = EmbeddingCache(path)
    assert cache.get("good") == [1.0]
    assert cache.get("also") == [2.0]


def test_embedding_cache_hit_avoids_call(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.jsonl")
    client = EmbeddingClient(make_settings(tmp_path), cache)
    # Prime via the fake transport by monkeypatching _ensure_client.
    fake = FakeEmbedder()
    client._client = type("C", (), {"embeddings": fake_embeddings_api(fake)})()

    first = client.embed(["alpha"])
    calls_after_first = fake.calls
    second = client.embed(["alpha"])
    assert first is not None and second is not None
    assert fake.calls == calls_after_first, "cached text must not re-call the provider"
    assert second.cached is True


def fake_embeddings_api(fake: FakeEmbedder):
    class _Datum:
        def __init__(self, embedding):
            self.embedding = embedding

    class _Resp:
        def __init__(self, vectors, model):
            self.data = [_Datum(v) for v in vectors]
            self.model = model
            self.usage = type("U", (), {"prompt_tokens": 1, "total_tokens": 1})()

    class _Embeddings:
        def create(self, model, input, timeout=None):
            return _Resp([fake._vector(t) for t in input], model)

    return _Embeddings()


# --------------------------------------------------------------------------
# store-level card vector search
# --------------------------------------------------------------------------


def _ingestion(store: Store, user_id: str, request_id: str):
    return store.begin_ingestion(
        user_id=user_id, request_id=request_id, session_id="s",
        digest=request_id, mode="full", lease_seconds=60,
    )


def _card_row(store: Store, state, experience_id: str, content: str) -> None:
    from codemem.common.store import now_iso

    store.insert_experience(
        user_id="u1",
        session_id="s",
        experience_id=experience_id,
        ingestion_id=state.ingestion_id,
        kind="debug",
        payload_json="{}",
        full_content=content,
        extraction_version="v1",
        created_at=now_iso(),
        terms=[],
        entities=[],
        anchors=[],
    )


def test_no_chunk_vector_store_exists(tmp_path):
    """Chunks are never embedded: the chunk vector table is gone for good."""
    settings = make_settings(tmp_path)
    store = Store(settings)
    seed(store, settings, "u1", "r1", "alpha beta gamma delta epsilon zeta")
    tables = {
        row[0] for row in store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "chunk_vectors" not in tables
    assert "experience_vectors" in tables


def test_experience_vectors_are_isolated_and_follow_their_card(tmp_path):
    settings = make_settings(tmp_path)
    store = Store(settings)
    state = _ingestion(store, "u1", "r1")
    _card_row(store, state, "e:1", "card one")
    _card_row(store, state, "e:2", "card two")

    unit = [1.0] + [0.0] * (DIM - 1)
    other = [0.0, 1.0] + [0.0] * (DIM - 2)
    store.store_experience_vectors("u1", [
        ("e:1", "fake-embed", DIM, pack_vector(unit)),
        ("e:2", "fake-embed", DIM, pack_vector(other)),
    ])
    hits = dict(store.experience_vector_search("u1", unit, "fake-embed", 10))
    assert hits["e:1"] == pytest.approx(1.0, abs=1e-6)
    assert hits["e:2"] == pytest.approx(0.0, abs=1e-6)

    # A vector belongs to its card, not to the call that produced it: once the
    # card is deleted with its session the join must stop returning the
    # embedding, or a stale vector keeps recalling memory that no longer exists.
    store.delete_session_experiences("u1", "s")
    assert store.experience_vector_search("u1", unit, "fake-embed", 10) == []

    # Another scope is invisible even with an identical vector.
    other_state = _ingestion(store, "u2", "r2")
    store.insert_experience(
        user_id="u2", session_id="s", experience_id="e:9",
        ingestion_id=other_state.ingestion_id,
        kind="debug", payload_json="{}", full_content="other", extraction_version="v1",
        created_at="2026-01-01T00:00:00Z", terms=[], entities=[], anchors=[],
    )
    store.store_experience_vectors("u2", [("e:9", "fake-embed", DIM, pack_vector(unit))])
    assert "e:9" not in dict(store.experience_vector_search("u1", unit, "fake-embed", 10))


def test_experience_vector_coverage_reports_fraction(tmp_path):
    settings = make_settings(tmp_path)
    store = Store(settings)
    state = _ingestion(store, "u1", "r1")
    _card_row(store, state, "e:1", "one")
    _card_row(store, state, "e:2", "two")
    store.store_experience_vectors("u1", [("e:1", "fake-embed", DIM, pack_vector([1.0] * DIM))])
    coverage = store.experience_vector_coverage("u1", "fake-embed")
    assert coverage["cards"] == 2
    assert coverage["vectorized"] == 1


def test_card_vector_index_degrades_to_empty(tmp_path):
    settings = make_settings(tmp_path)
    store = Store(settings)
    assert StoreExperienceVectorIndex(store, None, "fake-embed").search("u1", "x", 5) == []
    broken = StoreExperienceVectorIndex(store, FakeEmbedder(fail=True), "fake-embed")
    assert broken.available is True
    assert broken.search("u1", "x", 5) == []


def test_card_vector_channel_orders_cards(tmp_path):
    from codemem.search.query import build_query_features
    from codemem.search.retrieval import SearchDiagnostics

    settings = make_settings(tmp_path)
    store = Store(settings)
    state = _ingestion(store, "u1", "r1")
    _card_row(store, state, "e:1", "card one")

    unit = [1.0] + [0.0] * (DIM - 1)
    store.store_experience_vectors("u1", [("e:1", "fake-embed", DIM, pack_vector(unit))])

    class FixedEmbedder(FakeEmbedder):
        def _vector(self, text: str) -> list[float]:
            return unit

    index = StoreExperienceVectorIndex(store, FixedEmbedder(dim=DIM), "fake-embed")
    engine = SearchEngine(store, settings, card_vector_index=index)
    features = build_query_features("unrelated wording", None, denoise=settings.query_denoise)
    diag = SearchDiagnostics()
    pool = engine._recall_cards("u1", features, diag)
    assert diag.channels["card_vector"] == 1
    card = next(iter(pool.values()))
    assert "card_vector" in card.channels


def test_vector_failure_leaves_lexical_card_retrieval_intact(tmp_path):
    """A vector outage must not break retrieval: the lexical card channels
    still answer."""
    import json as _json

    from codemem.add.experience import ExperienceCard, card_to_payload

    settings = make_settings(tmp_path)
    store = Store(settings)
    state = _ingestion(store, "u1", "r1")
    card = ExperienceCard(
        experience_id="e:1", user_id="u1", session_id="s",
        problem="src/loader.py raised ConfigParseError",
        implement="guard the parse call",
    )
    store.insert_experience(
        user_id="u1", session_id="s", experience_id="e:1",
        ingestion_id=state.ingestion_id, kind="debug",
        payload_json=_json.dumps(card_to_payload(card), ensure_ascii=False),
        full_content="src/loader.py raised ConfigParseError",
        extraction_version="v1", created_at="2026-01-01T00:00:00Z",
        terms=["problemconfigparseerror", "locateloader"], entities=[],
    )
    engine = SearchEngine(
        store, settings,
        card_vector_index=StoreExperienceVectorIndex(store, FakeEmbedder(fail=True), "fake-embed"),
    )
    data, diag = engine.search(user_id="u1", query="ConfigParseError", top_k=10)
    assert data, "lexical card retrieval must survive a vector outage"
    assert "ConfigParseError" in data[0]["content"]


# --------------------------------------------------------------------------
# service-level card embedding
# --------------------------------------------------------------------------


def test_embed_cards_writes_vectors_and_fails_soft(tmp_path, monkeypatch):
    import codemem.service as service_module
    from codemem.add.experience import ExperienceCard
    from codemem.service import MemoryService

    settings = make_settings(tmp_path)
    store = Store(settings)
    state = _ingestion(store, "u1", "r1")
    _card_row(store, state, "e:1", "card one")

    fake = FakeEmbedder(dim=DIM)
    monkeypatch.setattr(service_module, "get_embedder", lambda *a, **k: fake)
    svc = MemoryService(store, settings)
    card = ExperienceCard(
        experience_id="e:1", user_id="u1", session_id="s",
        problem="ConfigParseError in src/loader.py", locate="load_config",
        implement="guard the parse",
    )
    assert svc._embed_cards("u1", [card]) == 1
    assert store.experience_vector_coverage("u1", "fake-embed")["vectorized"] == 1

    # A failing provider leaves the card lexically searchable and writes nothing.
    svc.embedder = FakeEmbedder(dim=DIM, fail=True)
    assert svc._embed_cards("u1", [card]) == 0
