"""Dense retrieval: isolation, degradation, and default-off behaviour.

The isolation tests use synthetic vectors, not a real model, so they are fast
and deterministic. That matters because the dense channel is the easiest place
to leak memory across users: a shared vector index would return another user's
nearest neighbours no matter what filter is applied afterwards.
"""

from __future__ import annotations

import pytest

from codemem.add.pipeline import AddPipeline
from codemem.core.config import Settings
from codemem.embed import Instance, cosine
from codemem.index.store import Store
from codemem.search.service import SearchPipeline


class _M:
    def __init__(self, content: str, ts: int | None = None) -> None:
        self.role = "user"
        self.content = content
        self.timestamp = ts


def _add(store: Store, settings: Settings, user: str, request: str, content: str) -> None:
    AddPipeline(settings, store).handle(
        request_id=request, user_id=user, session_id=request, messages=[_M(content)]
    )


def _unit(seed: int, dim: int = 8) -> list[float]:
    """A deterministic unit vector, so tests never need a model."""
    import math

    raw = [math.sin((seed + 1) * (i + 1)) for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]


# --------------------------------------------------------------- isolation --


def test_dense_search_is_user_scoped(settings):
    """Vectors written for one user must never be returned for another."""
    store = Store(settings)
    try:
        _add(store, settings, "alice", "a", "Alice stores the deploy key rotation policy.")
        _add(store, settings, "bob", "b", "Bob stores the database migration steps.")

        with store._read() as conn:  # noqa: SLF001
            rows = conn.execute(
                "SELECT id, user_id FROM memory ORDER BY id"
            ).fetchall()
        alice_id = next(r["id"] for r in rows if r["user_id"] == "alice")
        bob_id = next(r["id"] for r in rows if r["user_id"] == "bob")

        vector = _unit(1)
        store.store_vectors("alice", [(int(alice_id), vector)])
        store.store_vectors("bob", [(int(bob_id), vector)])

        alice_hits = store.dense_search("alice", vector, 10)
        bob_hits = store.dense_search("bob", vector, 10)
        carol_hits = store.dense_search("carol", vector, 10)

        assert [mid for mid, _ in alice_hits] == [int(alice_id)]
        assert [mid for mid, _ in bob_hits] == [int(bob_id)]
        assert carol_hits == [], "a user with no vectors must get nothing"
    finally:
        store.close()


def test_identical_vectors_across_users_stay_separate(settings):
    """Byte-identical embeddings under two users must still not cross."""
    store = Store(settings)
    try:
        _add(store, settings, "alice", "a", "Identical text about src/a/mod.py.")
        _add(store, settings, "bob", "b", "Identical text about src/a/mod.py.")
        with store._read() as conn:  # noqa: SLF001
            ids = {
                r["user_id"]: r["id"]
                for r in conn.execute("SELECT id, user_id FROM memory")
            }
        vector = _unit(7)
        store.store_vectors("alice", [(int(ids["alice"]), vector)])
        store.store_vectors("bob", [(int(ids["bob"]), vector)])

        alice = store.dense_search("alice", vector, 10)
        bob = store.dense_search("bob", vector, 10)
        assert len(alice) == 1 and len(bob) == 1
        assert alice[0][0] != bob[0][0]
    finally:
        store.close()


def test_delete_user_removes_vectors(settings):
    store = Store(settings)
    try:
        _add(store, settings, "alice", "a", "Alice fact about src/x/y.py.")
        with store._read() as conn:  # noqa: SLF001
            mid = conn.execute("SELECT id FROM memory").fetchone()["id"]
        store.store_vectors("alice", [(int(mid), _unit(3))])
        assert store.vector_coverage("alice")[0] == 1

        removed = store.delete_user("alice")
        assert removed["memory_vector"] == 1
        assert store.dense_search("alice", _unit(3), 10) == []
        assert store.vector_coverage("alice") == (0, 0)
    finally:
        store.close()


# -------------------------------------------------------------- degradation --


def test_dense_enabled_by_default(settings):
    """Dense recall is on by default. The cost/benefit flips with hardware —
    on a GPU it improves nDCG@10 and recall@10 over rerank-only, on CPU it did
    not — and with device="auto" each host gets the appropriate outcome."""
    assert Settings().dense_enabled is True
    store = Store(settings)
    try:
        _add(store, settings, "u1", "a", "A fact about src/mod.py retry logic.")
        # Nothing embedded, so the dense channel has no candidates and the
        # lexical channels still answer.
        assert store.vector_coverage("u1") == (0, 1)
        items = SearchPipeline(settings, store).handle(
            user_id="u1", query="src/mod.py retry", options=None, top_k=10
        )
        assert items, "the service must work with the dense channel off"
    finally:
        store.close()


def test_search_survives_unavailable_encoder(settings):
    """An encoder that cannot load must degrade, not raise."""
    settings.dense_enabled = True
    settings.embed_backend = "local"
    settings.embed_model = "definitely/not-a-real-model-xyz"
    settings.embed_offline = True
    Instance.reset()
    store = Store(settings)
    try:
        _add(store, settings, "u1", "a", "A fact about src/mod.py retry logic.")
        instance = Instance.get(settings)
        assert not instance.available, "a missing model must report unavailable"
        assert instance.embed(["anything"]) is None

        items = SearchPipeline(settings, store, embedder=instance).handle(
            user_id="u1", query="src/mod.py retry logic", options=None, top_k=10
        )
        assert items, "lexical channels must still return the memory"
    finally:
        store.close()
        Instance.reset()


def test_dimension_mismatch_is_skipped(settings):
    """Vectors from a different encoder must be ignored, not crash scoring."""
    store = Store(settings)
    try:
        _add(store, settings, "u1", "a", "Fact about src/a.py.")
        with store._read() as conn:  # noqa: SLF001
            mid = conn.execute("SELECT id FROM memory").fetchone()["id"]
        store.store_vectors("u1", [(int(mid), _unit(4, dim=8))])
        # Query with a different dimension.
        assert store.dense_search("u1", _unit(4, dim=16), 10) == []
        # Matching dimension still works.
        assert len(store.dense_search("u1", _unit(4, dim=8), 10)) == 1
    finally:
        store.close()


# ------------------------------------------------------------------ caching --


def test_instance_cache_is_keyed_on_configuration(settings):
    """An unkeyed singleton made an A/B run report identical metrics for two
    different configurations, because the second run reused the first's encoder."""
    Instance.reset()
    disabled = Settings(data_dir=settings.data_dir, dense_enabled=False)
    enabled = Settings(data_dir=settings.data_dir, dense_enabled=True)
    a = Instance.get(disabled)
    b = Instance.get(enabled)
    assert a is not b, "different configurations must not share an instance"
    assert Instance.get(disabled) is a, "same configuration should be reused"
    Instance.reset()


# ------------------------------------------------------------------- vectors --


def test_store_vectors_is_idempotent(settings):
    store = Store(settings)
    try:
        _add(store, settings, "u1", "a", "Fact about src/a.py.")
        with store._read() as conn:  # noqa: SLF001
            mid = int(conn.execute("SELECT id FROM memory").fetchone()["id"])
        store.store_vectors("u1", [(mid, _unit(5))])
        store.store_vectors("u1", [(mid, _unit(6))])  # re-embed
        assert store.vector_coverage("u1")[0] == 1, "re-embedding must replace"
        hits = store.dense_search("u1", _unit(6), 5)
        assert hits and hits[0][0] == mid
        assert hits[0][1] > 0.999, "the replacement vector should win"
    finally:
        store.close()


def test_cosine_of_normalized_vectors(settings):
    vector = _unit(1)
    assert abs(cosine(vector, vector) - 1.0) < 1e-9


def test_device_auto_resolves_and_explicit_is_honoured():
    """auto must pick a real device; an explicit choice must not be overridden,
    so a misconfiguration stays visible instead of being silently downgraded."""
    from codemem.embed import resolve_device

    resolved = resolve_device("auto")
    assert resolved in ("cpu", "cuda")
    assert resolve_device("cpu") == "cpu"
    # An explicit cuda request is passed through untouched, even on a CPU host.
    assert resolve_device("cuda") == "cuda"
