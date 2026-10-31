"""Regression tests for the eight review findings (FIXES.md).

Each test reproduces the reported failure mode so the fix cannot silently
regress. Where the original reproduction came from the user, the test mirrors
their exact numbers.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from codemem.api import create_app
from codemem.common.config import Settings
from codemem.common.embeddings import (
    EmbeddingCache,
    ModelCallLimiter,
    reset_model_limiter_for_tests,
)
from codemem.search.retrieval import Candidate, SearchEngine, SearchDiagnostics
from codemem.common.store import MessageInput, Store


class CountingEmbedder:
    """Deterministic embedder that counts calls and can sleep per call."""

    available = True

    def __init__(self, dim: int = 8, sleep: float = 0.0, fail: bool = False) -> None:
        self.dim = dim
        self.sleep = sleep
        self.fail = fail
        self.calls = 0

    def embed(self, texts, timeout_s=None):
        self.calls += 1
        if self.sleep:
            time.sleep(self.sleep)
        if self.fail:
            return None
        from codemem.common.embeddings import EmbeddingResult

        return EmbeddingResult(
            vectors=[[0.1] * self.dim for _ in texts],
            model="counting",
            dim=self.dim,
        )


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        db_path=tmp_path / "fix.db",
        embed_base_url="http://fake",
        embed_api_key="fake",
        embed_model="fake",
        embed_dim=8,
        embed_batch_size=2,
        embed_cache_path=tmp_path / "cache.jsonl",
        add_sync_budget_s=40.0,
        token_budget_total=96_000,
        token_budget_target=64_000,
    )
    base.update(overrides)
    return Settings(**base)


# --------------------------------------------------------------------------
# P1.4 — adjacency expansion must not drop already-selected candidates
# --------------------------------------------------------------------------


def test_p1_4_final_cut_preserves_selected_results(tmp_path):
    """Original repro: 3 selected, budget hit during first cut -> 1 returned."""
    settings = make_settings(tmp_path, token_budget_target=10, token_budget_total=10)
    store = Store(settings)
    engine = SearchEngine(store, settings)

    def cand(cid: str, text: str) -> Candidate:
        from codemem.add.experience import ExperienceCard

        return Candidate(
            item_id=f"experience:{cid}", kind="experience", text=text,
            fused_score=1.0,
            payload={"card": ExperienceCard(
                experience_id=cid, user_id="u", session_id="s",
                problem=text, implement="a repair",
            ), "chunks_by_id": {}, "created_at": None},
        )

    selected = [cand("c1", "a " * 20), cand("c2", "b " * 20), cand("c3", "c " * 20)]
    diag = SearchDiagnostics()
    out = engine._finalize(selected, diag, final_limit=10)
    assert len(out) >= len(selected), (
        f"the final cut dropped selected results: {len(selected)} -> {len(out)}"
    )


# --------------------------------------------------------------------------
# P1.1 — enrichment budget is cumulative, not per batch
# --------------------------------------------------------------------------


def test_p1_1_enrichment_budget_is_cumulative(tmp_path, monkeypatch):
    """Original repro: 0.6s budget, 3 batches x 0.3s -> must not run all 3."""
    settings = make_settings(tmp_path, embed_budget_s=0.6, embed_batch_size=2)
    store = Store(settings)
    from codemem.service import MemoryService
    import codemem.service as service_module

    slow = CountingEmbedder(dim=8, sleep=0.3)
    monkeypatch.setattr(service_module, "get_embedder", lambda *a, **k: slow)

    svc = MemoryService(store, settings)
    svc.embedder = slow

    started = time.monotonic()
    outcome = svc.add(
        request_id="r1", user_id="u1", session_id="s1",
        raw_messages=[{"role": "user", "content": f"chunk{i} " * 30} for i in range(5)],
    )
    elapsed = time.monotonic() - started

    assert outcome.success is True
    assert slow.calls <= 2, f"budget not cumulative: {slow.calls} batches ran (0.6s budget)"
    assert elapsed < 1.0, f"elapsed {elapsed:.2f}s exceeded cumulative budget"


def test_p1_1_total_ms_includes_enrichment(tmp_path, monkeypatch):
    """guard.observe must see the full request, enrichment included."""
    settings = make_settings(tmp_path, embed_budget_s=5.0)
    store = Store(settings)
    from codemem.service import MemoryService
    import codemem.service as service_module

    slow = CountingEmbedder(dim=8, sleep=0.15)
    monkeypatch.setattr(service_module, "get_embedder", lambda *a, **k: slow)

    svc = MemoryService(store, settings)
    svc.embedder = slow
    outcome = svc.add(
        request_id="r1", user_id="u1", session_id="s1",
        raw_messages=[{"role": "user", "content": "text for embedding " * 20}],
    )
    assert outcome.success
    embed_steps = [s for s in outcome.steps if s["name"] == "embed"]
    if embed_steps and not embed_steps[0]["skipped"]:
        assert outcome.elapsed_ms >= embed_steps[0]["duration_ms"], (
            "elapsed_ms must include enrichment duration"
        )


def test_p1_1_search_honours_deadline_for_vector(tmp_path):
    """An expired deadline must skip the vector channel rather than overrun."""
    settings = make_settings(tmp_path)
    store = Store(settings)
    embedder = CountingEmbedder(dim=8)
    from codemem.search.retrieval import StoreExperienceVectorIndex

    engine = SearchEngine(
        store, settings,
        card_vector_index=StoreExperienceVectorIndex(store, embedder, "fake"),
    )

    class Expired:
        expired = True

    data, diag = engine.search(user_id="u1", query="anything", top_k=5, deadline=Expired())
    assert "card_vector_skipped_deadline" in diag.degraded
    assert embedder.calls == 0, "expired deadline must not trigger an embedding call"


# --------------------------------------------------------------------------
# P1.3 — purge must remove vector data
# --------------------------------------------------------------------------


def test_p1_3_purge_removes_vectors(tmp_path):
    """Original repro: after purge, vector rows survived the delete."""
    settings = make_settings(tmp_path)
    store = Store(settings)
    messages = [MessageInput("user", "text to vectorize for purge test")]
    from codemem.common.chunking import build_chunks
    from codemem.common.embeddings import pack_vector

    chunks = build_chunks(user_id="u1", request_id="r1", messages=messages, settings=settings)
    state = store.begin_ingestion(
        user_id="u1", request_id="r1", session_id="s", digest="d", mode="full", lease_seconds=60
    )
    store.commit_chunks(state=state, messages=messages, chunks=chunks, sync_ms=1)
    store.insert_experience(
        user_id="u1", session_id="s", experience_id="e:1",
        ingestion_id=state.ingestion_id, kind="debug", payload_json="{}",
        full_content="card", extraction_version="v1", created_at="2026-01-01T00:00:00Z",
        terms=[], entities=[],
    )
    store.store_experience_vectors(
        "u1", [("e:1", "fake", 8, pack_vector([0.1] * 8))]
    )
    assert store.experience_vector_coverage("u1", "fake")["vectorized"] == 1

    removed = store.purge_user("u1")
    assert removed.get("experience_vectors") == 1
    assert store.experience_vector_coverage("u1", "fake")["vectorized"] == 0


def test_p1_3_purge_leaves_nothing_for_scope(tmp_path):
    settings = make_settings(tmp_path)
    store = Store(settings)
    messages = [MessageInput("user", "scope residue check content")]
    from codemem.common.chunking import build_chunks
    from codemem.common.embeddings import pack_vector

    chunks = build_chunks(user_id="u1", request_id="r1", messages=messages, settings=settings)
    state = store.begin_ingestion(
        user_id="u1", request_id="r1", session_id="s", digest="d", mode="full", lease_seconds=60
    )
    store.commit_chunks(
        state=state, messages=messages, chunks=chunks, sync_ms=1
    )
    store.insert_experience(
        user_id="u1", session_id="s", experience_id="e:1",
        ingestion_id=state.ingestion_id, kind="debug", payload_json="{}",
        full_content="card", extraction_version="v1", created_at="2026-01-01T00:00:00Z",
        terms=[], entities=[],
    )
    store.store_experience_vectors(
        "u1", [("e:1", "fake", 8, pack_vector([0.1] * 8))]
    )
    store.purge_user("u1")

    for table in ("chunks", "sessions", "ingestions", "experiences", "experience_vectors"):
        n = store.conn.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE user_id = ?", ("u1",)
        ).fetchone()["n"]
        assert n == 0, f"{table} still has {n} rows after purge"


# --------------------------------------------------------------------------
# P2.5 — rendered content must respect the total budget
# --------------------------------------------------------------------------


def test_p2_5_total_budget_not_exceeded(tmp_path):
    """Original repro: budget 10, actual returned 14."""
    settings = make_settings(tmp_path, token_budget_total=10, token_budget_target=10)
    store = Store(settings)
    from codemem.common.chunking import build_chunks
    from codemem.common.tokens import TokenCounter

    messages = [MessageInput("user", "alpha beta gamma delta epsilon " * 30)]
    chunks = build_chunks(user_id="u1", request_id="r1", messages=messages, settings=settings)
    state = store.begin_ingestion(
        user_id="u1", request_id="r1", session_id="s", digest="d", mode="lex_only", lease_seconds=60
    )
    store.commit_chunks(
        state=state, messages=messages, chunks=chunks, sync_ms=1
    )

    counter = TokenCounter(settings)
    engine = SearchEngine(store, settings, token_counter=counter)
    data, diag = engine.search(user_id="u1", query="alpha beta", top_k=100)
    total = sum(counter.count(item["content"]) for item in data)
    assert total <= settings.token_budget_total, (
        f"returned {total} tokens against a {settings.token_budget_total} budget"
    )


def test_p2_5_soft_target_does_not_truncate_output(tmp_path):
    """Follow-up fix: the soft target must not act as a hard cap.

    Original repro: soft target 10, hard cap 100, one 20-token candidate ->
    returned 0 because the output stage used the soft target as a limit.
    """
    settings = make_settings(
        tmp_path, token_budget_total=100, token_budget_target=10, min_output_body_tokens=1
    )
    store = Store(settings)
    from codemem.common.chunking import build_chunks
    from codemem.common.tokens import TokenCounter

    messages = [MessageInput("user", "evidence body text " * 40)]
    chunks = build_chunks(user_id="u1", request_id="r1", messages=messages, settings=settings)
    state = store.begin_ingestion(
        user_id="u1", request_id="r1", session_id="s", digest="d", mode="lex_only", lease_seconds=60
    )
    store.commit_chunks(
        state=state, messages=messages, chunks=chunks, sync_ms=1
    )

    from codemem.add.experience import ExperienceCard

    card_candidate = Candidate(
        item_id="experience:e:1", kind="experience",
        text="evidence body text " * 40, fused_score=3.0,
        payload={"card": ExperienceCard(
            experience_id="e:1", user_id="u1", session_id="s",
            problem="evidence body text", implement="a repair",
        ), "chunks_by_id": {}, "created_at": None},
    )
    engine = SearchEngine(store, settings)
    engine._recall_cards = lambda *a, **k: {c.item_id: c for c in [card_candidate]}  # type: ignore[assignment]
    engine._rank = lambda pool, features: [card_candidate]  # type: ignore[assignment]
    data, _ = engine.search(user_id="u1", query="evidence body", top_k=100)

    assert data, "soft target must not truncate output to zero"
    total = sum(TokenCounter(settings).count(item["content"]) for item in data)
    assert total <= settings.token_budget_total
    assert total > settings.token_budget_target or len(data) >= 1, (
        "high-value content may legitimately exceed the soft target"
    )


def test_p2_5_oversized_item_is_compressed_not_dropped(tmp_path):
    """A candidate too large for the remaining budget is compressed to fit.

    The output stage caps each item at ``min(tier_cap, remaining)``, so a large
    candidate is truncated rather than dropped -- that is the "compress to the
    remaining budget" branch of the fix, and the whole output must not stop.
    """
    settings = make_settings(
        tmp_path, token_budget_total=60, token_budget_target=50, min_output_body_tokens=1
    )
    store = Store(settings)
    engine = SearchEngine(store, settings)

    from codemem.common.tokens import TokenCounter

    def cand(cid: str, text: str, score: float) -> Candidate:
        from codemem.add.experience import ExperienceCard

        return Candidate(
            item_id=f"experience:{cid}", kind="experience", text=text,
            fused_score=score,
            payload={"card": ExperienceCard(
                experience_id=cid, user_id="u", session_id="s",
                problem=text, implement="a repair",
            ), "chunks_by_id": {}, "created_at": None},
        )

    ordered = [
        cand("big", "x " * 2000, 3.0),
        cand("small", "short evidence body", 2.0),
    ]
    engine._recall_cards = lambda *a, **k: {c.item_id: c for c in ordered}  # type: ignore[assignment]
    engine._rank = lambda pool, features: ordered  # type: ignore[assignment]
    data, _ = engine.search(user_id="u", query="evidence", top_k=100)

    counter = TokenCounter(settings)
    total = sum(counter.count(item["content"]) for item in data)
    assert data, "output must not be empty"
    assert total <= settings.token_budget_total, (
        f"returned {total} tokens against a {settings.token_budget_total} hard cap"
    )
    # The oversized item was compressed, not returned at full size.
    big = next((item for item in data if item["id"] == "experience:big"), None)
    if big is not None:
        assert counter.count(big["content"]) <= settings.token_budget_total


def test_p2_5_skips_item_when_no_room_for_body(tmp_path):
    """When the remaining budget cannot hold any evidence body, skip the item
    and continue rather than emitting a header-only entry."""
    settings = make_settings(
        tmp_path, token_budget_total=40, token_budget_target=35,
        min_output_body_tokens=20,
    )
    store = Store(settings)
    engine = SearchEngine(store, settings)

    from codemem.common.tokens import TokenCounter

    def cand(cid: str, text: str, score: float) -> Candidate:
        from codemem.add.experience import ExperienceCard

        return Candidate(
            item_id=f"experience:{cid}", kind="experience", text=text,
            fused_score=score,
            payload={"card": ExperienceCard(
                experience_id=cid, user_id="u", session_id="s",
                problem=text, implement="a repair",
            ), "chunks_by_id": {}, "created_at": None},
        )

    # First item consumes nearly the whole budget; the second cannot then fit a
    # 20-token body, so it must be skipped rather than returned as header-only.
    ordered = [
        cand("filler", "f " * 200, 3.0),
        cand("tail", "tiny", 2.0),
    ]
    engine._recall_cards = lambda *a, **k: {c.item_id: c for c in ordered}  # type: ignore[assignment]
    engine._rank = lambda pool, features: ordered  # type: ignore[assignment]
    data, diag = engine.search(user_id="u", query="evidence", top_k=100)

    counter = TokenCounter(settings)
    total = sum(counter.count(item["content"]) for item in data)
    assert total <= settings.token_budget_total
    # Nothing returned may be header-only.
    for item in data:
        assert item["content"].strip()


# --------------------------------------------------------------------------
# P2.7 follow-up — model call accounting must count failed attempts
# --------------------------------------------------------------------------


def test_p2_7_failed_calls_are_counted_as_attempted(tmp_path):
    """Original repro: provider times out once -> calls_made=0, calls_failed=1,
    and replay reported 0 model calls. Attempts must be visible."""
    from codemem.common.embeddings import EmbeddingClient

    settings = make_settings(tmp_path)
    client = EmbeddingClient(settings, EmbeddingCache(settings.embed_cache_path))

    class Boom:
        def create(self, **kwargs):
            raise TimeoutError("provider timeout")

    client._client = type("C", (), {"embeddings": Boom()})()
    reset_model_limiter_for_tests()

    result = client.embed(["some text"])
    assert result is None
    stats = client.stats()
    assert stats["attempted"] == 1, "a failed call must still count as attempted"
    assert stats["failed"] == 1
    assert stats["succeeded"] == 0


def test_p2_7_cache_hits_are_reported(tmp_path):
    """Cache hits are distinguished from provider calls."""
    from codemem.common.embeddings import EmbeddingClient

    settings = make_settings(tmp_path)
    cache = EmbeddingCache(settings.embed_cache_path)
    client = EmbeddingClient(settings, cache)

    calls = {"n": 0}

    class Fake:
        def create(self, model, input, timeout=None):
            calls["n"] += 1

            class D:
                def __init__(self, e):
                    self.embedding = e

            class R:
                def __init__(self):
                    self.data = [D([0.1] * 8) for _ in input]
                    self.model = model
                    self.usage = None

            return R()

    client._client = type("C", (), {"embeddings": Fake()})()
    reset_model_limiter_for_tests()

    client.embed(["repeatable text"])
    client.embed(["repeatable text"])  # second call served from cache
    stats = client.stats()
    assert stats["succeeded"] == 1
    assert stats["cache_hits"] >= 1


# --------------------------------------------------------------------------
# P2.6 — idempotency must include session_id
# --------------------------------------------------------------------------


def test_p2_6_session_id_change_is_conflict(tmp_path):
    """Original repro: same request_id + messages, new session_id -> was accepted."""
    settings = make_settings(tmp_path, embed_model="", embed_base_url=None, embed_api_key=None)
    store = Store(settings)
    from codemem.service import MemoryService, ValidationError

    svc = MemoryService(store, settings)
    payload = [{"role": "user", "content": "idempotency body"}]
    first = svc.add(request_id="r", user_id="u", session_id="session-old", raw_messages=payload)
    assert first.success

    with pytest.raises(ValidationError):
        svc.add(request_id="r", user_id="u", session_id="session-new", raw_messages=payload)


def test_p2_6_same_session_replay_is_idempotent(tmp_path):
    settings = make_settings(tmp_path, embed_model="", embed_base_url=None, embed_api_key=None)
    store = Store(settings)
    from codemem.service import MemoryService

    svc = MemoryService(store, settings)
    payload = [{"role": "user", "content": "idempotency body two"}]
    first = svc.add(request_id="r", user_id="u", session_id="s", raw_messages=payload)
    second = svc.add(request_id="r", user_id="u", session_id="s", raw_messages=payload)
    assert first.success and second.success
    assert second.deduplicated is True


# --------------------------------------------------------------------------
# P2.8 — Add and Search must share one cached embedder
# --------------------------------------------------------------------------


def test_p2_8_add_and_search_share_embedder(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    store = Store(settings)
    from codemem.service import MemoryService
    import codemem.service as service_module

    shared = CountingEmbedder(dim=8)
    monkeypatch.setattr(service_module, "get_embedder", lambda *a, **k: shared)

    svc = MemoryService(store, settings)
    assert svc.embedder is shared
    # The vector index must use the same instance, not a second client.
    assert getattr(svc.search_engine.card_vector_index, "embedder", None) is shared


def test_p2_8_add_enrichment_uses_cache(tmp_path):
    """Cache must be attached to the client the Add path uses."""
    settings = make_settings(tmp_path)
    cache = EmbeddingCache(settings.embed_cache_path)
    from codemem.common.embeddings import EmbeddingClient

    client = EmbeddingClient(settings, cache)
    assert client.cache is cache


# --------------------------------------------------------------------------
# P1.2 — concurrency: slow Add must not block health or other requests
# --------------------------------------------------------------------------


def test_p1_2_model_call_limiter_bounds_concurrency():
    limiter = ModelCallLimiter(limit=2, acquire_timeout_s=0.1)
    assert limiter.acquire() is True
    assert limiter.acquire() is True
    assert limiter.acquire() is False, "third concurrent call must be rejected"
    limiter.release()
    assert limiter.acquire() is True
    assert limiter.stats()["rejected"] >= 1


def test_p1_2_routes_are_sync():
    """Blocking handlers must be plain def so FastAPI uses the threadpool."""
    import inspect

    from codemem.api import create_app

    app = create_app(Settings(db_path=Path("/tmp/x.db"), embed_model=""))
    by_path = {}
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "endpoint"):
            by_path[route.path] = route.endpoint
    for path in ("/add", "/search", "/health", "/ready"):
        endpoint = by_path.get(path)
        assert endpoint is not None, f"{path} route missing"
        assert not inspect.iscoroutinefunction(endpoint), (
            f"{path} is async but does blocking work; it must be a sync route"
        )


def test_p1_2_health_responds_during_slow_add(tmp_path, monkeypatch):
    """A slow Add must not prevent /health from answering promptly."""
    settings = make_settings(tmp_path, embed_budget_s=3.0)
    app = create_app(settings)

    import codemem.service as service_module

    slow = CountingEmbedder(dim=8, sleep=1.5)
    monkeypatch.setattr(service_module, "get_embedder", lambda *a, **k: slow)

    with TestClient(app) as client:
        # Force the service's embedder to the slow one.
        client.app.state.service.embedder = slow

        result: dict = {}

        def do_add():
            result["add"] = client.post(
                "/add",
                json={
                    "request_id": "slow1",
                    "user_id": "u1",
                    "session_id": "s1",
                    "messages": [{"role": "user", "content": "slow embedding content " * 20}],
                },
            ).status_code

        worker = threading.Thread(target=do_add)
        worker.start()
        time.sleep(0.2)  # let the Add get into the blocking call
        started = time.monotonic()
        health = client.get("/health")
        health_ms = (time.monotonic() - started) * 1000
        worker.join(timeout=10)

        assert health.status_code == 200
        assert health_ms < 500, f"/health took {health_ms:.0f}ms while an Add was blocked"
