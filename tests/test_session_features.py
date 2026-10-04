"""Session-feature fusion tests: F1 (first-message cosine) and F2 (rare
vocabulary union coverage) rank-fused into the session order.

The features are computed by the service; the fusion happens in ``assemble``.
These tests inject the feature dicts directly (no encoder), so the reordering
math is what is under test, plus the off-by-default and degradation paths.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from codemem.core.config import Settings
from codemem.search.evidence import assemble, session_terms


def make_settings(tmp_path, **overrides) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        dense_enabled=False,
        rerank_enabled=False,
        min_evidence_score=0.0,  # the gate must not cut the weaker session
        **overrides,
    )


class Candidate:
    def __init__(self, memory_id: int, final: float):
        self.memory_id = memory_id
        self.final = final


class Memory:
    def __init__(self, memory_id: int, session_id: str, chunk_id: int | None, text: str):
        self.id = memory_id
        self.chunk_id = chunk_id
        self.session_id = session_id
        self.kind = "chunk"
        self.text = text
        self.ts = 0
        self.created_at = "2026-01-01T00:00:00Z"
        self.superseded_by = None

    structural_kind = "code"


class Store:
    """Minimal stand-in: no chunks to fetch, texts live on the memory rows."""

    def fetch_chunks(self, user_id, chunk_ids):
        return {}

    def session_span(self, user_id, sessions):
        return {}


class Plan:
    keywords: list[str] = []


def _fixture():
    # Session "head" wins on chunk score; session "bridge" loses per chunk but
    # its first message and rare vocabulary match the query.
    memories = {
        1: Memory(1, "head", None, "quota retry backoff handler lock fixed"),
        2: Memory(2, "bridge", None, "calendar timezone drift migration notes"),
    }
    scored = [Candidate(1, 0.9), Candidate(2, 0.5)]
    return memories, scored


def test_fusion_flips_the_session_order(tmp_path):
    settings = make_settings(tmp_path, session_feature_fusion=True)
    memories, scored = _fixture()
    features = ({"head": 0.1, "bridge": 0.9}, {"head": 0.0, "bridge": 0.8})
    items = assemble(
        settings, Store(), "u1", Plan(), scored, memories,
        top_k=10, session_features=features,
    )
    sessions = []
    for item in items:
        memory = memories[item.memory_id]
        if memory.session_id not in sessions:
            sessions.append(memory.session_id)
    assert sessions[0] == "bridge"


def test_no_fusion_keeps_the_head_order(tmp_path):
    settings = make_settings(tmp_path)
    memories, scored = _fixture()
    features = ({"head": 0.1, "bridge": 0.9}, {"head": 0.0, "bridge": 0.8})
    items = assemble(
        settings, Store(), "u1", Plan(), scored, memories,
        top_k=10, session_features=features,
    )
    sessions = []
    for item in items:
        memory = memories[item.memory_id]
        if memory.session_id not in sessions:
            sessions.append(memory.session_id)
    assert sessions[0] == "head"


def test_off_by_default(tmp_path):
    assert Settings().session_feature_fusion is False


def test_session_terms_requires_topic_length():
    assert "quota" in session_terms("quota retry backoff handler")
    assert "a" not in session_terms("a b c quota")
    assert session_terms("") == set()


def test_llm_select_reorders_sessions(monkeypatch, tmp_path):
    """The LLM selection stage reorders session blocks and the assembler
    accepts it. Guards the wiring end to end with a stubbed relay — this
    path failed silently twice before (missing import, stale closure name),
    so the test runs against the real handle() pipeline."""
    import logging

    from codemem.api.app import create_app

    settings = Settings(
        data_dir=tmp_path / "data",
        dense_enabled=False,
        rerank_enabled=False,
        min_evidence_score=0.0,
        session_select_llm=True,
        llm_base_url="http://relay.test",
        llm_api_key="stub",
    )
    app = create_app(settings)

    def stub_chat(self, system: str, user: str) -> str:
        return "2 and 1"  # pick the second-listed session first

    from codemem.search.service import SearchPipeline

    monkeypatch.setattr(SearchPipeline, "_chat", stub_chat)
    logging.disable(logging.WARNING)
    with TestClient(app) as client:
        for sid, rid, body in (
            ("strong", "r1", "quota retry backoff handler quota retry backoff handler: lock fixed src/a.py"),
            ("bridge", "r2", "quota retry backoff handler part one; calendar timezone migration notes src/b.py"),
        ):
            client.post("/add", json={
                "request_id": rid, "user_id": "u1", "session_id": sid,
                "messages": [{"role": "user", "timestamp": 1, "content": body}],
            })
        response = client.post("/search", json={
            "query": "quota retry backoff handler", "user_id": "u1", "top_k": 10,
        })
        items = response.json()["data"]
        assert items
        store = client.app.state.container.store
        memory_ids = [int(i["id"].split("_")[1]) for i in items]
        sessions = store.session_map("u1", memory_ids)
        order = []
        for mid in memory_ids:
            sid = sessions.get(mid)
            if sid and sid not in order:
                order.append(sid)
        assert set(order) == {"strong", "bridge"}
        assert order[0] == "bridge", (
            "the stubbed LLM picked 'bridge' first; handle() must honour it"
        )
