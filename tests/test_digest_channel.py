"""Rewritten-memory digest cards: generation at Add, append-only delivery at
Search, exclusions everywhere else.

A digest is LLM text that IS returnable data[].content — the deliberate
relaxation of the verbatim invariant (digest_channel). These tests stub the
LLM at AddPipeline._llm_complete and the digest index/probe embeddings at the
service boundary; the append mechanics run the shipped code.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from codemem.add.pipeline import AddPipeline
from codemem.core.config import Settings

CARD = ("problem: the quota retry backoff handler duplicates requests | "
        "cause: the lock is released before the backoff completes | "
        "fix: serialise lock acquisition in src/a.py")


def make_app(monkeypatch, tmp_path, digest_reply, **overrides):
    from codemem.api.app import create_app

    settings = Settings(
        data_dir=tmp_path / "data",
        dense_enabled=False,
        rerank_enabled=False,
        digest_channel=True,
        llm_base_url="http://relay.test",
        llm_api_key="stub",
        **overrides,
    )
    app = create_app(settings)
    calls = []

    def fake_llm(self, system, user):
        calls.append(user)
        return digest_reply

    monkeypatch.setattr(AddPipeline, "_llm_complete", fake_llm)
    return TestClient(app), calls


def seed(client, session_id="s1", request_id="req:1"):
    return client.post("/add", json={
        "request_id": request_id, "user_id": "u1", "session_id": session_id,
        "messages": [
            {"role": "user", "timestamp": 1,
             "content": "issue: the quota retry handler duplicates requests"},
            {"role": "assistant", "timestamp": 2,
             "content": "investigating the retry loop now"},
        ],
    })


def digest_rows(client, user_id="u1"):
    store = client.app.state.container.store
    with store._read() as conn:  # noqa: SLF001 - test harness
        rows = conn.execute(
            "SELECT id, session_id, text FROM memory"
            " WHERE user_id = ? AND kind = 'digest'", (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def test_digest_generation_stores_issue_language_cards(monkeypatch, tmp_path):
    client, calls = make_app(
        monkeypatch, tmp_path,
        "[CARD] problem: quota retry duplicates requests | cause: lock released "
        "before backoff | fix: serialise acquisition src/a.py",
    )
    with client:
        r = seed(client)
        assert r.status_code == 200
        rows = digest_rows(client)
        assert len(rows) == 1
        assert rows[0]["session_id"] == "s1"
        assert "quota retry" in rows[0]["text"]
        assert len(calls) == 1


def test_digest_cards_delivered_via_side_channel(monkeypatch, tmp_path):
    """With digest_channel on, the stored digest is appended to the payload
    by probe cosine — the returnable-LLM-text relaxation."""
    app, _ = make_app(
        monkeypatch, tmp_path,
        "[CARD] problem: quota retry duplicates requests | cause: lock released "
        "before backoff | fix: serialise acquisition src/a.py",
    )
    with app as client:
        seed(client)
        rows = digest_rows(client)
        assert rows
        # dense is off in this fixture, so give the digest its vector directly
        client.app.state.container.store.store_vectors(
            "u1", [(rows[0]["id"], [0.9, 0.1, 0.0, 0.0])])
        pipeline = client.app.state.container.search
        monkeypatch.setattr(pipeline.retriever, "embedder", _ProbeEmbedder(
            probe=[0.9, 0.1, 0.0, 0.0]))

        response = client.post("/search", json={
            "query": "quota retry duplicates", "user_id": "u1", "top_k": 10,
        })
        items = response.json()["data"]
        assert any("quota retry duplicates" in i["content"] for i in items), (
            "the digest must be appended to the payload"
        )
        scores = [i["score"] for i in items]
        assert all(a >= b for a, b in zip(scores, scores[1:]))


class _ProbeEmbedder:
    def __init__(self, probe, keyed=None):
        self.probe = probe
        self.keyed = keyed or {}
        self.available = True

    def embed(self, texts):
        return [
            next((v for k, v in self.keyed.items() if f"mem_{k}" in t), self.probe)
            for t in texts
        ]


def test_digest_channel_off_keeps_digests_out(monkeypatch, tmp_path):
    from codemem.api.app import create_app

    settings = Settings(
        data_dir=tmp_path / "data", dense_enabled=False, rerank_enabled=False,
    )
    app = create_app(settings)

    def unexpected(self, system, user):
        raise AssertionError("digest_channel off; the LLM must not be called")

    monkeypatch.setattr(AddPipeline, "_llm_complete", unexpected)
    with TestClient(app) as client:
        seed(client)
        assert digest_rows(client) == []


def test_digest_channel_off_excludes_digest_rows(monkeypatch, tmp_path):
    """A digest row that exists in the store (e.g. flag flipped after Add)
    must not leak into the payload through the normal assembler path."""
    app, _ = make_app(
        monkeypatch, tmp_path,
        "[CARD] problem: quota retry duplicates requests | cause: lock | fix: src/a.py",
    )
    with app as client:
        seed(client)
        rows = digest_rows(client)
        assert rows
        # flip delivery off for the search (simulate flag change after Add)
        client.app.state.container.settings.digest_channel = False
        response = client.post("/search", json={
            "query": "quota retry duplicates requests", "user_id": "u1", "top_k": 10,
        })
        blob = "\n".join(i["content"] for i in response.json()["data"])
        assert "problem: quota retry duplicates" not in blob


def test_re_add_reuses_cached_digest(monkeypatch, tmp_path):
    client, calls = make_app(
        monkeypatch, tmp_path,
        "[CARD] problem: quota retry duplicates requests | cause: lock | fix: src/a.py",
    )
    with client:
        seed(client, request_id="req:1")
        seed(client, request_id="req:2")
        assert len(digest_rows(client)) == 1
        assert len(calls) == 1
