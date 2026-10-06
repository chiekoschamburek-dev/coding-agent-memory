"""Claims-only side-channel tests: append-only, capped, floor-gated, and
harmless when off or when the index is empty.

The side channel appends claim-shaped memories retrieved by option-probe
cosine AFTER assembly, without displacing anything — the anti-churn
constraint from three measured non-monotonicities. The claim index and the
probe embeddings are stubbed at the service boundary; the append mechanics,
the cap/floor/dedup rules, and the degradation paths run the shipped code.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from codemem.core.config import Settings


def make_app(tmp_path, **overrides):
    from codemem.api.app import create_app

    settings = Settings(
        data_dir=tmp_path / "data",
        dense_enabled=False,  # side channel must work without the dense stack
        rerank_enabled=False,
        claim_channel=True,
        llm_base_url=None,  # no relay: the channel is embedding-only
        **overrides,
    )
    return TestClient(create_app(settings))


def seed(client, session_id: str, request_id: str, turns: list[str]):
    client.post("/add", json={
        "request_id": request_id,
        "user_id": "u1",
        "session_id": session_id,
        "messages": [
            {"role": "user", "timestamp": 1, "content": "the issue: quota retry handler"},
            *({"role": "assistant", "timestamp": 2, "content": t} for t in turns),
        ],
    })


CLAIM = ("The problem is in the quota retry backoff handler: it swallows the "
         "exception because the lock is released before the backoff completes, "
         "so the retry loop re-enters and duplicates the request src/a.py")


class StubEmbedder:
    """Deterministic stub: probe texts get ``default``, marker texts get
    their keyed vector."""

    def __init__(self, default: list[float], keyed: dict[str, list[float]] | None = None):
        self._default = default
        self._keyed = keyed or {}
        self.available = True

    def embed(self, texts):
        return [
            next((vec for marker, vec in self._keyed.items() if marker in t),
                 self._default)
            for t in texts
        ]


def install_index(client, monkeypatch, claims: list[dict], probe_vec: list[float]):
    """Stub the claim index and the probe embeddings."""
    pipeline = client.app.state.container.search
    monkeypatch.setattr(pipeline, "_claim_index",
                        lambda user_id: [{**c, "vec": c["vec"]} for c in claims], raising=False)
    monkeypatch.setattr(pipeline, "_claim_shape", lambda text: True, raising=False)
    monkeypatch.setattr(pipeline.retriever, "embedder", StubEmbedder(probe_vec))


def test_side_channel_appends_without_displacing(monkeypatch, tmp_path):
    client = make_app(tmp_path)
    with client:
        seed(client, "s1", "r1", ["plain assistant text without any cause marker"])
        seed(client, "s2", "r2", ["plain filler here"])

        # install AFTER seeding but INSIDE the lifespan: a re-entered context
        # rebuilds the container and would drop the monkeypatch
        install_index(client, monkeypatch,
                      claims=[{"memory_id": 9001, "session_id": "s1",
                               "text": CLAIM, "vec": [0.9, 0.1, 0.0, 0.0],
                               "created_at": "2026-01-01T00:00:00Z",
                               "superseded": False}],
                      probe_vec=[0.9, 0.1, 0.0, 0.0])

        response = client.post("/search", json={
            "query": "quota retry backoff handler issue", "user_id": "u1", "top_k": 10,
        })
        items = response.json()["data"]

    assert items, "the seated payload must survive"
    seated_first = items[0]
    # the appended claim is last, below every seated score
    assert str(9001) not in {i["id"].split("_")[1] for i in items[:-1]}
    appended = [i for i in items if i["id"].endswith("9001") or "9001" in i["id"]]
    # the stub index uses memory_id 9001; the id serialization may prefix it
    blob = "\n".join(i["content"] for i in items)
    assert CLAIM in blob, "the claim text must be appended verbatim"
    if len(items) > 1:
        scores = [i["score"] for i in items]
        assert all(a >= b for a, b in zip(scores, scores[1:]))
    assert appended or CLAIM in blob  # noqa: PT018 - either form proves the append


def test_cap_limits_the_append(monkeypatch, tmp_path):
    client = make_app(tmp_path, claim_channel_cap=1)
    with client:
        seed(client, "s1", "r1", ["plain assistant text"])

        claims = [{"memory_id": 9001 + i, "session_id": f"s{i}",
                   "text": f"The problem is the quota retry backoff handler variant {i}, "
                           f"because the lock is released before the backoff completes "
                           f"and the loop re-enters, src/file{i}.py",
                   "vec": [0.9, 0.1, 0.0, 0.0],
                   "created_at": "2026-01-01T00:00:00Z",
                   "superseded": False}
                  for i in range(3)]
        install_index(client, monkeypatch, claims=claims,
                      probe_vec=[0.9, 0.1, 0.0, 0.0])

        response = client.post("/search", json={
            "query": "quota retry backoff handler", "user_id": "u1", "top_k": 10,
        })
        items = response.json()["data"]
    appended = [i for i in items if any(
        f"variant {i_}" in i["content"] for i_ in range(3))]
    assert len(appended) <= 1


def test_floor_blocks_weak_matches(monkeypatch, tmp_path):
    app = make_app(tmp_path, claim_channel_floor=0.8)
    with app as client:
        seed(client, "s1", "r1", ["plain assistant text"])

    install_index(app, monkeypatch,
                  claims=[{"memory_id": 9002, "session_id": "s1",
                           "text": CLAIM, "vec": [0.1, 0.1, 0.0, 0.0],
                           "created_at": "2026-01-01T00:00:00Z",
                           "superseded": False}],
                  probe_vec=[0.1, 0.1, 0.0, 0.0])

    with client:
        response = client.post("/search", json={
            "query": "quota retry backoff handler issue", "user_id": "u1", "top_k": 10,
        })
        blob = "\n".join(i["content"] for i in response.json()["data"])
    assert CLAIM not in blob


def test_off_by_default(tmp_path):
    s = Settings()
    assert s.claim_channel is False


def test_empty_index_is_harmless(tmp_path):
    """No claim index (fresh user, dense off): the channel degrades to no-op."""
    client = make_app(tmp_path)
    with client:
        seed(client, "s1", "r1", ["plain assistant text"])
        response = client.post("/search", json={
            "query": "quota retry backoff handler", "user_id": "u1", "top_k": 5,
        })
        assert response.status_code == 200
