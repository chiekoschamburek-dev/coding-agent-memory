"""Session-score estimator tests: max vs top-k mass.

The shipped session score is the max over the session's admitted members.
``session_score_topk > 1`` replaces it with the sum of the top-k member
scores, so a session with several moderately matching chunks can outrank one
with a single lucky high scorer — the estimator failure the attribution work
measured (a session's head chunk names the task file only 12.9 % of the
time). The noise gate keeps reading the head either way.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from codemem.core.config import Settings


def make_app(tmp_path, **overrides):
    from codemem.api.app import create_app

    settings = Settings(
        data_dir=tmp_path / "data",
        dense_enabled=False,
        rerank_enabled=False,
        min_evidence_score=0.0,  # the gate must not cut the weaker session
        **overrides,
    )
    return TestClient(create_app(settings))


def add(client, *, session_id: str, request_id: str, body: str):
    return client.post(
        "/add",
        json={
            "request_id": request_id,
            "user_id": "u1",
            "session_id": session_id,
            "messages": [
                {"role": "user", "timestamp": 1, "content": body},
            ],
        },
    )


def session_order(client) -> list[str]:
    store = client.app.state.container.store
    response = client.post(
        "/search",
        json={"query": "quota retry backoff handler", "user_id": "u1", "top_k": 10},
    )
    items = response.json()["data"]
    memory_ids = [int(item["id"].split("_")[1]) for item in items]
    sessions = store.session_map("u1", memory_ids)
    order: list[str] = []
    for mid in memory_ids:
        session = sessions.get(mid)
        if session and session not in order:
            order.append(session)
    return order


def test_default_max_lets_one_lucky_chunk_lead(tmp_path):
    """Shipped estimator: the session whose single chunk scores highest leads."""
    client = make_app(tmp_path)
    with client:
        add(client, session_id="lonely", request_id="r1",
            body="quota retry backoff handler quota retry backoff handler: the lock was missing. fixed in src/a.py")
        add(client, session_id="broad", request_id="r2",
            body="quota retry backoff handler part one; also touched the scheduler config, the logger setup and the metrics exporter")
        add(client, session_id="broad", request_id="r3",
            body="quota retry backoff handler part two; reviewed the pagination path, renamed two helpers, updated the docs page")

    assert session_order(client)[0] == "lonely"


def test_topk_mass_lets_broad_evidence_lead(tmp_path):
    """top-k mass: two moderately matching chunks outweigh one strong match."""
    client = make_app(tmp_path, session_score_topk=2)
    with client:
        add(client, session_id="lonely", request_id="r1",
            body="quota retry backoff handler quota retry backoff handler: the lock was missing. fixed in src/a.py")
        add(client, session_id="broad", request_id="r2",
            body="quota retry backoff handler part one; also touched the scheduler config, the logger setup and the metrics exporter")
        add(client, session_id="broad", request_id="r3",
            body="quota retry backoff handler part two; reviewed the pagination path, renamed two helpers, updated the docs page")

    order = session_order(client)
    assert "broad" in order and "lonely" in order
    assert order.index("broad") < order.index("lonely")


def test_topk_one_matches_the_max_estimator(tmp_path):
    """k=1 must behave exactly like the shipped max estimator."""
    client_a = make_app(tmp_path / "a")
    client_b = make_app(tmp_path / "b", session_score_topk=1)
    for client in (client_a, client_b):
        with client:
            add(client, session_id="lonely", request_id="r1",
                body="quota retry backoff handler quota retry backoff handler: the lock was missing. fixed in src/a.py")
            add(client, session_id="broad", request_id="r2",
                body="quota retry backoff handler part one; also touched the scheduler config, the logger setup and the metrics exporter")
    assert session_order(client_a) == session_order(client_b)
