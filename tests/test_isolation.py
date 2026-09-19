"""Isolation and retention.

``user_id`` is the only isolation boundary, and rule 2 forbids sharing or
retrieving memory across user_ids, tasks, samples, or teams. These tests try to
break that, including through the less obvious routes: entity matching, the
recency channel, and the FTS index.
"""

from __future__ import annotations

import pytest

from codemem.index.store import Store

SHARED_TEXT = "The retry count is defined in src/net/retry.py and defaults to 5."


def _add(client, user_id, request_id, content=SHARED_TEXT, session_id="s1"):
    return client.post(
        "/add",
        json={
            "request_id": request_id,
            "user_id": user_id,
            "session_id": session_id,
            "messages": [{"role": "user", "content": content, "timestamp": 1704067200000}],
        },
    )


def test_search_cannot_read_another_users_memory(client):
    _add(client, "alice", "r-a", "Alice stores the deploy key rotation policy here.")
    data = client.post(
        "/search",
        json={"query": "deploy key rotation policy", "user_id": "bob", "top_k": 100},
    ).json()["data"]
    assert data == []


def test_identical_text_under_two_users_stays_isolated(client):
    _add(client, "alice", "r-a", SHARED_TEXT)
    _add(client, "bob", "r-b", SHARED_TEXT)

    for user_id in ("alice", "bob"):
        data = client.post(
            "/search",
            json={"query": "retry count default", "user_id": user_id, "top_k": 100},
        ).json()["data"]
        assert len(data) == 1, f"{user_id} saw {len(data)} copies"

    # A user with no writes at all must see nothing, however similar the corpus.
    assert (
        client.post(
            "/search",
            json={"query": "retry count default", "user_id": "carol", "top_k": 100},
        ).json()["data"]
        == []
    )


def test_entity_channel_is_user_scoped(client):
    """Identifier matching is the strongest recall channel, so verify it
    filters by user_id rather than only by identifier."""
    _add(
        client,
        "alice",
        "r-a",
        "Fixed IndexError in src/parser/tokenizer.py by guarding an empty buffer.",
    )
    data = client.post(
        "/search",
        json={
            "query": "What caused the IndexError in src/parser/tokenizer.py?",
            "user_id": "mallory",
            "top_k": 100,
        },
    ).json()["data"]
    assert data == []


def test_recency_channel_is_user_scoped(client):
    """Recency is query-independent, so it is the easiest channel to leak
    through; it must still be scoped."""
    _add(client, "alice", "r-a", "A very recent note about the build pipeline.")
    data = client.post(
        "/search",
        json={"query": "build pipeline", "user_id": "eve", "top_k": 100},
    ).json()["data"]
    assert data == []


def test_delete_user_removes_every_trace(client, settings):
    from codemem.index.store import Store

    _add(client, "alice", "r-a", SHARED_TEXT)
    _add(client, "bob", "r-b", SHARED_TEXT)

    store = Store(settings)
    try:
        removed = store.delete_user("alice")
        assert removed["memory"] >= 1
        assert removed["raw_message"] >= 1
        assert removed["request_seen"] >= 1

        # Alice is gone from both the table and the FTS index.
        assert store.counts()[1] == 1
        rows = store.bm25_search("alice", "retry count default", 10)
        assert rows == []
        # Bob is untouched.
        assert store.bm25_search("bob", "retry count default", 10) != []
    finally:
        store.close()


def test_deleted_user_cannot_be_searched(client, settings):
    _add(client, "alice", "r-a", SHARED_TEXT)
    store = Store(settings)
    try:
        store.delete_user("alice")
    finally:
        store.close()
    data = client.post(
        "/search", json={"query": "retry count default", "user_id": "alice", "top_k": 100}
    ).json()["data"]
    assert data == []


def test_cross_user_entity_timeline_is_isolated(client, settings):
    """The governance structures (entity timeline, repo profile) must also be
    scoped, since they could otherwise leak corpus shape across users."""
    _add(client, "alice", "r-a", "Alpha uses src/alpha/module.py heavily.")
    _add(client, "bob", "r-b", "Beta uses src/beta/other.py heavily.")

    store = Store(settings)
    try:
        alice_profile = store.repo_profile("alice")
        bob_profile = store.repo_profile("bob")
        alice_values = {v for vals in alice_profile.values() for v, _ in vals}
        bob_values = {v for vals in bob_profile.values() for v, _ in vals}
        assert "src/alpha" in alice_values or "module.py" in alice_values
        assert not (alice_values & bob_values), "repo profiles leaked across users"
    finally:
        store.close()
