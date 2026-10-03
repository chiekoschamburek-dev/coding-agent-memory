"""Experience-card invariants, one test per guarantee in docs/DESIGN.md.

A card is one generated overview per session, stored as a ``memory`` row with
``kind='card'``. It competes in recall and scoring like any memory row — that
is the point: the cross-encoder reads (query, overview) as a session-level
pair — but it is never returned, never reaches ``data[].content``, and a
session whose only members are cards emits nothing. Add degrades instead of
failing when the overview cannot be produced.

The LLM is stubbed at ``AddPipeline._llm_complete``; everything else runs the
shipped code paths.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from codemem.add.pipeline import AddPipeline
from codemem.core.config import Settings

SHARED_BODY = (
    "the checkout flow returns 500 for empty carts. "
    "[tool Edit] fixed the migration retry backoff handler in src/cart.py."
)


def card_settings(tmp_path, **overrides) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        dense_enabled=False,
        rerank_enabled=False,
        card_enabled=True,
        llm_base_url="http://relay.test",
        llm_api_key="stub-key",
        **overrides,
    )


def make_app(monkeypatch, tmp_path, stub, **overrides):
    """Test client with cards on and ``stub`` standing in for the LLM."""
    from fastapi.testclient import TestClient

    from codemem.api.app import create_app

    settings = card_settings(tmp_path, **overrides)
    app = create_app(settings)
    calls: list[str] = []

    def fake_llm(self, system: str, user: str) -> str:
        calls.append(user)
        return stub(user)

    monkeypatch.setattr(AddPipeline, "_llm_complete", fake_llm)
    return TestClient(app), calls


def add_session(
    client,
    *,
    user_id: str,
    session_id: str,
    request_id: str,
    body: str = SHARED_BODY,
    ts: int = 1_700_000_000_000,
):
    return client.post(
        "/add",
        json={
            "request_id": request_id,
            "user_id": user_id,
            "session_id": session_id,
            "messages": [
                {"role": "user", "timestamp": ts, "content": body},
                {
                    "role": "assistant",
                    "timestamp": ts + 1000,
                    "content": f"[tool Edit] follow-up on {session_id}. {body}",
                },
            ],
        },
    )


def card_row(client, user_id: str) -> dict | None:
    store = client.app.state.container.store
    with store._read() as conn:  # noqa: SLF001 - test harness
        row = conn.execute(
            "SELECT id, kind, chunk_id, text, ts, created_at FROM memory"
            " WHERE user_id = ? AND kind = 'card'",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def test_card_is_written_once_and_cache_survives_a_re_add(monkeypatch, tmp_path):
    client, calls = make_app(
        monkeypatch,
        tmp_path,
        lambda user: "Session overview: the assistant investigated a checkout failure.",
    )
    with client:
        # Different request ids, same content: idempotency must not be what
        # saves the second card — the overview cache is.
        add_session(client, user_id="u1", session_id="s1", request_id="req:1")
        add_session(client, user_id="u1", session_id="s1", request_id="req:2")

        row = card_row(client, "u1")
        assert row is not None
        # The card covers the session, not one chunk, and is generated text.
        assert row["chunk_id"] is None
        assert row["kind"] == "card"
        assert "overview" in row["text"].lower()
        # One LLM call despite two Adds: the second hit the content-hash cache.
        assert len(calls) == 1


def test_card_source_time_is_the_earliest_message(monkeypatch, tmp_path):
    client, _ = make_app(
        monkeypatch,
        tmp_path,
        lambda user: "Session overview: a checkout failure investigation.",
    )
    with client:
        add_session(
            client,
            user_id="u1",
            session_id="s1",
            request_id="req:1",
            ts=1_700_000_123_000,
        )
        row = card_row(client, "u1")
        assert row is not None
        assert row["ts"] == 1_700_000_123_000


def test_card_never_enters_payload(monkeypatch, tmp_path):
    """A query matching the card's overview and every chunk still returns only
    chunk ids: data[].content is always a verbatim span of Add input."""
    client, _ = make_app(
        monkeypatch,
        tmp_path,
        lambda user: "Session overview: checkout cart validation failure.",
    )
    with client:
        add_session(client, user_id="u1", session_id="s1", request_id="req:1")
        row = card_row(client, "u1")
        assert row is not None

        response = client.post(
            "/search",
            json={
                "query": "overview checkout cart validation failure",
                "user_id": "u1",
                "top_k": 10,
            },
        )
        items = response.json()["data"]
        assert items, "chunks matched lexically, the payload must not be empty"
        assert str(row["id"]) not in {item["id"] for item in items}
        # And no returned content is the generated overview itself.
        assert all(item["content"] != row["text"] for item in items)


def test_card_alone_qualifies_nothing(monkeypatch, tmp_path):
    """A session whose only admitted member is its card emits nothing: the
    card cannot walk an approximate hit past the noise gate."""
    client, _ = make_app(
        monkeypatch,
        tmp_path,
        lambda user: "Session overview: debugging the quantum flux capacitor "
        "calibration drift across temperature cycles.",
    )
    with client:
        # Chunks share nothing with the query; only the overview does.
        add_session(client, user_id="u1", session_id="s1", request_id="req:1")

        response = client.post(
            "/search",
            json={
                "query": "quantum flux capacitor calibration drift",
                "user_id": "u1",
                "top_k": 10,
            },
        )
        assert response.status_code == 200
        assert response.json()["data"] == []


def test_card_boosts_its_session_rank(monkeypatch, tmp_path):
    """The comparable session-level object changes which session leads: two
    sessions with byte-identical, equally-matching chunks, where only the
    card overview differs in how densely it matches the query."""
    zebra_card = (
        "Session overview: zebra migration retry backoff handler; the zebra "
        "migration retry backoff handler was fixed and tested."
    )
    plain_card = "Session overview: migration retry backoff handler, routine work."
    client, _ = make_app(
        monkeypatch,
        tmp_path,
        lambda user: zebra_card if "ZQ-91" in user else plain_card,
    )
    with client:
        add_session(
            client,
            user_id="u1",
            session_id="plain",
            request_id="req:1",
            body=f"{SHARED_BODY} ticket AB-12",
        )
        add_session(
            client,
            user_id="u1",
            session_id="zebra",
            request_id="req:2",
            body=f"{SHARED_BODY} ticket ZQ-91",
        )

        response = client.post(
            "/search",
            json={
                "query": "zebra migration retry backoff handler",
                "user_id": "u1",
                "top_k": 10,
            },
        )
        items = response.json()["data"]
        assert items, "both sessions' chunks match lexically"

        store = client.app.state.container.store
        memory_ids = [int(item["id"].split("_")[1]) for item in items]
        sessions = store.session_map("u1", memory_ids)
        order = [sessions.get(i) for i in memory_ids]
        assert "zebra" in order and "plain" in order, order
        assert order.index("zebra") < order.index("plain"), (
            "the card's denser session-level match must lift its session "
            "ahead of the one whose chunks are byte-identical"
        )


def test_card_failure_degrades_add_but_never_fails_it(monkeypatch, tmp_path):
    from codemem.api.app import create_app

    settings = card_settings(tmp_path)
    app = create_app(settings)

    def broken_llm(self, system: str, user: str) -> str:
        raise RuntimeError("relay down")

    monkeypatch.setattr(AddPipeline, "_llm_complete", broken_llm)
    with TestClient(app) as client:
        response = add_session(client, user_id="u1", session_id="s1", request_id="req:1")
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert card_row(client, "u1") is None
        # The raw memory is still durably searchable.
        search = client.post(
            "/search",
            json={"query": "checkout flow 500", "user_id": "u1", "top_k": 5},
        )
        assert search.json()["data"]


def test_no_card_without_flag(monkeypatch, tmp_path):
    from codemem.api.app import create_app

    settings = Settings(
        data_dir=tmp_path / "data",
        dense_enabled=False,
        rerank_enabled=False,
    )
    app = create_app(settings)

    def unexpected(self, system: str, user: str) -> str:
        raise AssertionError("cards are disabled; the LLM must not be called")

    monkeypatch.setattr(AddPipeline, "_llm_complete", unexpected)
    with TestClient(app) as client:
        add_session(client, user_id="u1", session_id="s1", request_id="req:1")
        assert card_row(client, "u1") is None
