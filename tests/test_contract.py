"""Contract and consistency tests (plan.md section 8.1).

These are the gate for stage 0: they must pass before any model token is
spent on extraction work.
"""

from __future__ import annotations

import concurrent.futures

import pytest


SAMPLE_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Our auth service started throwing a 401 after the token refresh change.\n\n"
            "```python\n"
            "def refresh_token(user_id):\n"
            "    return jwt.encode({'sub': user_id}, SECRET, algorithm='HS256')\n"
            "```\n\n"
            "The stack trace was:\n"
            "Traceback (most recent call last):\n"
            '  File "src/auth/token.py", line 42, in refresh_token\n'
            "    raise TokenExpiredError('refresh window elapsed')\n"
            "TokenExpiredError: refresh window elapsed\n\n"
            "We ran the tests:\n$ pytest tests/test_token.py -q\n"
            "3 passed, 1 failed in 0.42s\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "The cause was that the refresh window was compared against the wrong clock.\n"
            "Fix: compare against the monotonic clock instead of wall time.\n"
            "After the change the suite reported:\n"
            "4 passed in 0.51s\n"
        ),
    },
]


def add_sample(client, *, user_id="u1", request_id="r1", session_id="s1", messages=None):
    return client.post(
        "/add",
        json={
            "request_id": request_id,
            "messages": messages if messages is not None else SAMPLE_MESSAGES,
            "user_id": user_id,
            "session_id": session_id,
        },
    )


# --------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------


def test_add_echoes_identifiers_exactly(client):
    response = add_sample(client)
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "success": True,
        "request_id": "r1",
        "user_id": "u1",
        "session_id": "s1",
    }


def test_add_then_search_finds_the_card_immediately(card_client):
    client = card_client
    add_sample(client)
    response = client.post(
        "/search",
        json={"query": "TokenExpiredError refresh window", "user_id": "u1", "top_k": 100},
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data, "search must find evidence written by a preceding add"
    assert all(item["content"].strip() for item in data)
    assert all(item["id"] for item in data)


def test_search_returns_empty_data_array_not_missing(client):
    response = client.post(
        "/search",
        json={"query": "nothing like this exists anywhere", "user_id": "nobody", "top_k": 100},
    )
    assert response.status_code == 200
    assert response.json() == {"data": []}


def test_search_never_exceeds_top_k(client):
    add_sample(client)
    response = client.post(
        "/search",
        json={"query": "token refresh auth", "user_id": "u1", "top_k": 2},
    )
    assert len(response.json()["data"]) <= 2


def test_top_k_zero_returns_empty(client):
    add_sample(client)
    response = client.post(
        "/search",
        json={"query": "token", "user_id": "u1", "top_k": 0},
    )
    assert response.status_code == 200
    assert response.json() == {"data": []}


def test_score_is_optional_but_monotone_when_present(client):
    add_sample(client)
    data = client.post(
        "/search",
        json={"query": "TokenExpiredError", "user_id": "u1", "top_k": 100},
    ).json()["data"]
    scores = [item["score"] for item in data if item.get("score") is not None]
    assert scores == sorted(scores, reverse=True), "higher score must mean more relevant"


def test_ids_and_content_are_stable_across_identical_searches(client):
    add_sample(client)
    payload = {"query": "refresh window clock", "user_id": "u1", "top_k": 10}
    first = client.post("/search", json=payload).json()
    second = client.post("/search", json=payload).json()
    assert first == second


def test_created_at_is_persistence_time(card_client):
    """Cards carry the extraction/persistence time; api_guide.md allows either
    the source time or the persistence time for ``created_at``."""
    client = card_client
    add_sample(client, user_id="u_ts", request_id="r_ts",
               messages=[{"role": "user", "content": "no timestamp on this one at all"}])
    data = client.post(
        "/search", json={"query": "timestamp", "user_id": "u_ts", "top_k": 10}
    ).json()["data"]
    assert data
    assert all(item.get("created_at") for item in data), (
        "a card always knows when it was persisted"
    )


# --------------------------------------------------------------------------
# idempotency
# --------------------------------------------------------------------------


def test_duplicate_add_is_idempotent(client):
    first = add_sample(client)
    second = add_sample(client)
    assert first.json() == second.json()

    data = client.post(
        "/search", json={"query": "TokenExpiredError", "user_id": "u1", "top_k": 100}
    ).json()["data"]
    fingerprints = [" ".join(item["content"].split()) for item in data]
    assert len(fingerprints) == len(set(fingerprints)), "duplicate add must not duplicate memory"


def test_same_key_different_body_conflicts(client):
    add_sample(client)
    conflicting = [{"role": "user", "content": "completely different content"}]
    response = add_sample(client, messages=conflicting)
    assert response.status_code == 400


def test_concurrent_duplicate_add_does_not_duplicate(client):
    def fire():
        return add_sample(client).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        codes = list(pool.map(lambda _: fire(), range(6)))

    assert all(code in (200, 409) for code in codes)
    assert 200 in codes
    data = client.post(
        "/search", json={"query": "TokenExpiredError refresh", "user_id": "u1", "top_k": 100}
    ).json()["data"]
    fingerprints = [" ".join(item["content"].split()) for item in data]
    assert len(fingerprints) == len(set(fingerprints))


def test_distinct_request_ids_same_content_both_persist(client):
    add_sample(client, request_id="r_a")
    add_sample(client, request_id="r_b")
    data = client.post(
        "/search", json={"query": "TokenExpiredError", "user_id": "u1", "top_k": 100}
    ).json()["data"]
    ids = [item["id"] for item in data]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------
# isolation (plan.md 8.2)
# --------------------------------------------------------------------------


def test_unknown_user_sees_nothing(client):
    add_sample(client, user_id="alice", request_id="ra")
    response = client.post(
        "/search", json={"query": "TokenExpiredError refresh window", "user_id": "bob", "top_k": 100}
    )
    assert response.json() == {"data": []}


def test_same_query_and_paths_do_not_leak_across_users(card_client):
    client = card_client
    payload = [
        {
            "role": "user",
            "content": "src/auth/token.py raises TokenExpiredError in refresh_token",
        }
    ]
    add_sample(client, user_id="tenant_a", request_id="r1", messages=payload)
    add_sample(client, user_id="tenant_b", request_id="r1", messages=payload)

    # Same request_id is legal across scopes; results must stay separated.
    data_a = client.post(
        "/search", json={"query": "src/auth/token.py", "user_id": "tenant_a", "top_k": 100}
    ).json()["data"]
    data_b = client.post(
        "/search", json={"query": "src/auth/token.py", "user_id": "tenant_b", "top_k": 100}
    ).json()["data"]
    assert data_a and data_b
    assert not ({item["id"] for item in data_a} & {item["id"] for item in data_b})


# --------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "messages, expected_detail",
    [
        ([], "empty"),
        ([{"role": "system", "content": "x"}], "role"),
        ([{"role": "user"}], "content"),
        ([{"role": "user", "content": 123}], "string"),
        ([{"role": "user", "content": "ok", "timestamp": "2024"}], "timestamp"),
    ],
)
def test_invalid_add_requests_are_rejected_with_reason(client, messages, expected_detail):
    response = add_sample(client, messages=messages)
    assert response.status_code == 400
    assert expected_detail in response.json()["detail"].lower()


def test_missing_required_fields_rejected(client, legacy_status_codes=""):
    # Missing keys are rejected with a specifically named field, not a generic
    # parse error, so an integrator can tell what to fix.
    for body in (
        {"messages": []},
        {"request_id": "r", "user_id": "u", "session_id": "s"},
        {},
    ):
        response = client.post("/add", json=body)
        assert response.status_code in (400, 422)
        assert "request_id" in response.json()["detail"] or "messages" in response.json()["detail"]

    assert client.post("/search", json={"query": "x"}).status_code in (400, 422)


def test_search_input_validation(client):
    assert (
        client.post("/search", json={"query": "", "user_id": "u", "top_k": 10}).status_code == 400
    )
    assert (
        client.post("/search", json={"query": "x", "user_id": "u", "top_k": -1}).status_code == 400
    )
    assert (
        client.post("/search", json={"query": "x", "user_id": "u", "top_k": 10_000}).status_code
        == 400
    )
    assert (
        client.post(
            "/search", json={"query": "x", "user_id": "u", "top_k": 10, "options": ["a", 2]}
        ).status_code
        == 400
    )


def test_unicode_and_long_and_plain_text_are_handled(card_client):
    client = card_client
    payload = [
        {"role": "user", "content": "普通的纯字符串，没有任何结构。" * 200},
        {"role": "assistant", "content": "emoji 🚀 与 mixed ASCII 混合内容 " * 50},
    ]
    response = add_sample(client, user_id="u_uni", request_id="r_uni", messages=payload)
    assert response.status_code == 200
    data = client.post(
        "/search", json={"query": "普通的纯字符串", "user_id": "u_uni", "top_k": 100}
    ).json()["data"]
    assert data


def test_options_are_accepted_but_never_stored_as_facts(client):
    add_sample(client, user_id="u_opt", request_id="r_opt")
    response = client.post(
        "/search",
        json={
            "query": "What caused the 401?",
            "options": ["A. The refresh window used wall time", "B. TLS misconfiguration"],
            "user_id": "u_opt",
            "top_k": 100,
        },
    )
    assert response.status_code == 200
    # Options must not appear as memory content.
    for item in response.json()["data"]:
        assert "B. TLS misconfiguration" not in item["content"]

    # And they must not be retrievable as if they were stored memory.
    leak = client.post(
        "/search",
        json={"query": "TLS misconfiguration", "user_id": "u_opt", "top_k": 100},
    ).json()["data"]
    assert not any("TLS misconfiguration" in item["content"] for item in leak)


# --------------------------------------------------------------------------
# ops surface
# --------------------------------------------------------------------------


def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json() == {"ready": True}


def test_health_does_not_leak_secrets(client):
    body = client.get("/health").text
    assert "sk-" not in body
    assert "api_key" not in body.lower()


def test_purge_removes_scope(client):
    add_sample(client, user_id="u_purge", request_id="r_purge")
    removed = client.post("/admin/purge", json={"user_id": "u_purge"}).json()
    assert removed["removed"]["chunks"] > 0
    data = client.post(
        "/search", json={"query": "TokenExpiredError", "user_id": "u_purge", "top_k": 100}
    ).json()["data"]
    assert data == []


def test_auth_enforced_when_key_configured(tmp_path):
    from fastapi.testclient import TestClient

    from codemem.api import create_app
    from codemem.common.config import Settings

    cfg = Settings(db_path=tmp_path / "auth.db", api_key="secret-key", embed_model="")
    app = create_app(cfg)
    with TestClient(app) as authed:
        payload = {"query": "x", "user_id": "u", "top_k": 10}
        assert authed.post("/search", json=payload).status_code == 401
        assert (
            authed.post("/search", json=payload, headers={"Authorization": "Bearer wrong"}).status_code
            == 401
        )
        assert (
            authed.post("/search", json=payload, headers={"Authorization": "Bearer secret-key"}).status_code
            == 200
        )
        assert (
            authed.post("/search", json=payload, headers={"Authorization": "Token secret-key"}).status_code
            == 200
        )
        assert authed.post("/search", json=payload, headers={"X-API-Key": "secret-key"}).status_code == 200
        # Health stays open for orchestrator liveness probes.
        assert authed.get("/health").status_code == 200


def test_llm_probe_is_local_by_default_and_sends_nothing(client):
    report = client.post("/admin/llm-probe", json={}).json()
    assert report["live"] is False
    assert report["status"] in ("not_configured", "configured")