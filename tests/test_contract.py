"""Add/Search contract conformance — the shape the platform validates.

These assert the properties the platform is documented to check: echoed
identifiers, ``success`` after durability, always-present ``data``, non-empty
item fields, the ``top_k`` ceiling, and the error envelope.
"""

from __future__ import annotations


def _add(client, request_id="r1", user_id="u1", session_id="s1", content="hello"):
    return client.post(
        "/add",
        json={
            "request_id": request_id,
            "user_id": user_id,
            "session_id": session_id,
            "messages": [{"role": "user", "content": content}],
        },
    )


def test_health_is_unauthenticated_and_2xx(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_health_needs_no_key_even_when_auth_enabled(auth_client):
    assert auth_client.get("/health").status_code == 200


def test_add_echoes_identifiers_verbatim(client):
    resp = _add(client, "req-abc", "user-x", "sess-y", "some content")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "success": True,
        "request_id": "req-abc",
        "user_id": "user-x",
        "session_id": "sess-y",
    }


def test_add_response_has_no_extra_fields(client):
    body = _add(client).json()
    assert set(body) == {"success", "request_id", "user_id", "session_id"}


def test_added_memory_is_immediately_searchable(client):
    """The contract requires durability before HTTP 200, so a search issued
    right after a successful Add must find it."""
    _add(client, content="We set the Redis connection pool size to 32 in src/cache/redis_client.py.")
    resp = client.post(
        "/search",
        json={"query": "What is the Redis connection pool size?", "user_id": "u1", "top_k": 100},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data, "memory added before the 200 must already be retrievable"
    assert any("32" in item["content"] for item in data)


def test_search_always_returns_data_key(client):
    resp = client.post(
        "/search", json={"query": "anything at all", "user_id": "nobody", "top_k": 100}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body
    assert body["data"] == []


def test_search_items_have_required_fields(client):
    _add(client, content="The parser retries three times on a transient socket error.")
    data = client.post(
        "/search", json={"query": "parser retry socket error", "user_id": "u1", "top_k": 100}
    ).json()["data"]
    assert data
    for item in data:
        assert set(item) >= {"id", "content"}
        assert isinstance(item["id"], str) and item["id"]
        assert isinstance(item["content"], str) and item["content"]
        if item.get("score") is not None:
            assert isinstance(item["score"], (int, float))


def test_scores_are_monotonically_non_increasing(client):
    _add(
        client,
        content=(
            "We fixed the tokenizer IndexError in src/parser/tokenizer.py by guarding "
            "the empty buffer before pop. Ran pytest tests/test_tokenizer.py afterwards."
        ),
    )
    data = client.post(
        "/search",
        json={
            "query": "Why did src/parser/tokenizer.py raise IndexError?",
            "user_id": "u1",
            "top_k": 100,
        },
    ).json()["data"]
    scores = [i["score"] for i in data if i.get("score") is not None]
    assert scores == sorted(scores, reverse=True)


def test_search_never_exceeds_top_k(client):
    for n in range(12):
        _add(
            client,
            request_id=f"r{n}",
            session_id=f"s{n}",
            content=f"Session {n}: the queue worker in src/queue/worker.py retries failed jobs.",
        )
    for top_k in (1, 2, 5, 10):
        data = client.post(
            "/search",
            json={"query": "queue worker retries failed jobs", "user_id": "u1", "top_k": top_k},
        ).json()["data"]
        assert len(data) <= top_k, f"returned {len(data)} for top_k={top_k}"


def test_options_are_accepted_and_do_not_leak_into_content(client):
    _add(client, content="The retry backoff is exponential with a base of two seconds.")
    resp = client.post(
        "/search",
        json={
            "query": "What is the retry backoff strategy?",
            "options": ["A. Linear backoff", "B. Exponential backoff with base 2"],
            "user_id": "u1",
            "top_k": 100,
        },
    )
    assert resp.status_code == 200
    for item in resp.json()["data"]:
        # The platform sends options without gold answers; we must never echo an
        # option label back as if it were memory content.
        assert "A. Linear backoff" not in item["content"]


def test_add_idempotent_for_same_request_id(client):
    """Retries reuse the request_id and must not duplicate memory."""
    first = _add(client, "req-dup", content="The cache TTL is 300 seconds.")
    second = _add(client, "req-dup", content="The cache TTL is 300 seconds.")
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()

    data = client.post(
        "/search", json={"query": "What is the cache TTL?", "user_id": "u1", "top_k": 100}
    ).json()["data"]
    assert len(data) == 1, f"retry duplicated memory: {len(data)} items"


def test_add_retry_with_different_body_still_returns_success(client):
    """A retry may carry an identical payload; whatever arrives, the echoed
    identifiers and success flag must match the request."""
    _add(client, "req-x", content="first body")
    body = _add(client, "req-x", content="second body").json()
    assert body["request_id"] == "req-x" and body["success"] is True


def test_validation_error_uses_detail_envelope(client):
    resp = client.post("/add", json={"request_id": "r"})
    assert resp.status_code == 422
    body = resp.json()
    assert "detail" in body
    assert "reason" in body["detail"]
    assert isinstance(body["detail"].get("errors"), list)


def test_missing_query_is_rejected(client):
    resp = client.post("/search", json={"user_id": "u1", "top_k": 5})
    assert resp.status_code == 422
    assert "reason" in resp.json()["detail"]


def test_top_k_zero_is_rejected(client):
    resp = client.post("/search", json={"query": "q", "user_id": "u1", "top_k": 0})
    assert resp.status_code == 422


def test_empty_topics_return_no_items(client):
    """A question unrelated to anything stored must not be answered with
    irrelevant memory padded to fill the budget."""
    _add(client, content="We bumped cryptography to 42.0 in requirements.txt.")
    data = client.post(
        "/search",
        json={"query": "How do I proof sourdough overnight?", "user_id": "u1", "top_k": 100},
    ).json()["data"]
    assert data == []


def test_content_accepts_ordered_parts_defensively(client):
    """The Coding track sends strings; a ContentPart-style body should degrade
    rather than fail contract validation."""
    resp = client.post(
        "/add",
        json={
            "request_id": "r-parts",
            "user_id": "u1",
            "session_id": "s1",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Fallback path works."}]}
            ],
        },
    )
    assert resp.status_code == 200
