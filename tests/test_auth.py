"""Authentication: Token, Bearer, and X-Api-Key, plus the unauthenticated
health check the platform requires.
"""

from __future__ import annotations

ADD_BODY = {
    "request_id": "r1",
    "user_id": "u1",
    "session_id": "s1",
    "messages": [{"role": "user", "content": "The build cache lives in .cache/build."}],
}

SEARCH_BODY = {"query": "where is the build cache", "user_id": "u1", "top_k": 10}


def test_add_requires_credentials_when_key_configured(auth_client):
    resp = auth_client.post("/add", json=ADD_BODY)
    assert resp.status_code == 401
    assert "reason" in resp.json()["detail"]
    assert resp.headers.get("www-authenticate") == "Bearer"


def test_search_requires_credentials_when_key_configured(auth_client):
    assert auth_client.post("/search", json=SEARCH_BODY).status_code == 401


def test_bearer_token_accepted(auth_client):
    resp = auth_client.post(
        "/add", json=ADD_BODY, headers={"Authorization": "Bearer sekret-key"}
    )
    assert resp.status_code == 200 and resp.json()["success"] is True


def test_raw_token_scheme_accepted(auth_client):
    resp = auth_client.post(
        "/add", json=ADD_BODY, headers={"Authorization": "Token sekret-key"}
    )
    assert resp.status_code == 200


def test_x_api_key_accepted(auth_client):
    resp = auth_client.post("/add", json=ADD_BODY, headers={"X-Api-Key": "sekret-key"})
    assert resp.status_code == 200


def test_wrong_key_rejected(auth_client):
    resp = auth_client.post(
        "/add", json=ADD_BODY, headers={"Authorization": "Bearer wrong"}
    )
    assert resp.status_code == 401


def test_search_works_with_valid_credentials(auth_client):
    auth_client.post("/add", json=ADD_BODY, headers={"X-Api-Key": "sekret-key"})
    resp = auth_client.post(
        "/search", json=SEARCH_BODY, headers={"X-Api-Key": "sekret-key"}
    )
    assert resp.status_code == 200
    assert resp.json()["data"]


def test_no_auth_configured_allows_anonymous_access(client):
    """Unauthenticated operation is what the platform permits for public smoke."""
    assert client.post("/add", json=ADD_BODY).status_code == 200
    assert client.post("/search", json=SEARCH_BODY).status_code == 200
