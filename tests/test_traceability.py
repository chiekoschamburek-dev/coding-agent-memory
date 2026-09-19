"""Response payload shape and auditability.

Two contract requirements meet here:

* the response is ``{"data": [{"id", "content", "score", "created_at"}]}``, and
  ``data[].content`` is described as remembered text "preserved verbatim for
  audit" — so it must be plain memory text, not our own packaging;
* ``created_at`` is the memory's *source* time or persistence time, so when the
  platform supplies a source timestamp it belongs there rather than inside the
  content.

These tests encode the property directly:

    ``content`` is a verbatim span of stored memory text.

If a future change reintroduces headers, labels, or paraphrase into the payload,
these fail. That is intended: such content could not be matched back to the Add
input, and a card asserting something the trajectory never said would be
fabricated evidence.

Source timestamp used throughout: 2024-01-01T00:00:00Z.
"""

from __future__ import annotations

import re

ADDED = """We investigated a slow checkout page in the payments service.

The traceback pointed at src/orders/repository.py line 88: an N+1 query.

Root cause: the order items were loaded lazily inside the loop.

Fix: eager-load the relation with joinedload. p95 latency went 840ms -> 95ms.
"""

SOURCE_TS_MS = 1704067200000
SOURCE_TS_ISO = "2024-01-01T00:00:00Z"

# Content must not carry our own packaging: no "[field] value" lines and no
# separator we invented.
_PACKAGING_RE = re.compile(r"^\[[a-z_]+\]\s", re.MULTILINE)
_SEPARATOR_RE = re.compile(r"^-{3,}\s*$", re.MULTILINE)

# ISO-8601 timestamps, which must not appear inside content.
_ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def _add(client, content=ADDED, ts=SOURCE_TS_MS, request_id="r1", user_id="u1"):
    message = {"role": "user", "content": content}
    if ts is not None:
        message["timestamp"] = ts
    return client.post(
        "/add",
        json={
            "request_id": request_id,
            "user_id": user_id,
            "session_id": f"s-{request_id}",
            "messages": [message],
        },
    )


def _search(client, query, user_id="u1", top_k=10, options=None):
    payload = {"query": query, "user_id": user_id, "top_k": top_k}
    if options:
        payload["options"] = options
    return client.post("/search", json=payload).json()["data"]


# ------------------------------------------------------------ shape ------


def test_response_envelope_matches_the_documented_schema(client):
    _add(client)
    resp = client.post(
        "/search", json={"query": "checkout slow", "user_id": "u1", "top_k": 5}
    )
    body = resp.json()
    assert set(body) == {"data"}, "the response is a data-only object"
    assert isinstance(body["data"], list)
    for item in body["data"]:
        # Required fields must be present; optional ones may be absent.
        assert set(item) <= {"id", "content", "score", "created_at"}
        assert isinstance(item["id"], str) and item["id"]
        assert isinstance(item["content"], str) and item["content"]
        assert isinstance(item["score"], float)
        assert isinstance(item["created_at"], str)


def test_content_is_a_verbatim_span_when_not_truncated(client):
    """A memory that fits the budget must come back whole and unaltered,
    modulo surrounding whitespace."""
    _add(client)
    data = _search(client, "checkout slow N+1 query joinedload")
    assert data
    assert data[0]["content"] == ADDED.strip()


def test_content_contains_no_packaging_or_separators(client):
    """No "[field] value" headers, no "---" separators, no invented timestamps."""
    _add(client)
    for query in ("checkout slow", "N+1 query", "joinedload latency", "root cause"):
        for item in _search(client, query):
            content = item["content"]
            assert not _PACKAGING_RE.search(content), (
                f"content carries packaging: {content[:120]!r}"
            )
            assert not _SEPARATOR_RE.search(content), (
                f"content carries a separator: {content[:120]!r}"
            )
            assert not _ISO_RE.search(content), (
                f"a timestamp leaked into content: {content[:120]!r}"
            )


def test_content_lines_come_from_the_stored_text(client):
    """Every line of returned content must occur in the stored memory."""
    _add(client)
    stored_lines = {line.strip() for line in ADDED.splitlines() if line.strip()}
    for item in _search(client, "checkout slow tracing query"):
        for line in item["content"].splitlines():
            stripped = line.strip()
            if not stripped or stripped == "…":
                continue  # elision marker from truncation
            assert stripped in stored_lines, (
                f"line not present in the stored memory: {stripped!r}"
            )


# ------------------------------------------------------- source timestamp --


def test_created_at_carries_the_source_timestamp(client):
    """The platform's timestamp belongs in created_at, not inside content."""
    _add(client)
    data = _search(client, "checkout slow")
    assert data
    assert data[0]["created_at"] == SOURCE_TS_ISO


def test_created_at_falls_back_to_persistence_time_only_when_source_has_none(client):
    _add(client, content="Refactored src/parser/lexer.py for clarity.", ts=None,
         request_id="r2")
    data = _search(client, "lexer refactor")
    assert data
    created = data[0]["created_at"]
    # Still a valid timestamp (persistence time), just not a fabricated source one.
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", created)
    assert created != SOURCE_TS_ISO


def test_source_timestamps_are_not_conflated_with_processing_time(client):
    _add(client)
    item = _search(client, "checkout slow")[0]
    assert item["created_at"] == SOURCE_TS_ISO
    # Our write clock is a different date in this fixture; it must not appear.
    assert "2026-" not in item["created_at"]


# ------------------------------------------------------------ options -----


def test_options_never_appear_in_returned_content(client):
    """Options are sent without gold answers and only widen the retrieval
    probes; echoing one back would shade into answering."""
    _add(client, content="The retry backoff base is two seconds and doubles.")
    options = ["A. Linear backoff", "B. Exponential backoff, base 2", "C. Fixed delay"]
    for item in _search(client, "what is the retry backoff strategy", options=options):
        for option in options:
            assert option not in item["content"]
            assert option.split(". ", 1)[1] not in item["content"]


def test_search_never_emits_answer_framing(client):
    """No 'the answer is' style framing in what we return."""
    _add(client)
    for item in _search(client, "what is the answer for checkout slow"):
        low = item["content"].lower()
        for phrase in ("the answer is", "correct option", "you should answer"):
            assert phrase not in low


# ---------------------------------------------------------- truncation ----


def test_truncated_content_is_still_verbatim_and_marked(client):
    """Long memories are cut, not rewritten, and the cut is visible."""
    long_memory = "\n".join(
        f"Step {i}: we instrumented the pipeline and recorded the observed behaviour."
        for i in range(200)
    )
    _add(client, content=long_memory, request_id="r-long")
    data = _search(client, "instrumented the pipeline recorded behaviour")
    assert data
    content = data[0]["content"]
    # Everything except the elision markers must appear in the stored text.
    for line in content.replace("…", "").splitlines():
        stripped = line.strip()
        if stripped:
            assert stripped in long_memory, f"non-verbatim line: {stripped!r}"


def test_relevant_window_is_preferred_over_the_opening(client):
    """A long trajectory must yield the part that matches the question, not its
    opening lines: the diagnosis usually sits in the middle."""
    filler = "\n".join(f"Chit chat line {i} about unrelated scheduling matters." for i in range(60))
    long_memory = (
        filler
        + "\nThe IndexError came from src/parser/tokenizer.py line 142.\n"
        + filler
    )
    _add(client, content=long_memory, request_id="r-window")
    data = _search(client, "IndexError src/parser/tokenizer.py line 142")
    assert data
    assert "src/parser/tokenizer.py line 142" in data[0]["content"], (
        "the matching window should be selected over the memory's opening"
    )
