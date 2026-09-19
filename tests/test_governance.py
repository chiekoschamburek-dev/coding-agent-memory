"""Memory governance and the design invariants.

The invariants under test:

1. Search never generates content — everything returned was produced during Add
   and is traceable to stored raw text.
2. Add never fails because enrichment failed.
3. Same-repository noise is suppressed rather than padded into the prefix.
"""

from __future__ import annotations

from codemem.add.pipeline import AddPipeline
from codemem.core.config import Settings
from codemem.index.store import Store
from codemem.search.service import SearchPipeline


def _store(settings: Settings) -> Store:
    return Store(settings)


def test_search_content_is_traceable_to_stored_text(client):
    """Every returned payload must be built from stored text, never invented."""
    original = (
        "We fixed the flaky CI job by pinning the container image tag in "
        ".github/workflows/ci.yml to ubuntu-22.04."
    )
    client.post(
        "/add",
        json={
            "request_id": "r1",
            "user_id": "u1",
            "session_id": "s1",
            "messages": [{"role": "user", "content": original}],
        },
    )
    data = client.post(
        "/search",
        json={
            "query": "How was the flaky CI job fixed?",
            "user_id": "u1",
            "top_k": 100,
        },
    ).json()["data"]
    assert data
    body = data[0]["content"]
    # The stored sentence appears verbatim; the header only adds identifiers
    # that were deterministically extracted from that same text.
    assert "pinning the container image tag" in body
    assert "ubuntu-22.04" in body
    assert ".github/workflows/ci.yml" in body


def test_search_does_not_answer_the_question(client):
    """Search must return evidence, not an answer dressed as memory."""
    client.post(
        "/add",
        json={
            "request_id": "r1",
            "user_id": "u1",
            "session_id": "s1",
            "messages": [
                {
                    "role": "user",
                    "content": "The database connection pool maximum is 64 connections.",
                }
            ],
        },
    )
    data = client.post(
        "/search",
        json={
            "query": "What is the maximum database connection pool size?",
            "options": ["A. 16", "B. 32", "C. 64", "D. 128"],
            "user_id": "u1",
            "top_k": 100,
        },
    ).json()["data"]
    assert data
    for item in data:
        # No option labels, no "the answer is" framing.
        assert "C." not in item["content"]
        assert "the answer is" not in item["content"].lower()


def test_add_succeeds_even_when_enrichment_is_unavailable(settings):
    """Add must degrade, not fail: raw memory persists regardless."""
    settings.llm_enabled = True
    settings.llm_base_url = "http://127.0.0.1:9/does-not-exist"
    store = _store(settings)
    try:
        pipeline = AddPipeline(settings, store)

        class M:
            role = "user"
            content = "A durable fact about src/module.py that must survive."
            timestamp = None

        outcome, degraded = pipeline.handle(
            request_id="r1", user_id="u1", session_id="s1", messages=[M()]
        )
        assert outcome.memories_written > 0, "raw memory must be persisted"
        # Enrichment could not run, but the write still happened.
        assert store.counts()[1] > 0
        assert isinstance(degraded, bool)
    finally:
        store.close()


def test_duplicate_request_writes_nothing(settings):
    store = _store(settings)
    try:
        pipeline = AddPipeline(settings, store)

        class M:
            role = "user"
            content = "Idempotent content with src/x/y.py mentioned once."
            timestamp = None

        first, _ = pipeline.handle(
            request_id="same", user_id="u1", session_id="s1", messages=[M()]
        )
        second, _ = pipeline.handle(
            request_id="same", user_id="u1", session_id="s1", messages=[M()]
        )
        assert first.duplicate is False and first.memories_written > 0
        assert second.duplicate is True and second.memories_written == 0
    finally:
        store.close()


def test_identical_text_in_different_requests_is_deduplicated(settings):
    """The same text added under two request_ids must not become two memories,
    or retries and re-sends would silently duplicate the retrieval pool."""
    store = _store(settings)
    try:
        pipeline = AddPipeline(settings, store)

        class M:
            role = "user"
            content = "The feature flag lives in src/flags/registry.py."
            timestamp = None

        pipeline.handle(request_id="a", user_id="u1", session_id="s1", messages=[M()])
        pipeline.handle(request_id="b", user_id="u1", session_id="s1", messages=[M()])
        assert store.counts()[1] == 1
    finally:
        store.close()


def test_noise_does_not_displace_relevant_memory(settings):
    """Same-repository distractors share vocabulary and paths, so the retriever
    must still rank the genuinely relevant evidence first."""
    store = _store(settings)
    try:
        add = AddPipeline(settings, store)

        class M:
            def __init__(self, content):
                self.role = "user"
                self.content = content
                self.timestamp = None

        relevant = (
            "Fixed the tokenizer crash. The traceback showed IndexError from "
            "Tokenizer.read_token in src/parser/tokenizer.py because the buffer "
            "was already drained when a BOM was present. Guarded the empty buffer."
        )
        distractors = [
            "Refactored src/parser/tokenizer.py to use dataclasses; no behaviour change.",
            "Updated the formatter config so src/parser/tokenizer.py matches the new style.",
            "Added type annotations to src/parser/tokenizer.py for the public API only.",
        ]
        add.handle(request_id="rel", user_id="u1", session_id="s1", messages=[M(relevant)])
        for i, text in enumerate(distractors):
            add.handle(
                request_id=f"n{i}", user_id="u1", session_id=f"n{i}", messages=[M(text)]
            )

        search = SearchPipeline(settings, store)
        items = search.handle(
            user_id="u1",
            query="Why did src/parser/tokenizer.py raise IndexError for BOM input?",
            options=None,
            top_k=100,
        )
        assert items, "relevant memory must be retrieved"
        assert "IndexError" in items[0].content, (
            "the stack-trace memory must outrank stylistic edits to the same file"
        )
    finally:
        store.close()


def test_empty_corpus_returns_nothing(settings):
    store = _store(settings)
    try:
        search = SearchPipeline(settings, store)
        assert search.handle(user_id="nobody", query="anything", options=None, top_k=100) == []
    finally:
        store.close()


def test_supersede_marks_old_memory_and_scores_it_lower(settings):
    """Soft forgetting is scaffolded: the link and the penalty exist, and a
    superseded memory stays retrievable rather than being deleted."""
    store = _store(settings)
    try:
        add = AddPipeline(settings, store)

        class M:
            def __init__(self, content):
                self.role = "user"
                self.content = content
                self.timestamp = None

        add.handle(
            request_id="old",
            user_id="u1",
            session_id="s1",
            messages=[M("The retry limit for src/net/client.py is 3 attempts.")],
        )
        add.handle(
            request_id="new",
            user_id="u1",
            session_id="s2",
            messages=[M("Correction: the retry limit for src/net/client.py is 5 attempts.")],
        )

        ids = sorted(store.fetch_memories("u1", [1, 2]).keys())
        store.mark_superseded("u1", ids[0], ids[1], "later session corrected it")

        rows = store.fetch_memories("u1", ids)
        assert rows[ids[0]].superseded_by == ids[1]
        assert rows[ids[1]].superseded_by is None

        # The superseded memory is down-weighted but still returned, because it
        # remains a plausible precedent. The newer memory leads; the superseded
        # one is flagged in place rather than deleted.
        search = SearchPipeline(settings, store)
        items = search.handle(
            user_id="u1", query="retry limit src/net/client.py", options=None, top_k=100
        )
        assert len(items) == 2, "retrieval must not drop it, only demote it"
        assert "5 attempts" in items[0].content, "the newer memory should lead"
        assert "3 attempts" in items[1].content
        # The superseded marker travels on the identifier, not in the content:
        # content must stay verbatim memory text.
        assert items[1].superseded is True
        assert items[0].superseded is False
        assert "superseded" not in items[1].content
    finally:
        store.close()


def test_supersede_rejects_cross_user_links(settings):
    store = _store(settings)
    try:
        add = AddPipeline(settings, store)

        class M:
            def __init__(self, content):
                self.role = "user"
                self.content = content
                self.timestamp = None

        add.handle(
            request_id="a",
            user_id="alice",
            session_id="s1",
            messages=[M("Alice's fact about src/a/mod.py.")],
        )
        add.handle(
            request_id="b",
            user_id="bob",
            session_id="s2",
            messages=[M("Bob's fact about src/b/mod.py.")],
        )

        import pytest

        with pytest.raises(ValueError):
            store.mark_superseded("alice", 1, 2, "cross-user attempt")
    finally:
        store.close()
