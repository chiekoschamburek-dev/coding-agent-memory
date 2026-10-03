"""Retrieval quality regressions that only appear at realistic corpus size.

Both bugs locked in here were invisible on small corpora and only surfaced with
hundreds of same-repository distractors — which is exactly the condition the
Coding track supplies. Small-corpus tests would have stayed green while the
system ranked real evidence 5th.
"""

from __future__ import annotations

from codemem.add.pipeline import AddPipeline
from codemem.core.config import Settings
from codemem.index.sparse import fts_query_terms
from codemem.index.store import Store
from codemem.search.service import SearchPipeline

RELEVANT = """Session 42: investigating slow checkout latency.

We profiled the handler and found an N+1 query in src/orders/repository.py.

```python
for order in orders:
    items = session.query(Item).filter(Item.order_id == order.id).all()
```

Fix was to eager-load the relationship with joinedload, cutting p95 latency
from 840ms to 95ms. The regression test lives in tests/test_orders_latency.py.
"""

# Deliberately drawn from the same repository: they share vocabulary ("test",
# "error", "src", "fixed", "latency") so lexical similarity alone cannot
# separate them from the relevant memory.
DISTRACTORS = [
    "Session {i}: fixed a flaky test in tests/test_module_{i}.py by adding a retry.",
    "Session {i}: refactored src/core/util_{i}.py to use dataclasses.",
    "Session {i}: the build failed with a missing dependency; updated requirements.txt.",
    "Session {i}: added logging to src/services/handler_{i}.py for easier debugging.",
    "Session {i}: upgraded a dependency and fixed the resulting deprecation warnings.",
    "Session {i}: the error message in src/validation/rules_{i}.py was unclear, improved it.",
    "Session {i}: an N+1 query was reported in src/reports/export_{i}.py; added an index.",
    "Session {i}: ran the test suite and fixed type errors in src/models/entity_{i}.py.",
    "Session {i}: latency of the export job was reduced from 2s to 1.5s by batching.",
    "Session {i}: the query in src/admin/search_{i}.py returned stale rows; fixed the cache.",
]


class _M:
    def __init__(self, content: str, ts: int | None = None) -> None:
        self.role = "user"
        self.content = content
        self.timestamp = ts


def _corpus(settings: Settings, n_distractors: int) -> Store:
    store = Store(settings)
    add = AddPipeline(settings, store)
    add.handle(
        request_id="rel",
        user_id="u1",
        session_id="rel",
        messages=[_M(RELEVANT, ts=1704000000000)],
    )
    # Distractors are strictly newer, so a recency-heavy ranker would prefer them.
    for i in range(n_distractors):
        add.handle(
            request_id=f"d{i}",
            user_id="u1",
            session_id=f"d{i}",
            messages=[
                _M(DISTRACTORS[i % len(DISTRACTORS)].format(i=i),
                   ts=1704067200000 + (i + 1) * 1000)
            ],
        )
    return store


def _rank_of_relevant(items) -> str:
    """Position of the first item from the relevant trajectory.

    Matched on any of that session's own chunks rather than on one phrase: which
    chunk of a session leads is the assembler's choice (``evidence_position_weight``
    tilts it toward the code that was changed), and what this test exists to catch
    is a *different* session displacing it.
    """
    markers = ("checkout", "joinedload", "session.query(Item)")
    for position, item in enumerate(items, start=1):
        if any(marker in item.content for marker in markers):
            return str(position)
    return "-"


def test_stopwords_are_removed_from_fts_terms():
    """Regression: FTS terms are OR-combined, so a surviving function word
    makes every memory match every question and disables gating entirely."""
    terms = fts_query_terms("What is the capital of Mongolia?")
    assert "the" not in terms
    assert "of" not in terms
    assert "is" not in terms
    # A content word must survive.
    assert any(t in terms for t in ("capital", "mongolia"))


def test_identifier_terms_survive_stopword_filtering():
    terms = fts_query_terms("why does read_token fail in src/a/b.py")
    assert "read_token" in terms or "read" in terms
    assert "src" in terms or "b.py" in terms


def test_unrelated_query_returns_nothing_at_scale(settings):
    """With many same-repo distractors, an unrelated question must still return
    no memory rather than a page of shared-vocabulary noise."""
    store = _corpus(settings, n_distractors=200)
    try:
        search = SearchPipeline(settings, store)
        for query in (
            "How do I bake sourdough bread with a crisp crust?",
            "What is the capital of Mongolia?",
            "How do we handle database queries in general?",
        ):
            items = search.handle(user_id="u1", query=query, options=None, top_k=100)
            assert items == [], f"{query!r} returned {len(items)} distractors"
    finally:
        store.close()


def test_relevant_memory_ranks_first_despite_newer_distractors(settings):
    """Regression: rank fusion discarded BM25 magnitude while a peer-weighted
    recency channel contributed noise, so newer unrelated memories displaced the
    decisive match as the corpus grew. The relevant memory must rank first at
    both small and realistic corpus sizes."""
    for n in (3, 20, 200):
        store = _corpus(settings, n_distractors=n)
        try:
            search = SearchPipeline(settings, store)
            items = search.handle(
                user_id="u1",
                query="Why was checkout latency slow and how was it fixed?",
                options=None,
                top_k=100,
            )
            assert items, f"no memory returned with {n} distractors"
            assert _rank_of_relevant(items) == "1", (
                f"relevant memory ranked {_rank_of_relevant(items)} "
                f"with {n} same-repo distractors"
            )
        finally:
            store.close()


def test_stronger_lexical_match_outranks_weaker_one(settings):
    """The magnitude of a match must matter, not only its rank position."""
    store = Store(settings)
    try:
        add = AddPipeline(settings, store)
        add.handle(
            request_id="strong",
            user_id="u1",
            session_id="strong",
            messages=[_M("The retry backoff for src/net/client.py is exponential. "
                         "Retry backoff doubled each attempt. Retry backoff config "
                         "lives in src/net/client.py.")],
        )
        add.handle(
            request_id="weak",
            user_id="u1",
            session_id="weak",
            messages=[_M("We discussed a retry once, briefly, in passing.")],
        )
        search = SearchPipeline(settings, store)
        items = search.handle(
            user_id="u1", query="retry backoff src/net/client.py", options=None, top_k=10
        )
        assert items
        assert "exponential" in items[0].content
    finally:
        store.close()


def test_operative_chunk_beats_its_own_siblings_for_a_slot(settings):
    """Within one session the edit is outscored by its own reads.

    A trajectory reads a file several times and edits it once. The read records
    repeat the path with surrounding explanation, so on every channel they
    outscore the one line that says what was changed, and the per-session cap
    keeps only the top three — which are reads. The word ``edit`` is in none of
    the queries, so term weighting cannot recover it either. Promotion of the
    operative chunk inside the top-ranked session is what puts it in the payload.

    The intra-session position tilt reaches the same chunk by another route, so it
    is pinned off here: this test is about promotion, and with the tilt on it would
    pass for a reason it does not describe.
    """
    settings.evidence_position_weight = 0.0
    store = Store(settings)
    try:
        add = AddPipeline(settings, store)

        def read(index: int) -> str:
            return (
                '[tool Read] {"file_path": "src/orders/checkout.py", "limit": 40}\n'
                f"Inspecting checkout latency in src/orders/checkout.py, pass {index}: "
                "the handler queries the order, then loops over items and loads each "
                "one, so checkout in src/orders/checkout.py issues one query per order "
                "item here."
            )

        edit = (
            '[tool Edit] {"file_path": "src/orders/checkout.py", '
            '"old_string": "for item in order.items:", '
            '"new_string": "for item in order.items_loaded:"}\n'
        )
        messages = [_M(read(i)) for i in range(6)] + [_M(edit)]
        add.handle(
            request_id="work",
            user_id="u1",
            session_id="work",
            messages=messages,
        )
        search = SearchPipeline(settings, store)
        query = "Why was checkout slow in src/orders/checkout.py and what changed?"

        settings.evidence_operative_promotion = 0
        without = search.handle(user_id="u1", query=query, options=None, top_k=100)
        settings.evidence_operative_promotion = 1
        with_promotion = search.handle(user_id="u1", query=query, options=None, top_k=100)
    finally:
        store.close()

    assert without, "the session must be retrievable at all"
    assert not any(
        "[tool Edit]" in item.content for item in without
    ), "the fixture no longer reproduces the loss this test exists for"
    assert any("[tool Edit]" in item.content for item in with_promotion), (
        "promotion failed to place the operative chunk in the payload"
    )
    scores = [item.score for item in with_promotion]
    assert all(a > b for a, b in zip(scores, scores[1:])), (
        "session-major ordering must still return strictly decreasing scores"
    )


def test_position_tilt_delivers_the_edit_with_promotion_off(settings):
    """The shipped intra-session lever, on its own.

    `evidence_position_weight` defaults to 1.0 and is what currently carries the
    edit into the payload when promotion is off, so this pins the behaviour the
    service actually ships: a read-heavy session still returns the chunk that
    records the change.
    """
    store = Store(settings)
    try:
        add = AddPipeline(settings, store)

        def read(index: int) -> str:
            return (
                '[tool Read] {"file_path": "src/orders/checkout.py", "limit": 40}\n'
                f"Inspecting checkout latency in src/orders/checkout.py, pass {index}: "
                "the handler queries the order, then loops over items and loads each "
                "one, so checkout in src/orders/checkout.py issues one query per order "
                "item here."
            )

        edit = (
            '[tool Edit] {"file_path": "src/orders/checkout.py", '
            '"old_string": "for item in order.items:", '
            '"new_string": "for item in order.items_loaded:"}\n'
        )
        add.handle(
            request_id="work",
            user_id="u1",
            session_id="work",
            messages=[_M(read(i)) for i in range(6)] + [_M(edit)],
        )
        search = SearchPipeline(settings, store)
        settings.evidence_operative_promotion = 0
        assert settings.evidence_position_weight, "the tilt is the shipped default"
        items = search.handle(
            user_id="u1",
            query="Why was checkout slow in src/orders/checkout.py and what changed?",
            options=None,
            top_k=100,
        )
    finally:
        store.close()

    assert any("[tool Edit]" in i.content for i in items), (
        "the intra-session tilt failed to deliver the operative chunk"
    )
    assert all(i.score <= 1.0 for i in items), (
        "scores must stay on the 0..1 scale the noise gate is calibrated against"
    )


def test_operative_rank_term_stays_off_by_default(settings):
    """The ranking-score form of the same signal was measured worse; keep it off.

    On the proxy corpus it moved its own target the wrong way (the chunk carrying
    a session's score named a task file 12.9% of the time at weight 0, 10.8% at
    0.3), because distractor sessions are full of edits to *other* files. See
    eval/README.md. What this test does hold is the scale contract: the term
    rescales the other four rather than adding to them.
    """
    assert Settings().operative_rank_weight == 0.0
    settings.operative_rank_weight = 0.3
    store = Store(settings)
    try:
        add = AddPipeline(settings, store)
        add.handle(
            request_id="work",
            user_id="u1",
            session_id="work",
            messages=[
                _M(
                    '[tool Edit] {"file_path": "src/orders/checkout.py"}\n'
                    "checkout in src/orders/checkout.py loops over order items."
                )
            ],
        )
        items = SearchPipeline(settings, store).handle(
            user_id="u1",
            query="Why was checkout slow in src/orders/checkout.py?",
            options=None,
            top_k=100,
        )
    finally:
        store.close()
    assert all(0.0 < i.score <= 1.0 for i in items), (
        "the term must keep the score on the 0..1 range the noise gate uses"
    )


def test_position_tilt_reorders_within_a_session_only():
    """The intra-session position prior is bounded and cannot flip a clear win.

    It exists because a trajectory reads a file before changing it, so among the
    chunks of one session the later one is likelier to hold the edit. Two
    properties have to hold: a chunk that scores clearly higher stays ahead (the
    tilt is bounded by the weight), and it is applied inside a session group, so
    it never gets to decide which session is returned.
    """
    from codemem.search.evidence import _position_ordered
    from codemem.search.retriever import Candidate

    def group(*pairs: tuple[int, float]) -> list[Candidate]:
        return [Candidate(memory_id=m, final=f) for m, f in pairs]

    span = (0, 100)
    # 0.9 at position 10 beats 0.8 at position 90 once tilted: 0.9*0.76 < 0.8*1.24
    tilted = _position_ordered(group((10, 0.9), (90, 0.8)), span, 0.6)
    assert [c.memory_id for c in tilted] == [90, 10]
    # Weight 0 is inert, so the shipped behaviour is unchanged by default.
    assert [c.memory_id for c in _position_ordered(group((10, 0.9), (90, 0.8)), span, 0.0)] == [10, 90]
    # A 2x score gap survives the maximum tilt the weight allows.
    assert [c.memory_id for c in _position_ordered(group((10, 1.0), (99, 0.5)), span, 0.6)] == [10, 99]
    # A session that is one memory long has no position to speak of.
    single = group((7, 1.0))
    assert _position_ordered(single, (7, 7), 0.6) == single


def test_recency_alone_cannot_qualify_a_memory(settings):
    """Recency is not evidence: it may break ties among matching memories but
    must never pull a non-matching one into the results."""
    store = Store(settings)
    try:
        add = AddPipeline(settings, store)
        add.handle(
            request_id="only",
            user_id="u1",
            session_id="only",
            messages=[_M("We use Redis for caching in src/cache/redis_client.py.")],
        )
        search = SearchPipeline(settings, store)
        assert (
            search.handle(
                user_id="u1", query="sourdough bread crust", options=None, top_k=100
            )
            == []
        )
    finally:
        store.close()
