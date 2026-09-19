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
    for position, item in enumerate(items, start=1):
        if "checkout" in item.content:
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
