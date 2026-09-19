"""Attribute the rank degradation to a specific channel.

Hypothesis: the recency channel injects every recent memory into the fused
ranking with weight 0.35, so with hundreds of same-repo distractors (all of
which carry timestamps) their recency contributions accumulate and push the
genuinely relevant memory down.

This measures the relevant memory's rank with each channel ablated, rather
than guessing which one is responsible.

Run:  PYTHONPATH=src python scripts/diagnose_channel_attribution.py
"""

from __future__ import annotations

import logging
import pathlib
import sys
import tempfile

sys.path.insert(0, "src")

logging.disable(logging.INFO)

from codemem.add.pipeline import AddPipeline  # noqa: E402
from codemem.core.config import Settings  # noqa: E402
from codemem.index.store import Store  # noqa: E402

from codemem.search.query import plan_query  # noqa: E402

QUERY = "Why was checkout latency slow and how was it fixed?"

RELEVANT = """Session 42: investigating slow checkout latency.

We profiled the handler and found an N+1 query in src/orders/repository.py.

```python
for order in orders:
    items = session.query(Item).filter(Item.order_id == order.id).all()
```

Fix was to eager-load the relationship with joinedload, cutting p95 latency
from 840ms to 95ms. The regression test lives in tests/test_orders_latency.py.
"""

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


class M:
    def __init__(self, content: str, ts: int | None = None) -> None:
        self.role = "user"
        self.content = content
        self.timestamp = ts


def build(n_distractors: int) -> Store:
    settings = Settings(data_dir=pathlib.Path(tempfile.mkdtemp()))
    store = Store(settings)
    add = AddPipeline(settings, store)
    add.handle(
        request_id="rel", user_id="u1", session_id="rel", messages=[M(RELEVANT, ts=1704000000000)]
    )
    # Distractors are newer than the relevant memory, which is the realistic
    # case: the relevant prior work happened earlier.
    for i in range(n_distractors):
        add.handle(
            request_id=f"d{i}",
            user_id="u1",
            session_id=f"d{i}",
            messages=[M(DISTRACTORS[i % len(DISTRACTORS)].format(i=i),
                        ts=1704067200000 + (i + 1) * 1000)],
        )
    return store


def rank_of_relevant(store: Store, enabled: list[str]) -> tuple[str, int]:
    """Rank the relevant memory while only ``enabled`` channels contribute.

    Channels are disabled by setting the *ranker* into single/dual-channel mode
    rather than by removing a weight, because ``CHANNEL_WEIGHTS.get(name, 0.5)``
    would silently restore a default for any channel left out of the dict.
    """
    from codemem.search.retriever import Retriever

    original_recall = Retriever.recall

    def patched(self, user_id: str, plan):
        return _recall_with(self, user_id, plan, enabled)

    Retriever.recall = patched
    try:
        from codemem.search.service import SearchPipeline

        settings = Settings(data_dir=store.settings.data_dir)
        items = SearchPipeline(settings, store).handle(
            user_id="u1", query=QUERY, options=None, top_k=100
        )
        for position, item in enumerate(items, start=1):
            if "checkout" in item.content:
                return str(position), len(items)
        return "-", len(items)
    finally:
        Retriever.recall = original_recall


def _recall_with(retriever, user_id: str, plan, enabled: list[str]):
    """Reimplementation of Retriever.recall that runs only the named channels."""
    from codemem.search.retriever import CHANNEL_WEIGHTS, Candidate

    per_channel = retriever.settings.recall_per_channel
    k = retriever.settings.rrf_k
    candidates: dict[int, Candidate] = {}

    def merge(channel: str, ranked):
        weight = CHANNEL_WEIGHTS.get(channel, 0.5)
        for rank, (memory_id, score) in enumerate(ranked, start=1):
            if memory_id <= 0:
                continue
            cand = candidates.get(memory_id)
            if cand is None:
                cand = Candidate(memory_id=memory_id)
                candidates[memory_id] = cand
            if channel in cand.channels:
                continue
            cand.add(channel, rank, score, weight, k)

    if "lexical" in enabled:
        lexical: dict[int, float] = {}
        for probe in plan.probes:
            for memory_id, score in retriever.store.bm25_search(user_id, probe, per_channel):
                if score > lexical.get(memory_id, float("-inf")):
                    lexical[memory_id] = score
        merge("lexical", sorted(lexical.items(), key=lambda kv: -kv[1])[:per_channel])
    if "entity" in enabled:
        merge("entity", retriever.store.entity_search(user_id, plan.entities, per_channel))
    if "recency" in enabled:
        newest = retriever.store.max_ts(user_id)
        if newest > 0:
            merge("recency", retriever._recency_ranking(user_id, newest, per_channel))

    return sorted(candidates.values(), key=lambda c: (-c.rrf, c.memory_id))[
        : retriever.settings.candidate_pool
    ]


def main() -> int:
    print(f"query: {QUERY!r}")
    print(f"plan entities: {plan_query(QUERY).entities}")
    print()

    configs = {
        "all channels (current)": ["lexical", "entity", "recency"],
        "no recency": ["lexical", "entity"],
        "lexical only": ["lexical"],
        "entity only": ["entity"],
        "recency only": ["recency"],
    }

    for n in (20, 200):
        print(f"--- corpus: 1 relevant + {n} same-repo distractors ---")
        store = build(n)
        try:
            for label, enabled in configs.items():
                rank, returned = rank_of_relevant(store, enabled)
                verdict = "OK" if rank == "1" else rank
                print(f"  {label:26} relevant rank={verdict:>6}  (returned {returned})")
        finally:
            store.close()
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
