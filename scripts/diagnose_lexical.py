"""Inspect which documents beat the relevant one on the lexical channel.

The channel-ablation diagnostic showed recency helps rather than hurts, so the
degradation originates in lexical scoring. This prints the top candidates with
the query terms they actually match, so the cause is observable rather than
inferred.

Run:  PYTHONPATH=src python scripts/diagnose_lexical.py
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
from codemem.index.sparse import fts_query_terms  # noqa: E402
from codemem.index.store import Store  # noqa: E402

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


def main() -> int:
    n = 200
    settings = Settings(data_dir=pathlib.Path(tempfile.mkdtemp()))
    store = Store(settings)
    try:
        add = AddPipeline(settings, store)
        add.handle(
            request_id="rel", user_id="u1", session_id="rel", messages=[M(RELEVANT)]
        )
        for i in range(n):
            add.handle(
                request_id=f"d{i}",
                user_id="u1",
                session_id=f"d{i}",
                messages=[M(DISTRACTORS[i % len(DISTRACTORS)].format(i=i))],
            )

        terms = fts_query_terms(QUERY)
        print(f"query: {QUERY!r}")
        print(f"terms: {terms}")
        print(f"corpus: {store.counts()[1]} memories, "
              f"{store.counts()[1]} chunks")
        print()

        with store._read() as conn:  # noqa: SLF001
            print("document frequency of each query term (how many memories contain it):")
            for term in terms:
                row = conn.execute(
                    "SELECT count(*) c FROM memory_fts WHERE memory_fts MATCH ?",
                    (f'"{term}"',),
                ).fetchone()
                print(f"  {term:12} appears in {row['c']:>4} memories")

        print("\ntop lexical candidates:")
        scored = store.bm25_search("u1", QUERY, 20)
        for memory_id, score in scored[:20]:
            mem = store.fetch_memories("u1", [memory_id])[memory_id]
            text_lower = mem.text.lower()
            matched = [t for t in terms if t in text_lower]
            marker = "  <-- RELEVANT" if mem.session_id == "rel" else ""
            print(
                f"  score={score:7.3f} matched={matched}  {mem.text[:46]!r}{marker}"
            )

        # Where exactly does the relevant memory sit?
        all_scored = store.bm25_search("u1", QUERY, 400)
        for position, (memory_id, score) in enumerate(all_scored, start=1):
            mem = store.fetch_memories("u1", [memory_id])[memory_id]
            if mem.session_id == "rel":
                print(f"\nrelevant memory lexical rank: {position} (score {score:.3f})")
                break
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
