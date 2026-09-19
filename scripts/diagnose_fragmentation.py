"""Why does the relevant trajectory lose to single-chunk distractors?

Hypothesis: a trajectory is split into several chunks and no single chunk
contains all the query terms, while a distractor is one dense chunk that
repeats a few query terms. Per-chunk BM25 then favours the distractor.

This prints, for one on-topic query, the per-chunk lexical scores of the
relevant trajectory against the best distractors.

Run:  PYTHONPATH=src python scripts/diagnose_fragmentation.py
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
    "Session 7: an N+1 query was reported in src/reports/export.py; added an index.",
    "Session 9: latency of the export job was reduced from 2s to 1.5s by batching.",
    "Session 11: the query in src/admin/search.py returned stale rows; fixed the cache.",
]


class M:
    def __init__(self, content: str) -> None:
        self.role = "user"
        self.content = content
        self.timestamp = None


def main() -> int:
    settings = Settings(data_dir=pathlib.Path(tempfile.mkdtemp()))
    store = Store(settings)
    try:
        add = AddPipeline(settings, store)
        add.handle(request_id="rel", user_id="u1", session_id="rel", messages=[M(RELEVANT)])
        for i, text in enumerate(DISTRACTORS):
            add.handle(
                request_id=f"d{i}", user_id="u1", session_id=f"d{i}", messages=[M(text)]
            )

        query = "Why was checkout latency slow and how was it fixed?"
        plan = plan_query(query)
        print(f"query: {query!r}")
        print(f"entities extracted: {plan.entities}")
        print(f"probes: {plan.probes}")
        print()

        # Show how the relevant trajectory was chunked.
        print("relevant trajectory chunks:")
        with store._read() as conn:  # noqa: SLF001 - diagnostic
            rows = conn.execute(
                "SELECT c.id, c.kind, c.text, m.id AS mid FROM chunk c"
                " JOIN memory m ON m.chunk_id = c.id"
                " WHERE c.user_id='u1' AND c.session_id='rel'"
            ).fetchall()
        for row in rows:
            print(f"  chunk {row['id']} (mem {row['mid']}, {row['kind']}): "
                  f"{row['text'][:70]!r}")

        print("\nper-chunk lexical scores (higher = better):")
        scored = store.bm25_search("u1", query, 50)
        for memory_id, score in scored[:10]:
            mem = store.fetch_memories("u1", [memory_id])[memory_id]
            marker = "  <-- RELEVANT" if mem.session_id == "rel" else ""
            print(
                f"  mem {memory_id} session={mem.session_id:6} score={score:7.3f}"
                f"  {mem.text[:52]!r}{marker}"
            )

        # What if we score the whole trajectory as one document?
        print("\nif the whole trajectory were one document, its terms:")
        whole = RELEVANT.lower()
        for term in ("checkout", "latency", "fixed", "slow"):
            print(f"  {term!r} appears {whole.count(term)}x")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
