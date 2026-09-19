"""Two anomalies found by the fragmentation diagnostic.

1. A chunk that literally contains the query term "latency" scored 0.000.
2. Under 200 same-repo distractors the relevant trajectory fell to rank 5,
   though it ranks 1st with only 3 distractors.

Run:  PYTHONPATH=src python scripts/diagnose_anomalies.py
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
from codemem.index.sparse import build_sparse, fts_query_terms  # noqa: E402
from codemem.index.store import Store  # noqa: E402
from codemem.search.query import plan_query  # noqa: E402
from codemem.search.service import SearchPipeline  # noqa: E402

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
    settings = Settings(data_dir=pathlib.Path(tempfile.mkdtemp()))
    store = Store(settings)
    try:
        add = AddPipeline(settings, store)

        print("=== anomaly 1: does a matching chunk score zero? ===")
        add.handle(request_id="rel", user_id="u1", session_id="rel", messages=[M(RELEVANT)])
        for i, text in enumerate(DISTRACTORS[:3]):
            add.handle(
                request_id=f"d{i}", user_id="u1", session_id=f"d{i}", messages=[M(text.format(i=i))]
            )
        with store._read() as conn:  # noqa: SLF001
            for row in conn.execute(
                "SELECT id, text, sparse FROM memory WHERE user_id='u1' ORDER BY id"
            ).fetchall():
                text = row["text"]
                has_latency_text = "latency" in text.lower()
                has_latency_sparse = "latency" in (row["sparse"] or "").split()
                print(
                    f"  mem {row['id']}: 'latency' in text={has_latency_text} "
                    f"in sparse={has_latency_sparse}"
                )
        print(f"  query terms: {fts_query_terms(QUERY)}")
        print("  bm25 scores:")
        for memory_id, score in store.bm25_search("u1", QUERY, 20):
            print(f"    mem {memory_id}: {score:.6f}")

        print("\n=== anomaly 2: rank of the relevant memory vs corpus size ===")
        for n_distractors in (0, 3, 20, 100, 200):
            settings2 = Settings(data_dir=pathlib.Path(tempfile.mkdtemp()))
            store2 = Store(settings2)
            try:
                add2 = AddPipeline(settings2, store2)
                add2.handle(
                    request_id="rel",
                    user_id="u1",
                    session_id="rel",
                    messages=[M(RELEVANT, ts=1704067200000)],
                )
                for i in range(n_distractors):
                    add2.handle(
                        request_id=f"d{i}",
                        user_id="u1",
                        session_id=f"d{i}",
                        messages=[M(DISTRACTORS[i % len(DISTRACTORS)].format(i=i),
                                   ts=1704067200000 + (i + 1) * 1000)],
                    )

                search = SearchPipeline(settings2, store2)
                items = search.handle(user_id="u1", query=QUERY, options=None, top_k=100)

                # The relevant memory is the one whose text has "checkout".
                rank = "-"
                for position, item in enumerate(items, start=1):
                    if "checkout" in item.content:
                        rank = str(position)
                        break
                verdict = "OK" if rank == "1" else ("MISS" if rank == "-" else f"rank {rank}")
                print(f"  {n_distractors:>4} distractors -> returned {len(items):>3}, "
                      f"relevant {verdict}")
            finally:
                store2.close()
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
