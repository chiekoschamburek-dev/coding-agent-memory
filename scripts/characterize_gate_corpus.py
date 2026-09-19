"""Characterize the noise gate under a realistic same-repository corpus.

The single-memory diagnostic is not representative: with only one memory, BM25
either matches or has nothing to match, so the gate looks perfect. The real
track supplies ~1,290 same-repository trajectories that share vocabulary
("test", "error", "src", "run"), so the question is whether an unrelated query
still drags back distractors.

Run:  PYTHONPATH=src python scripts/characterize_gate_corpus.py
"""

from __future__ import annotations

import logging
import pathlib
import sys
import tempfile

sys.path.insert(0, "src")

logging.disable(logging.INFO)

from fastapi.testclient import TestClient  # noqa: E402

from codemem.api.app import create_app  # noqa: E402
from codemem.core.config import Settings  # noqa: E402

# The genuinely relevant trajectory.
RELEVANT = """Session 42: investigating slow checkout latency.

We profiled the handler and found an N+1 query in src/orders/repository.py.

```python
for order in orders:
    items = session.query(Item).filter(Item.order_id == order.id).all()
```

Fix was to eager-load the relationship with joinedload, cutting p95 latency
from 840ms to 95ms. The regression test lives in tests/test_orders_latency.py.
"""

# Same-repository distractors: they share the repo's vocabulary and layout, so
# lexical similarity is uniformly high and naive scoring cannot separate them.
DISTRACTOR_TEMPLATES = [
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

PROBES = [
    ("exact topic", "Why was checkout latency slow and how was it fixed?"),
    ("same words", "What caused the slow checkout latency in src/orders/repository.py?"),
    ("paraphrase", "Our order page takes ages to load. What did we do about performance?"),
    ("vague on-topic", "What performance improvements were made?"),
    ("symbol only", "What is joinedload used for?"),
    ("path only", "What happened in src/orders/repository.py?"),
    ("mild overlap", "How do we handle database queries in general?"),
    ("generic repo", "What changes were made to the codebase?"),
    ("unrelated", "How do I bake sourdough bread with a crisp crust?"),
    ("other domain", "What is the capital of Mongolia?"),
]


def build_corpus(client: TestClient, n_distractors: int) -> int:
    client.post(
        "/add",
        json={
            "request_id": "relevant",
            "user_id": "u1",
            "session_id": "rel",
            "messages": [
                {"role": "user", "content": RELEVANT, "timestamp": 1704067200000}
            ],
        },
    )
    for i in range(n_distractors):
        template = DISTRACTOR_TEMPLATES[i % len(DISTRACTOR_TEMPLATES)]
        client.post(
            "/add",
            json={
                "request_id": f"d{i}",
                "user_id": "u1",
                "session_id": f"d{i}",
                "messages": [
                    {
                        "role": "user",
                        "content": template.format(i=i),
                        "timestamp": 1704067200000 + i * 1000,
                    }
                ],
            },
        )
    return n_distractors + 1


def main() -> int:
    app = create_app(Settings(data_dir=pathlib.Path(tempfile.mkdtemp())))
    with TestClient(app) as client:
        total = build_corpus(client, n_distractors=200)
        # Identify the relevant memory's id so we can measure its rank rather
        # than only how many items came back. Rank is what decides whether the
        # evidence survives the platform's prefix truncation.
        probe = client.post(
            "/search",
            json={
                "query": "What caused the slow checkout latency in src/orders/repository.py?",
                "user_id": "u1",
                "top_k": 5,
            },
        ).json()["data"]
        relevant_id = probe[0]["id"] if probe else "?"

        print(f"corpus: {total} memories (1 relevant, {total - 1} same-repo distractors)")
        print(f"relevant memory id: {relevant_id}")
        print(f"{'probe':16} {'n':>4} {'top':>7} {'rank':>5}  tokens  returned content")
        print("-" * 96)

        for label, query in PROBES:
            data = client.post(
                "/search", json={"query": query, "user_id": "u1", "top_k": 100}
            ).json()["data"]
            top = f"{data[0]['score']:.3f}" if data else "-"
            ranks = [i for i, item in enumerate(data, start=1) if item["id"] == relevant_id]
            rank = str(ranks[0]) if ranks else "-"
            tokens = sum(len(i["content"]) // 4 for i in data)
            snippet = data[0]["content"][:38].replace("\n", " / ") if data else ""
            print(
                f"{label:16} {len(data):>4} {top:>7} {rank:>5}  {tokens:>6}  {snippet}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
