"""Characterize where the noise gate fires.

Not a test: a diagnostic used to decide how aggressive the gate should be.
Run:  PYTHONPATH=src python scripts/characterize_gate.py
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

MEMORY = """Session 42: investigating slow checkout latency.

We profiled the handler and found an N+1 query in src/orders/repository.py:

```python
for order in orders:
    items = session.query(Item).filter(Item.order_id == order.id).all()
```

Fix was to eager-load the relationship with joinedload, cutting p95 latency
from 840ms to 95ms.
"""

PROBES = [
    ("exact topic", "Why was checkout latency slow and how was it fixed?"),
    ("same words", "What caused the slow checkout latency in src/orders/repository.py?"),
    ("paraphrase", "Our order page takes ages to load. What did we do about performance?"),
    ("vague on-topic", "What performance improvements were made?"),
    ("generic repo", "What changes were made to the codebase?"),
    ("mild overlap", "How do we handle database queries in general?"),
    ("unrelated", "How do I bake sourdough bread with a crisp crust?"),
    ("other domain", "What is the capital of Mongolia?"),
    ("symbol only", "What is joinedload used for?"),
    ("path only", "What happened in src/orders/repository.py?"),
]


def main() -> int:
    app = create_app(Settings(data_dir=pathlib.Path(tempfile.mkdtemp())))
    with TestClient(app) as client:
        client.post(
            "/add",
            json={
                "request_id": "r1",
                "user_id": "u1",
                "session_id": "s1",
                "messages": [
                    {"role": "user", "content": MEMORY, "timestamp": 1704067200000}
                ],
            },
        )
        print(f"{'probe':16} {'n':>3} {'top':>7}  returned content")
        print("-" * 88)
        for label, query in PROBES:
            data = client.post(
                "/search", json={"query": query, "user_id": "u1", "top_k": 100}
            ).json()["data"]
            top = f"{data[0]['score']:.3f}" if data else "-"
            snippet = data[0]["content"][:50].replace("\n", " / ") if data else ""
            print(f"{label:16} {len(data):>3} {top:>7}  {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
