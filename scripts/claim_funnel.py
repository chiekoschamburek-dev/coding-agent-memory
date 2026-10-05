"""The claim-anchor funnel: where do the 48/70 menu misses actually die?

The select-probe measured that the answer session reaches the top-8 menu
for only 22/70 claim questions. "Recall side" is the hypothesis, but the
funnel has three distinct stages and each implies a different lever:

  not pooled        no chunk of the answer session is in the recall pool
                    at all -> channel problem (card / query-side HyDE);
  pooled, gated out chunks reached the pool but none passed the
                    informative-channel gate (lexical/entity) — the answer
                    is dense-visible only -> ``dense_eligible@0.45`` (built,
                    one env flip, measured directional on anchor A);
  gated, ranked 9+  the session is admissible but the shipped order does
                    not seat it -> ranking (the parked LTR recipe).

Also measured on the same replay: the dense-only-admission variant
(score_candidates with dense_only_min_similarity=0.45) — how the buckets
move when the gate opens to dense-only candidates.

Zero relay calls; one Add pass. Tune split only — the sealed 35 are not
touched.

Usage::

    PYTHONPATH=src python scripts/claim_funnel.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codemem.core.config import Settings  # noqa: E402
from codemem.search.evidence import (  # noqa: E402
    INFORMATIVE_CHANNELS,
    score_candidates,
)
from codemem.search.query import plan_query  # noqa: E402

MENU = 8
DENSE_ONLY_FLOOR = 0.45


def bucket(order_pos: int | None) -> str:
    if order_pos is None:
        return "gated"
    return "menu" if order_pos < MENU else "rank9+"


def main() -> int:
    import logging

    logging.disable(logging.WARNING)

    from fastapi.testclient import TestClient

    from codemem.api.app import create_app

    bench = json.loads((ROOT / "eval/data/benchmark.json").read_text(encoding="utf-8"))
    claim = json.loads(
        (ROOT / "eval/data/qa_claim_tune.json").read_text(encoding="utf-8")
    )

    settings = Settings(data_dir=Path(tempfile.mkdtemp()))
    app = create_app(settings)
    client = TestClient(app)
    client.__enter__()
    container = app.state.container
    store = container.store
    pipeline = container.search

    for memory in bench["memories"]:
        client.post("/add", json={
            "request_id": f"cf:{memory['id']}",
            "user_id": memory["user_id"],
            "session_id": memory["session_id"],
            "messages": memory["messages"],
        })

    def stage(user_id: str, query: str, answer: str,
              dense_only: float | None) -> dict:
        plan = plan_query(query, None)
        candidates = pipeline.retriever.recall(user_id, plan)
        memories = pipeline.retriever.load(user_id, candidates)
        by_session: dict[str, set] = {}
        for c in candidates:
            m = memories.get(c.memory_id)
            if m is None:
                continue
            by_session.setdefault(m.session_id, set()).update(c.channels)
        if answer not in by_session:
            return {"stage": "not_pooled"}

        chunk_scores = store.entity_match_scores(user_id, plan.entities)
        entity_match = {}
        for cand in candidates:
            m = memories.get(cand.memory_id)
            if m is None or m.chunk_id is None:
                continue
            s = chunk_scores.get(m.chunk_id, 0.0)
            if s > 0:
                entity_match[m.id] = s
        scored = score_candidates(
            candidates, memories, plan, entity_match,
            dense_only_min_similarity=dense_only,
            operative_weight=settings.operative_rank_weight,
        )
        reranked = pipeline._rerank(user_id, plan, scored, memories)  # noqa: SLF001
        order: list[str] = []
        for cand in reranked:
            m = memories.get(cand.memory_id)
            if m is None:
                continue
            if m.session_id not in order:
                order.append(m.session_id)
        pos = order.index(answer) if answer in order else None
        return {"stage": bucket(pos), "pos": pos}

    rows = []
    for question in claim["questions"]:
        user_id = next(
            m["user_id"] for m in bench["memories"] if m["repo"] == question["repo"]
        )
        answer = question["answer_session"]
        base = stage(user_id, question["question"], answer, None)
        gate = stage(user_id, question["question"], answer, DENSE_ONLY_FLOOR)
        rows.append({
            "query_id": question["query_id"],
            "base": base["stage"],
            "base_pos": base.get("pos"),
            "gate045": gate["stage"],
            "gate045_pos": gate.get("pos"),
        })

    client.__exit__(None, None, None)

    for name in ("base", "gate045"):
        counts = Counter(r[name] for r in rows)
        total = len(rows)
        print(f"\n{name}: {dict(counts)}")
        for k in ("menu", "rank9+", "gated", "not_pooled"):
            print(f"  {k:11s} {counts.get(k, 0):3d}/{total}")

    moved = [
        r for r in rows
        if r["base"] != r["gate045"]
    ]
    print(f"\ndense-only admission @0.45 moves {len(moved)} questions:")
    for r in moved:
        print(f"  {r['query_id']:48s} {r['base']:11s} -> {r['gate045']:11s}")

    out = ROOT / "eval/results/claim_funnel.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
