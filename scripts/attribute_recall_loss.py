"""Attribute the recall/ranking loss to one pipeline stage.

The proxy metrics say *that* recall@10 is 0.47-0.66 but not *where* the missing
evidence is lost. This replays the search internals per query and, for every
(query, relevant session) pair, records the last stage the pair survived:

  1. pool      - at least one entry of the session reached the fused recall pool
  2. admitted  - at least one entry passed the informative-channel admission in
                 ``score_candidates`` (dense-only candidates do not count)
  3. payload   - the session appears in the payload when the session budget is
                 lifted (``evidence_max_sessions=0``), i.e. it cleared the noise
                 gate and the token/top_k budget
  4. emitted   - the session survives the shipped session budget
                 (``evidence_max_sessions``, default 2)
  5. window    - the session holds one of the first ``--k`` entry slots

Losses between consecutive stages are attributed to the stage that caused them,
which separates "recall scope too narrow", "ranking puts it too low", "session
budget too small" and "per-session packing crowds the window".

Every pair is also split by label strength (shared-file overlap), because the
weak labels (one shared hotspot file) are not something the text carries.

    PYTHONPATH=src python scripts/attribute_recall_loss.py
    PYTHONPATH=src python scripts/attribute_recall_loss.py -k 10 --out eval/results/funnel.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "eval")

import logging

logging.disable(logging.WARNING)

from codemem.api.app import Container  # noqa: E402
from codemem.core.config import Settings  # noqa: E402
from codemem.core.schemas import Message  # noqa: E402
from codemem.search.evidence import assemble, score_candidates  # noqa: E402
from codemem.search.query import plan_query  # noqa: E402


def build(container: Container, data: dict) -> None:
    add = container.add
    for memory in data["memories"]:
        add.handle(
            request_id=f"bench:{memory['id']}",
            user_id=memory["user_id"],
            session_id=memory["session_id"],
            messages=[Message(**m) for m in memory["messages"]],
        )


def analyse(container: Container, data: dict, k: int, overrides: dict) -> list[dict]:
    settings = container.settings
    pipeline = container.search
    store = container.store
    session_files = {m["session_id"]: set(m.get("files") or ()) for m in data["memories"]}

    rows: list[dict] = []
    for query in data["queries"]:
        relevant = list(query["relevant"])
        if not relevant:
            continue
        user_id = next(
            m["user_id"] for m in data["memories"] if m["repo"] == query["repo"]
        )
        plan = plan_query(query["query"], None)
        candidates = pipeline.retriever.recall(user_id, plan)
        memories = pipeline.retriever.load(user_id, candidates)

        chunk_scores = store.entity_match_scores(user_id, plan.entities)
        entity_match: dict[int, float] = {}
        for cand in candidates:
            memory = memories.get(cand.memory_id)
            if memory is None or memory.chunk_id is None:
                continue
            score = chunk_scores.get(memory.chunk_id, 0.0)
            if score > 0:
                entity_match[memory.id] = score

        scored = score_candidates(candidates, memories, plan, entity_match)
        reranked = pipeline._rerank(user_id, plan, scored, memories)  # noqa: SLF001

        def session_of(memory_id: int) -> str | None:
            memory = memories.get(memory_id)
            return memory.session_id if memory else None

        def sessions_of(cands) -> set[str]:
            return {s for s in (session_of(c.memory_id) for c in cands) if s}

        pool_sessions = sessions_of(candidates)
        admitted_sessions = sessions_of(scored)
        # Session order after scoring+rerank: first appearance of a session in the
        # ranked entry list is that session's rank.
        session_order: list[str] = []
        session_head_score: dict[str, float] = {}
        for cand in reranked:
            s = session_of(cand.memory_id)
            if s and s not in session_head_score:
                session_order.append(s)
                session_head_score[s] = cand.final

        wide_settings = replace(settings, evidence_max_sessions=0)
        items_wide = assemble(
            wide_settings, store, user_id, plan, list(reranked), memories, top_k=100
        )
        items_ship = assemble(
            settings, store, user_id, plan, list(reranked), memories, top_k=100
        )
        wide_sessions = sessions_of(items_wide)
        ship_sessions = sessions_of(items_ship)
        window_sessions = sessions_of(items_ship[:k])

        q_files = set(query.get("files") or ())
        for sid in relevant:
            overlap = len(q_files & session_files.get(sid, set()))
            rows.append(
                {
                    "query_id": query["query_id"],
                    "repo": query["repo"],
                    "session_id": sid,
                    "overlap": overlap,
                    "in_pool": sid in pool_sessions,
                    "admitted": sid in admitted_sessions,
                    "session_rank": (
                        session_order.index(sid) + 1 if sid in session_order else None
                    ),
                    "head_final": session_head_score.get(sid),
                    "cleared_gate": sid in wide_sessions,
                    "emitted": sid in ship_sessions,
                    "in_window": sid in window_sessions,
                }
            )
        rows.append(
            {
                "query_id": query["query_id"],
                "repo": query["repo"],
                "session_id": "__shape__",
                "overlap": 0,
                "n_entries_pool": len(candidates),
                "n_entries_admitted": len(scored),
                "n_entries_wide": len(items_wide),
                "n_sessions_wide": len(wide_sessions),
                "n_entries_ship": len(items_ship),
                "n_sessions_ship": len(ship_sessions),
            }
        )
    return rows


def report(rows: list[dict], k: int) -> dict:
    pairs = [r for r in rows if r["session_id"] != "__shape__"]
    shapes = [r for r in rows if r["session_id"] == "__shape__"]

    def stage_share(subset: list[dict]) -> dict[str, float]:
        n = len(subset) or 1
        return {
            "in_pool": sum(r["in_pool"] for r in subset) / n,
            "admitted": sum(r["admitted"] for r in subset) / n,
            "cleared_gate": sum(r["cleared_gate"] for r in subset) / n,
            "emitted": sum(r["emitted"] for r in subset) / n,
            "in_window": sum(r["in_window"] for r in subset) / n,
        }

    def losses(subset: list[dict]) -> dict[str, float]:
        """Share of pairs whose LAST survived stage is each boundary."""
        n = len(subset) or 1
        lost = {
            "1 recall scope (never pooled)": sum(1 for r in subset if not r["in_pool"]),
            "2 admission (no lexical/entity hit)": sum(
                1 for r in subset if r["in_pool"] and not r["admitted"]
            ),
            "3 noise gate / budget": sum(
                1 for r in subset if r["admitted"] and not r["cleared_gate"]
            ),
            "4 session budget (emitted too few)": sum(
                1 for r in subset if r["cleared_gate"] and not r["emitted"]
            ),
            "5 packing / rank inside window": sum(
                1 for r in subset if r["emitted"] and not r["in_window"]
            ),
            "recovered": sum(1 for r in subset if r["in_window"]),
        }
        return {key: value / n for key, value in lost.items()}

    strong = [r for r in pairs if r["overlap"] >= 2]
    weak = [r for r in pairs if r["overlap"] == 1]

    print(f"pairs: {len(pairs)} total / {len(strong)} strong (>=2 files) / {len(weak)} weak")
    print(f"queries: {len(shapes)}   window k = {k} entries")
    print()
    mean = lambda key: sum(r[key] for r in shapes) / len(shapes)  # noqa: E731
    print("payload shape (mean per query)")
    print(f"  entries in the recall pool        : {mean('n_entries_pool'):.0f}")
    print(f"  entries admitted for scoring      : {mean('n_entries_admitted'):.0f}")
    print(
        f"  payload with no session budget    : {mean('n_entries_wide'):.1f} entries / "
        f"{mean('n_sessions_wide'):.1f} sessions"
    )
    print(
        f"  payload as shipped                : {mean('n_entries_ship'):.1f} entries / "
        f"{mean('n_sessions_ship'):.1f} sessions"
    )
    rel_per_query = collections.Counter()
    for r in pairs:
        rel_per_query[r["query_id"]] += 1
    print(
        f"  relevant sessions per query       : {sum(rel_per_query.values())/len(rel_per_query):.2f}"
    )
    print()
    for label, subset in [("ALL", pairs), ("STRONG", strong), ("WEAK", weak)]:
        print(f"{label} — stage survival")
        for key, value in stage_share(subset).items():
            print(f"  {key:16s}{value:>8.1%}")
        print(f"{label} — loss attribution")
        for key, value in losses(subset).items():
            print(f"  {key:34s}{value:>8.1%}")
        print()

    ranks = [
        r["session_rank"]
        for r in strong
        if r["admitted"] and r["session_rank"] is not None
    ]
    if ranks:
        buckets = collections.Counter(
            "1-2" if v <= 2 else "3-6" if v <= 6 else "7-20" if v <= 20 else "21+"
            for v in ranks
        )
        total = sum(buckets.values())
        print("session-rank of ADMITTED strong-relevant sessions (shipped order)")
        for name in ("1-2", "3-6", "7-20", "21+"):
            print(f"  {name:6s}{buckets[name]:>5} ({buckets[name]/total:>6.1%})")
    return {
        "pairs": len(pairs),
        "all": {**stage_share(pairs), **losses(pairs)},
        "strong": {**stage_share(strong), **losses(strong)},
        "weak": {**stage_share(weak), **losses(weak)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("eval/data/benchmark.json"))
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = json.loads(args.data.read_text(encoding="utf-8"))
    if args.limit:
        data["queries"] = [
            q for q in data["queries"] if q["relevant"]
        ][: args.limit]

    overrides = {}
    if args.max_sessions is not None:
        overrides["evidence_max_sessions"] = args.max_sessions
    if args.cap is not None:
        overrides["max_evidence_per_session"] = args.cap

    settings = Settings(data_dir=Path(tempfile.mkdtemp()), **overrides)
    container = Container(settings)
    try:
        build(container, data)
        rows = analyse(container, data, args.k, overrides)
    finally:
        container.close()

    summary = report(rows, args.k)
    if args.out:
        args.out.write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
