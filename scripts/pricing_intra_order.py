"""Offline pricing for intra-session emission ordering (claim-delivery metric).

The menu is saturated (97.1 % top-8 under with-options plans) but only 13.5 %
of gold chunks are emitted — the constraint moved from "which session" to
"which content". This replay prices the lever directly at the content level:
does the gold claim (the verbatim sentence the question asks about) arrive
whole in the payload, as the weight lifts intra-session emission order by
query-keyword coverage?

Sweep w in {0, 0.5, 1.0, 2.0} on the claim-tune set (70 questions), under
BOTH plans (bare and with-options — the deployment runs the latter; the
instrument rule after two bare-plan reversals). Zero relay calls.

Pre-registered gate: w must deliver a net gain of >= +3 claims at some weight
on BOTH plans to advance to an answer-accuracy run.

Usage::

    PYTHONPATH=src python scripts/pricing_intra_order.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from codemem.core.config import Settings  # noqa: E402
from codemem.search.evidence import assemble as assemble_ship  # noqa: E402
from codemem.search.evidence import score_candidates  # noqa: E402
from codemem.search.query import plan_query  # noqa: E402

sys.path.insert(0, str(ROOT / "eval"))
from run_evidence import claim_delivered  # noqa: E402


def main() -> int:
    import logging

    logging.disable(logging.WARNING)

    from fastapi.testclient import TestClient

    from codemem.api.app import create_app

    bench = json.loads((ROOT / "eval/data/benchmark.json").read_text(encoding="utf-8"))
    qa = json.loads(
        (ROOT / "eval/data/qa_claim_tune.json").read_text(encoding="utf-8")
    )

    settings = Settings(data_dir=Path(tempfile.mkdtemp()))
    app = create_app(settings)
    client = TestClient(app)
    client.__enter__()
    container = app.state.container
    store = container.store
    pipeline = container.search

    if True:
        for memory in bench["memories"]:
            client.post("/add", json={
                "request_id": f"pio:{memory['id']}",
                "user_id": memory["user_id"],
                "session_id": memory["session_id"],
                "messages": memory["messages"],
            })

        weights = (0.0, 0.5, 1.0, 2.0, 99.0)  # 99 = coverage-primary (final as tiebreak only)
        delivered = {}
        for w in weights:
            delivered[f"w{w}"] = 0
            delivered[f"w{w}/opts"] = 0
        reach = {f"w{w}": 0 for w in weights}
        blob_tokens = {f"w{w}": 0 for w in weights}
        n = 0

        for question in qa["questions"]:
            gold_claim = question.get("gold_claim")
            answer_session = question.get("answer_session")
            if not gold_claim or not answer_session:
                continue
            n += 1
            user_id = next(
                m["user_id"] for m in bench["memories"]
                if m["repo"] == question["repo"]
            )
            query = question["question"]
            options = question.get("options") or []

            plans = {
                "": plan_query(query, None),
                "opts": plan_query(query, options),
            }
            for plan_key, plan in plans.items():
                candidates = pipeline.retriever.recall(user_id, plan)
                memories = pipeline.retriever.load(user_id, candidates)
                chunk_scores = store.entity_match_scores(user_id, plan.entities)
                entity_match = {}
                for cand in candidates:
                    memory = memories.get(cand.memory_id)
                    if memory is None or memory.chunk_id is None:
                        continue
                    score = chunk_scores.get(memory.chunk_id, 0.0)
                    if score > 0:
                        entity_match[memory.id] = score
                scored = score_candidates(
                    candidates, memories, plan, entity_match,
                    operative_weight=settings.operative_rank_weight,
                )
                reranked = pipeline._rerank(user_id, plan, scored, memories)  # noqa: SLF001

                for w in weights:
                    ws = replace_settings(settings, w)
                    items = assemble_ship(
                        ws, store, user_id, plan, list(reranked), memories, top_k=100,
                    )
                    blob = "\n".join(item.content for item in items)
                    ok = claim_delivered(blob, gold_claim)
                    delivered[f"w{w}{plan_key}"] = delivered.get(f"w{w}{plan_key}", 0) + (1 if ok else 0)
                    if w == 0.0:
                        shown_sessions = {
                            memories[item.memory_id].session_id
                            for item in items
                            if item.memory_id in memories
                        }
                        # count the bare plan only: the dict iterates bare
                        # first, then options — a naive += double-counts
                        if plan_key == "":
                            reach[f"w{w}"] += 1 if answer_session in shown_sessions else 0
                        blob_tokens[f"w{w}"] += sum(item.tokens for item in items)

    client.__exit__(None, None, None)

    print(f"questions: {n}")
    print(f"\nclaim delivered whole (payload-level, both plans):")
    base_bare = delivered["w0.0"]
    base_opts = delivered["w0.0/opts"]
    for w in weights:
        b = delivered[f"w{w}"]
        o = delivered[f"w{w}/opts"]
        print(f"  w={w:<4} bare {b}/{n} ({b - base_bare:+d})   with_options {o}/{n} ({o - base_opts:+d})")
    print(f"\nreach at w=0 (context): {reach['w0.0']}/{n}")
    return 0


def replace_settings(settings, w):
    from dataclasses import replace
    return replace(settings, intra_session_order_weight=w)


if __name__ == "__main__":
    raise SystemExit(main())
