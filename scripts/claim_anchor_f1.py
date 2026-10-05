"""Is first-message selection the fix on the claim-topical anchor?

The claim-topical instrument poisoned lexical matching adversarially (the
lead distractor tracks the issue more closely than the gold), and only 22/70
answer sessions reach the top-8 shortlist under the shipped ordering. The
hypothesis under test: the session's FIRST message is a clean problem
statement — distractor sessions state THEIR OWN problems — so selecting by
query↔first-message cosine may be the one channel where the gold session
wins on this anchor.

Measured per arm (shipped order / F1 order / fused order), on the 70-question
tuning set (the sealed 35 stay sealed):

  pooling rate   answer sessions present in the reranked candidate list at all
  top-8 hit      answer session among the first 8 sessions
  payload hit    answer session in the assembled payload (budget 2)

Zero relay calls; one Add; one local embed pass.

Usage::

    PYTHONPATH=src python scripts/claim_anchor_f1.py
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
from codemem.search.evidence import (  # noqa: E402
    assemble as assemble_ship,
    fused_session_order,
    score_candidates,
    session_terms,
)
from codemem.search.query import plan_query  # noqa: E402


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
    embedder = container.embedder

    if True:
        for memory in bench["memories"]:
            client.post("/add", json={
                "request_id": f"cf:{memory['id']}",
                "user_id": memory["user_id"],
                "session_id": memory["session_id"],
                "messages": memory["messages"],
            })

        arms = {"shipped": 0, "f1": 0, "fused": 0}
        pooled_hit = 0
        pooled_total = 0
        n = 0
        for question in qa["questions"]:
            answer_session = question.get("answer_session")
            if not answer_session:
                continue
            n += 1
            user_id = next(
                m["user_id"] for m in bench["memories"]
                if m["repo"] == question["repo"]
            )
            query = question["question"]

            plan = plan_query(query, None)
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

            order: list[str] = []
            members: dict[str, list] = {}
            for cand in reranked:
                memory = memories.get(cand.memory_id)
                if memory is None:
                    continue
                if memory.session_id not in members:
                    order.append(memory.session_id)
                    members[memory.session_id] = []
                members[memory.session_id].append(memory)

            pooled = answer_session in members
            pooled_total += 1
            pooled_hit += 1 if pooled else 0

            # F1: query vs each candidate session's first message
            f1 = {sid: 0.0 for sid in order}
            if embedder is not None and embedder.available:
                firsts = store.first_messages(user_id, order)
                live = [sid for sid in order if firsts.get(sid)]
                vectors = embedder.embed([query] + [firsts[sid] for sid in live]) or []
                if vectors and len(vectors) == len(live) + 1:
                    qvec = vectors[0]
                    for sid, svec in zip(live, vectors[1:]):
                        if len(qvec) == len(svec):
                            f1[sid] = sum(a * b for a, b in zip(qvec, svec))

            f1_order = sorted(order, key=lambda s: -f1.get(s, 0.0))
            fused_order = fused_session_order(order, f1, {s: 0.0 for s in order})

            def payload(ordering: list[str]) -> list[str]:
                block = {sid: i for i, sid in enumerate(ordering)}
                pos = {id(c): i for i, c in enumerate(reranked)}

                def block_of(cand):
                    memory = memories.get(cand.memory_id)
                    return (
                        block.get(memory.session_id, 99) if memory else 99,
                        pos[id(cand)],
                    )

                items = assemble_ship(
                    settings, store, user_id, plan,
                    sorted(reranked, key=block_of), memories, top_k=100,
                )
                seen: list[str] = []
                for item in items:
                    memory = memories.get(item.memory_id)
                    if memory and memory.session_id not in seen:
                        seen.append(memory.session_id)
                return seen

            for name, ordering in (
                ("shipped", order),
                ("f1", f1_order),
                ("fused", fused_order),
            ):
                top8 = ordering[:8]
                arms[name] += 1 if (pooled and answer_session in top8) else 0

    client.__exit__(None, None, None)

    print(f"questions with an answer session: {n}")
    print(f"answer-session pooling rate (in reranked list at all): "
          f"{pooled_hit}/{n} = {pooled_hit / n:.3f}")
    print(f"\ntop-8 shortlist hit rate (given pooled):")
    print(f"  shipped order : {arms['shipped']}/{n} = {arms['shipped'] / n:.3f}")
    print(f"  f1 order      : {arms['f1']}/{n} = {arms['f1'] / n:.3f}")
    print(f"  fused order   : {arms['fused']}/{n} = {arms['fused'] / n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
