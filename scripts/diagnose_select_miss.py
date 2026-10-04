"""Why did select-llm still miss 53.4%? A zero-relay decomposition.

select-llm and feature-fusion both reached the answer session 46.6% of the
time, but only select-llm converted reach into answers (0.414 vs 0.345).
Before any stacking experiment, this replay splits select-llm's misses into
three buckets that imply three different next moves:

  shortlist-bottleneck  the answer session was outside the top-8 the LLM
                        saw — fusion feeding a better shortlist would help,
                        and the expected gain is exactly this bucket's size;
  gate-blocked          the answer session was in the shortlist but no chunk
                        of it passes the informative-channel gate — picking
                        it cannot emit it; that is the door problem (cards/
                        dense_eligible territory), not a selection problem;
  llm-judgment          admissible and shortlisted but not picked — summary
                        quality (parked-card territory).

Also answered on the same replay: do fusion and select-llm fix the SAME
queries (their reach gains are identical at +12.1pp — too neat), which
decides whether fusion is a shortlist feeder for select or pure redundancy.

Zero relay calls: one Add, one local embed pass over first messages and
issue texts. The select-llm arm's per-question reach comes from the recorded
run dump (eval/results/e2eL_selL.json), not from re-running anything.

Usage::

    PYTHONPATH=src python scripts/diagnose_select_miss.py
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
    INFORMATIVE_CHANNELS,
    assemble as assemble_ship,
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
        (ROOT / "eval/data/qa_procedure_large.json").read_text(encoding="utf-8")
    )
    sel_run = json.loads(
        (ROOT / "eval/results/e2eL_selL.json").read_text(encoding="utf-8")
    )
    sel_out = {o["query_id"]: o for o in sel_run["outcomes"]["with_memory"]}

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
                "request_id": f"dm:{memory['id']}",
                "user_id": memory["user_id"],
                "session_id": memory["session_id"],
                "messages": memory["messages"],
            })

        rows = []
        for question in qa["questions"]:
            qid = question["query_id"]
            outcome = sel_out.get(qid)
            if outcome is None:
                continue
            user_id = next(
                m["user_id"] for m in bench["memories"]
                if m["repo"] == question["repo"]
            )
            # the e2e runs searched with the full question text
            query = question["question"]
            answer_session = question.get("answer_session")

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
            admitted_sessions: set[str] = set()
            for cand in reranked:
                memory = memories.get(cand.memory_id)
                if memory is None:
                    continue
                if memory.session_id not in members:
                    order.append(memory.session_id)
                    members[memory.session_id] = []
                    if any(name in cand.channels for name in INFORMATIVE_CHANNELS):
                        admitted_sessions.add(memory.session_id)
                members[memory.session_id].append(memory)

            shortlist = order[:8]

            # F1/F2 (the fusion signals) on this question
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
            query_terms = session_terms(query)
            unions = {
                sid: set().union(*(session_terms(m.text) for m in members[sid]))
                if members[sid] else set()
                for sid in order
            }
            rare = {
                t for t in query_terms
                if sum(1 for sid in order if t in unions[sid]) <= 2
            }
            rare_cov = {
                sid: (len(unions[sid] & rare) / len(rare) if rare else 0.0)
                for sid in order
            }
            fused_scores = {}
            head_rank = {sid: i for i, sid in enumerate(order)}
            f1_rank = {sid: i for i, sid in enumerate(
                sorted(order, key=lambda s: -f1.get(s, 0.0)))}
            rare_rank = {sid: i for i, sid in enumerate(
                sorted(order, key=lambda s: -rare_cov.get(s, 0.0)))}
            for sid in order:
                fused_scores[sid] = (
                    1.0 / (60 + head_rank.get(sid, 99))
                    + 2.0 / (60 + f1_rank.get(sid, 99))
                    + 2.0 / (60 + rare_rank.get(sid, 99))
                )
            fused_order = sorted(fused_scores, key=lambda s: -fused_scores[s])

            from dataclasses import replace as _replace

            def payload(ordering: list[str], max_sessions: int = 2) -> list[str]:
                block = {sid: i for i, sid in enumerate(ordering)}
                pos = {id(c): i for i, c in enumerate(reranked)}

                def block_of(cand):
                    memory = memories.get(cand.memory_id)
                    return (
                        block.get(memory.session_id, 99) if memory else 99,
                        pos[id(cand)],
                    )

                ordered = sorted(reranked, key=block_of)
                wide_settings = _replace(
                    settings, evidence_max_sessions=max_sessions
                )
                items = assemble_ship(
                    wide_settings, store, user_id, plan,
                    list(ordered), memories, top_k=100,
                )
                seen: list[str] = []
                for item in items:
                    memory = memories.get(item.memory_id)
                    if memory and memory.session_id not in seen:
                        seen.append(memory.session_id)
                return seen

            rows.append({
                "query_id": qid,
                "answer_session": answer_session,
                "sel_shown": bool(outcome["answer_session_shown"]),
                "shortlist": shortlist,
                "admitted": sorted(admitted_sessions),
                "shipped_payload": payload(order),
                "fusion_payload": payload(fused_order),
                "wide_payload": payload(order, max_sessions=0),
            })

    client.__exit__(None, None, None)

    # ---- decomposition ------------------------------------------------------
    missed = [r for r in rows if r["answer_session"] and not r["sel_shown"]]
    reached = [r for r in rows if r["answer_session"] and r["sel_shown"]]
    print(f"select-llm questions: {len(rows)}, reached: {len(reached)}, "
          f"missed: {len(missed)}")

    buckets = {"shortlist-bottleneck": 0, "gate-blocked": 0, "llm-judgment": 0}
    rescueable_by_shortlist = 0
    for r in missed:
        ans = r["answer_session"]
        if ans not in r["shortlist"]:
            buckets["shortlist-bottleneck"] += 1
            # would a perfect shortlist have delivered it? admissible AND
            # wide payload shows the session can be emitted
            if ans in r["wide_payload"]:
                rescueable_by_shortlist += 1
        elif ans not in r["wide_payload"]:
            buckets["gate-blocked"] += 1
        else:
            buckets["llm-judgment"] += 1
    print(f"\nmiss decomposition (n={len(missed)}):")
    for name, count in buckets.items():
        print(f"  {name:22s} {count:3d}")
    print(f"\nrescuable by a perfect shortlist (fusion's realistic ceiling "
          f"on select-llm): {rescueable_by_shortlist} queries")

    # fusion vs select: same queries fixed?
    base = {r["query_id"]: r for r in rows}
    fusion_fixed = {
        r["query_id"] for r in rows
        if r["answer_session"]
        and r["answer_session"] in r["fusion_payload"]
        and r["answer_session"] not in r["shipped_payload"]
    }
    select_fixed = {
        r["query_id"] for r in rows
        if r["answer_session"] and r["sel_shown"]
        and r["answer_session"] not in r["shipped_payload"]
    }
    print(f"\nfixed vs shipped payload: fusion {len(fusion_fixed)}, "
          f"select {len(select_fixed)}, both {len(fusion_fixed & select_fixed)}, "
          f"select-only {len(select_fixed - fusion_fixed)}, "
          f"fusion-only {len(fusion_fixed - select_fixed)}")

    # where would fusion-feeding-select land?
    # the fused shortlist approximation: the fusion payload's session order
    # restricted to gate-passing sessions; membership of the answer session
    # there says whether a fused shortlist would even have shown it to the LLM
    fusion_top8_hit = 0
    fusion_top8_hit_rescueable = 0
    for r in missed:
        ans = r["answer_session"]
        # the fusion payload's session order IS the fused ordering restricted
        # to gate-passing sessions; membership in its first 8 approximates the
        # fused shortlist
        if ans in r["fusion_payload"][:8] or ans in r["fusion_payload"]:
            fusion_top8_hit += 1
            if ans in r["wide_payload"]:
                fusion_top8_hit_rescueable += 1
    print(f"\nof the {len(missed)} misses, the answer session appears in the "
          f"fusion payload (first 8 or later) for {fusion_top8_hit}; "
          f"admissible: {fusion_top8_hit_rescueable}")

    out = ROOT / "eval/results/select_miss_diagnosis.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump({
            "rows": [
                {k: v for k, v in r.items()} for r in rows
            ],
            "buckets": buckets,
            "rescueable_by_shortlist": rescueable_by_shortlist,
            "fusion_fixed": sorted(fusion_fixed),
            "select_fixed": sorted(select_fixed),
        }, handle, ensure_ascii=False, indent=1, default=str)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
