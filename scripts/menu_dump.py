"""One expensive pass; every menu/digest analysis iterates on its dump.

Feeds the two post-campaign algorithm questions:

  A. shortlist diversification — the 20 shortlist-bottleneck misses happen
     because the score-greedy ``order[:8]`` lets same-file distractors crowd
     the answer session out of the LLM's menu. Testing any diversification
     rule needs the DEEP order (top-24), each session's file manifest and
     first message — none of which the archived diagnosis kept.

  B. digest composition — the 11 llm-judgment misses are summary quality.
     The shipped digest's "top chunks" are the two LONGEST member texts
     (service.py), not the query-relevant ones. Building upgraded digests
     needs per-session top chunks BY FINAL SCORE, cause-prose sentences,
     timestamps and the session's last pooled line.

Also re-measures F3 (cause x identifier) with a REPAIRED lexicon: the
session_features.py CAUSE regex ends in a literal double backslash, so it
never matched ordinary prose — F3's archived "too sparse, exactly 0.4746"
verdict measured a constant-zero feature. The lexicon here is the wider one
from build_qa_procedure.py (a3201d7), correctly terminated.

Feature and fusion arithmetic runs over the FULL session order exactly as
the archived diagnosis computed it (its rare-word document frequency is
defined over all of ``order``); only serialisation is cut at top-24.

Validation: the replayed shortlist, shipped payload and fusion payload must
match eval/results/select_miss_diagnosis.json bitwise — the same standard
the fusion replay was held to.

Usage::

    PYTHONPATH=src python scripts/menu_dump.py
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from dataclasses import replace as _replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from codemem.core.config import Settings  # noqa: E402
from codemem.search.evidence import (  # noqa: E402
    assemble as assemble_ship,
    score_candidates,
    session_terms,
)
from codemem.search.query import plan_query  # noqa: E402

# build_qa_procedure.py's wider connector lexicon (a3201d7), with the
# word boundary spelled correctly.
CAUSE = re.compile(
    r"\b(because|caused by|due to|which (?:causes?|fails?|leads? to|makes? it)|"
    r"root cause|the problem is|the issue is|"
    r"(?:does not|doesn't) handle|fails? when|instead of|rather than|"
    r"must (?:also|be|set|be updated)|requires? (?:that|both|updating|the)|"
    r"in order to|otherwise|silently (?:ignored|fails?)|needed to)\b",
    re.IGNORECASE,
)
SYMBOL = re.compile(r"`[^`]{2,}`|[a-z]+_[a-z_]+|[A-Z][a-z]+[A-Z]\w*|\w+\.\w+")
FILE = re.compile(r"[\w/\\.-]+\.\w{1,4}\b")

DEEP = 24          # sessions serialised per query, far past the 8-slot menu
TOP_CHUNKS = 4     # by final score, for digest composition
CAUSE_CAP = 3      # cause-prose members kept per session


def _rel_ord(members: list, m) -> float:
    ords = [x.ord for x in members]
    lo, hi = min(ords), max(ords)
    return 0.5 if hi <= lo else (m.ord - lo) / (hi - lo)


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
    archived = {
        r["query_id"]: r
        for r in json.loads(
            (ROOT / "eval/results/select_miss_diagnosis.json").read_text(
                encoding="utf-8"
            )
        )["rows"]
    }

    settings = Settings(data_dir=Path(tempfile.mkdtemp()))
    app = create_app(settings)
    client = TestClient(app)
    client.__enter__()
    container = app.state.container
    store = container.store
    pipeline = container.search
    embedder = container.embedder

    for memory in bench["memories"]:
        client.post("/add", json={
            "request_id": f"md:{memory['id']}",
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
            m["user_id"] for m in bench["memories"] if m["repo"] == question["repo"]
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
        final_by_mem = {c.memory_id: c.final for c in reranked}

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

        # ---- features over the FULL order, the archived recipe verbatim ---
        firsts = store.first_messages(user_id, order)
        f1 = {sid: 0.0 for sid in order}
        if embedder is not None and embedder.available:
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
        query_idents = set()
        for vals in plan.entities.values():
            query_idents.update(v.lower() for v in vals if v)
        query_idents.update(t.lower() for t in SYMBOL.findall(query))
        f3 = {}
        for sid in order:
            hits = [
                m for m in members[sid]
                if 60 <= len(m.text) <= 800 and CAUSE.search(m.text)
            ]
            idents = {
                s.lower() for m in hits for s in SYMBOL.findall(m.text)
            } & query_idents
            f3[sid] = len(idents) / len(query_idents) if query_idents else 0.0

        head_rank = {sid: i for i, sid in enumerate(order)}
        f1_rank = {sid: i for i, sid in enumerate(
            sorted(order, key=lambda s: -f1[s]))}
        rare_rank = {sid: i for i, sid in enumerate(
            sorted(order, key=lambda s: -rare_cov[s]))}
        f3_rank = {sid: i for i, sid in enumerate(
            sorted(order, key=lambda s: -f3[s]))}

        def fused(weights: dict[str, float]) -> list[str]:
            rank_maps = {
                "head": head_rank, "f1": f1_rank, "rare": rare_rank, "f3": f3_rank,
            }
            scores = {
                sid: sum(
                    w / (60 + rank_maps[name].get(sid, 99))
                    for name, w in weights.items()
                )
                for sid in order
            }
            return sorted(scores, key=lambda s: -scores[s])

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
            wide_settings = _replace(settings, evidence_max_sessions=max_sessions)
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

        orders = {
            "shipped": order,
            "fused12": fused({"head": 1.0, "f1": 2.0, "rare": 2.0}),
            "fused123c": fused(
                {"head": 1.0, "f1": 2.0, "rare": 2.0, "f3": 2.0}),
            "f3alone": fused({"f3": 2.0}),
        }

        sessions_out = []
        for rank, sid in enumerate(deep := order[:DEEP], start=1):
            mems = members[sid]
            by_score = sorted(
                mems, key=lambda m: -final_by_mem.get(m.id, 0.0)
            )
            head = by_score[0] if by_score else None
            cause_hits = [
                m for m in mems
                if 60 <= len(m.text) <= 800 and CAUSE.search(m.text)
            ]
            last = max(mems, key=lambda m: (m.ts or 0, m.ord)) if mems else None
            ts_vals = [m.ts for m in mems if m.ts]
            sessions_out.append({
                "sid": sid,
                "rank": rank,
                "n_members": len(mems),
                "head_score": round(
                    final_by_mem.get(head.id, 0.0) if head else 0.0, 4),
                "files": sorted({
                    f for t in (m.text for m in mems) for f in FILE.findall(t)
                })[:8],
                "first": (firsts.get(sid) or "")[:280],
                "last": (last.text[:240] if last else ""),
                "ts_min": min(ts_vals) if ts_vals else None,
                "ts_max": max(ts_vals) if ts_vals else None,
                "top_chunks": [
                    {
                        "text": m.text[:240],
                        "score": round(final_by_mem.get(m.id, 0.0), 4),
                        "rel_ord": round(_rel_ord(mems, m), 3),
                        "kind": m.structural_kind,
                    }
                    for m in by_score[:TOP_CHUNKS]
                ],
                "cause": [
                    {"text": m.text[:240], "rel_ord": round(_rel_ord(mems, m), 3)}
                    for m in sorted(
                        cause_hits, key=lambda m: -_rel_ord(mems, m)
                    )[:CAUSE_CAP]
                ],
                "longest": [
                    m.text[:240] for m in sorted(mems, key=lambda m: -len(m.text))[:2]
                ],
                "f1": round(f1[sid], 4),
                "rare_cov": round(rare_cov[sid], 4),
                "f3": round(f3[sid], 4),
            })

        rows.append({
            "query_id": qid,
            "question": query,
            "options": question.get("options"),
            "gold_claim": question.get("gold_claim"),
            "gold_index": question.get("gold_index"),
            "answer_session": question.get("answer_session"),
            "distractor_sessions": question.get("distractor_sessions"),
            "sel_shown": bool(outcome["answer_session_shown"]),
            "plan": {
                "intent": plan.intent,
                "entities": plan.entities,
                "keywords": plan.keywords[:30],
                "probes": plan.probes,
            },
            "sessions": sessions_out,
            "payloads": {name: payload(o) for name, o in orders.items()},
        })
        print(f"dumped {qid} ({len(rows)}/{len(qa['questions'])})", flush=True)

    client.__exit__(None, None, None)

    # ---- validation: bitwise against the archived diagnosis ----------------
    bad = 0
    for r in rows:
        a = archived.get(r["query_id"])
        if a is None:
            continue
        if [s["sid"] for s in r["sessions"][:8]] != a["shortlist"]:
            print(f"MISMATCH shortlist {r['query_id']}")
            bad += 1
        if r["payloads"]["shipped"] != a["shipped_payload"]:
            print(f"MISMATCH shipped payload {r['query_id']}")
            bad += 1
        if r["payloads"]["fused12"] != a["fusion_payload"]:
            print(f"MISMATCH fused12 payload {r['query_id']}")
            bad += 1
    print(f"\nvalidation vs archived diagnosis: "
          f"{len(rows) - bad}/{len(rows)} rows bitwise-identical")

    out = ROOT / "eval/results/menu_dump.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, ensure_ascii=False, indent=1, default=str)
    print(f"wrote {out}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
