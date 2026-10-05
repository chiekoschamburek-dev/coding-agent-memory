"""Feature dump for the small LTR combiner, both anchors, one Add pass.

Pre-registered in eval/README.md ("The small LTR combiner"). Ten features
per (query, session), all query-internal: the four scoring terms recomputed
verbatim from the Candidate fields the way score_candidates computes them
(base = rrf/max_rrf over the eligible pool, coverage = informative-channel
agreement, strength = within-channel normalised magnitude, entity =
IDF-weighted identifier match), plus F1 (first-message cosine), rare-
vocabulary union coverage, the corrected F3, rank in the shipped order,
and pooled-member count.

Labels:
  anchor A  the benchmark's own file-overlap ``relevant`` sets
            (multi-positive; strong = >= 2 shared files, computed from the
            benchmark's own per-session file lists);
  anchor B  the claim instrument's answer sessions (topical anchor).

Output: eval/results/ltr_data.json — {anchor, query_id, sessions:[{sid,
features, label, strong}]} rows. Zero relay calls.

Usage::

    PYTHONPATH=src python scripts/ltr_data.py
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from codemem.core.config import Settings  # noqa: E402
from codemem.search.evidence import (  # noqa: E402
    INFORMATIVE_CHANNELS,
    score_candidates,
    session_terms,
)
from codemem.search.query import plan_query  # noqa: E402

CAUSE = re.compile(
    r"\b(because|caused by|due to|which (?:causes?|fails?|leads? to|makes? it)|"
    r"root cause|the problem is|the issue is|"
    r"(?:does not|doesn't) handle|fails? when|instead of|rather than|"
    r"must (?:also|be|set|be updated)|requires? (?:that|both|updating|the)|"
    r"in order to|otherwise|silently (?:ignored|fails?)|needed to)\b",
    re.IGNORECASE,
)
SYMBOL = re.compile(r"`[^`]{2,}`|[a-z]+_[a-z_]+|[A-Z][a-z]+[A-Z]\w*|\w+\.\w+")

DEEP = 24


def four_terms(cand, max_rrf: float, max_entity: float,
               channel_max: dict[str, float]) -> dict[str, float]:
    """The scoring terms, recomputed the way score_candidates blends them."""
    base = cand.rrf / (max_rrf or 1.0)
    coverage = sum(
        1 for name in INFORMATIVE_CHANNELS if name in cand.channels
    ) / len(INFORMATIVE_CHANNELS)
    strength = 0.0
    for name in INFORMATIVE_CHANNELS:
        score = cand.channel_scores.get(name)
        top = channel_max.get(name, 0.0)
        if score is not None and top > 0:
            strength = max(strength, score / top)
    entity = cand.channel_scores.get("entity", 0.0)  # placeholder, replaced
    return {
        "base": base, "coverage": coverage, "strength": strength, "entity": entity,
    }


def main() -> int:
    import logging

    logging.disable(logging.WARNING)

    from fastapi.testclient import TestClient

    from codemem.api.app import create_app

    bench = json.loads((ROOT / "eval/data/benchmark.json").read_text(encoding="utf-8"))
    claim = json.loads(
        (ROOT / "eval/data/qa_claim_tune.json").read_text(encoding="utf-8")
    )
    session_files = {
        m["session_id"]: set(m.get("files") or []) for m in bench["memories"]
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
            "request_id": f"lt:{memory['id']}",
            "user_id": memory["user_id"],
            "session_id": memory["session_id"],
            "messages": memory["messages"],
        })

    def dump_query(user_id: str, qid: str, query: str) -> dict | None:
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
        if not reranked:
            return None

        order: list[str] = []
        members: dict[str, list] = {}
        head_cand: dict[str, object] = {}
        for cand in reranked:
            memory = memories.get(cand.memory_id)
            if memory is None:
                continue
            sid = memory.session_id
            if sid not in members:
                order.append(sid)
                members[sid] = []
                head_cand[sid] = cand
            members[sid].append(memory)
        deep = order[:DEEP]
        if len(deep) < 2:
            return None

        eligible = [
            c for c in scored
            if any(name in c.channels for name in INFORMATIVE_CHANNELS)
        ]
        max_rrf = max((c.rrf for c in eligible), default=0.0) or 1.0
        max_entity = max(entity_match.values(), default=0.0)
        channel_max: dict[str, float] = {}
        for cand in eligible:
            for name in INFORMATIVE_CHANNELS:
                score = cand.channel_scores.get(name)
                if score is not None:
                    channel_max[name] = max(channel_max.get(name, 0.0), score)

        firsts = store.first_messages(user_id, deep)
        f1 = {sid: 0.0 for sid in deep}
        if embedder is not None and embedder.available:
            live = [sid for sid in deep if firsts.get(sid)]
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
            for sid in deep
        }
        rare = {
            t for t in query_terms
            if sum(1 for sid in deep if t in unions[sid]) <= 2
        }
        rare_cov = {
            sid: (len(unions[sid] & rare) / len(rare) if rare else 0.0)
            for sid in deep
        }
        query_idents = set()
        for vals in plan.entities.values():
            query_idents.update(v.lower() for v in vals if v)
        query_idents.update(t.lower() for t in SYMBOL.findall(query))

        sessions = []
        for rank, sid in enumerate(deep):
            head = head_cand[sid]
            terms = four_terms(head, max_rrf, max_entity, channel_max)
            entity_val = (
                entity_match.get(getattr(head, "memory_id", -1), 0.0)
                / (max_entity or 1.0)
            )
            cause_hits = [
                m for m in members[sid]
                if 60 <= len(m.text) <= 800 and CAUSE.search(m.text)
            ]
            idents = {
                s.lower() for m in cause_hits for s in SYMBOL.findall(m.text)
            } & query_idents
            sessions.append({
                "sid": sid,
                "features": {
                    "base": round(terms["base"], 5),
                    "coverage": round(terms["coverage"], 5),
                    "strength": round(terms["strength"], 5),
                    "entity": round(entity_val, 5),
                    "f1": round(f1[sid], 5),
                    "rare_cov": round(rare_cov[sid], 5),
                    "f3": round(
                        len(idents) / len(query_idents) if query_idents else 0.0,
                        5,
                    ),
                    "rank": rank,
                    "final": round(getattr(head, "final", 0.0), 5),
                    "n_members": len(members[sid]),
                },
            })
        return {"query_id": qid, "sessions": sessions}

    rows: list[dict] = []

    # anchor A: the benchmark's file-overlap relevant sets
    for query in bench["queries"]:
        relevant = set(query.get("relevant") or [])
        if not relevant:
            continue
        user_id = next(
            m["user_id"] for m in bench["memories"] if m["repo"] == query["repo"]
        )
        row = dump_query(user_id, f"proxy::{query['query_id']}", query["query"])
        if row is None:
            continue
        task_files = set(query.get("files") or [])
        for s in row["sessions"]:
            s["label"] = 1 if s["sid"] in relevant else 0
            s["strong"] = (
                1 if len(session_files.get(s["sid"], set()) & task_files) >= 2 else 0
            )
        row["anchor"] = "A"
        rows.append(row)
        print(f"A {row['query_id']} ({len(rows)})", flush=True)

    # anchor B: the claim instrument's answer sessions
    for question in claim["questions"]:
        user_id = next(
            m["user_id"] for m in bench["memories"] if m["repo"] == question["repo"]
        )
        row = dump_query(user_id, question["query_id"], question["question"])
        if row is None:
            continue
        for s in row["sessions"]:
            s["label"] = 1 if s["sid"] == question["answer_session"] else 0
            s["strong"] = s["label"]
        row["anchor"] = "B"
        rows.append(row)
        print(f"B {row['query_id']}", flush=True)

    client.__exit__(None, None, None)

    n_pos = sum(
        1 for r in rows for s in r["sessions"] if s["label"]
    )
    n_hit = sum(
        1 for r in rows if any(s["label"] for s in r["sessions"])
    )
    print(f"\nrows {len(rows)}; positive (query, session) pairs {n_pos}; "
          f"queries with a positive in the deep list {n_hit}/{len(rows)}")

    out = ROOT / "eval/results/ltr_data.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, ensure_ascii=False, indent=1, default=str)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
