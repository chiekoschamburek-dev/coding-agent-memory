"""Offline session-feature replay: can free structural signals fix the ranking?

The attribution work left one measured gap: with the session budget at 2,
ideal ordering reaches recall 0.694 against 0.475 actual — and four
mechanisms that tried to close it (representative rerank, listwise judge,
cards, operative term) all failed. This script tests four session-level
features that were never scoring signals, entirely offline (one Add, one
embed pass, no answer model):

  F1 first_msg_cos   cosine(query, the session's FIRST message) — the issue
                     statement lives at the trajectory's head, and procedure
                     queries are written like problem statements, not diffs.
                     The free version of the HyDE card.
  F2 union_coverage  how much of the query's entity set and of its rare words
                     the session's pooled chunks cover *as a union* — invisible
                     to per-chunk max.
  F3 cause_x_ident   assistant prose that records a CAUSE (the qa builder's
                     lexicon) and names a query identifier — distractor
                     sessions record actions, not causes.
  F4 supersedes_out  how many other sessions' same-file chunks this session
                     postdates — the causal chain's tail (the fix that stuck)
                     has high out-degree. (The supersedes table is unpopulated
                     in this corpus; this is the timestamp-order proxy.)

Acceptance (per the plan): flipped pairs among the losing (query, relevant
session) pairs, fixes among the queries whose payload holds no relevant
session, and — continuously — top-2 session recall, validated against the
measured 0.4746 by replaying the shipped estimator first.

Usage::

    PYTHONPATH=src python scripts/session_features.py
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from codemem.core.config import Settings  # noqa: E402
from codemem.search.evidence import INFORMATIVE_CHANNELS, score_candidates  # noqa: E402
from codemem.search.query import plan_query  # noqa: E402

CAUSE = re.compile(
    r"\b(because|caused by|due to|root cause|the problem is|the issue is|"
    r"fails? when|which (?:causes?|fails?|means)|turns out|the fix (?:was|is)|"
    r"instead of|so that|in order to|otherwise|needed to|make sure)\\b",
    re.IGNORECASE,
)
SYMBOL = re.compile(r"`[^`]{2,}`|[a-z]+_[a-z_]+|[A-Z][a-z]+[A-Z]\w*|\w+\.\w+")


def content_terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_\.]+", text or "") if len(t) >= 4}


def main() -> int:
    import logging

    logging.disable(logging.WARNING)

    from fastapi.testclient import TestClient

    from codemem.api.app import create_app

    bench = json.loads((ROOT / "eval/data/benchmark.json").read_text(encoding="utf-8"))
    queries = [q for q in bench["queries"] if q["relevant"]]

    settings = Settings(data_dir=Path(tempfile.mkdtemp()))
    app = create_app(settings)
    client = TestClient(app)
    client.__enter__()  # runs the lifespan, which builds the container
    container = app.state.container

    if True:  # the replay body stays flat; the client is closed below
        for memory in bench["memories"]:
            client.post(
                "/add",
                json={
                    "request_id": f"sf:{memory['id']}",
                    "user_id": memory["user_id"],
                    "session_id": memory["session_id"],
                    "messages": memory["messages"],
                },
            )
        store = container.store
        embedder = container.embedder
        pipeline = container.search

        # Session profile: first message text, per-session file set and ts,
        # used by F1/F3/F4.
        session_first: dict[str, str] = {}
        session_files: dict[str, set[str]] = {}
        session_ts: dict[str, float] = {}
        session_cause_prose: dict[str, list[str]] = {}
        repo_of_session: dict[str, str] = {}
        for memory in bench["memories"]:
            sid = memory["session_id"]
            repo_of_session[sid] = memory["repo"]
            with store._read() as conn:  # noqa: SLF001 - replay harness
                first = conn.execute(
                    "SELECT content FROM raw_message WHERE user_id=? AND session_id=?"
                    " ORDER BY msg_index LIMIT 1",
                    (memory["user_id"], sid),
                ).fetchone()
                msgs = conn.execute(
                    "SELECT role, content, ts FROM raw_message WHERE user_id=? AND session_id=?"
                    " ORDER BY msg_index",
                    (memory["user_id"], sid),
                ).fetchall()
            if first:
                session_first[sid] = first["content"]
            session_files[sid] = {
                v for (v,) in conn_files(store, memory["user_id"], sid)
            }
            ts_list = [r["ts"] for r in msgs if r["ts"] is not None]
            session_ts[sid] = min(ts_list) if ts_list else 0.0
            session_cause_prose[sid] = [
                r["content"]
                for r in msgs
                if r["role"] == "assistant"
                and 80 <= len(r["content"] or "") <= 700
                and CAUSE.search(r["content"] or "")
                and SYMBOL.search(r["content"] or "")
            ]

        # F1: embed every session's first message once.
        first_ids = sorted(session_first)
        first_vecs = embedder.embed([session_first[s] for s in first_ids]) or []
        first_vec = dict(zip(first_ids, first_vecs))

        # F4 proxy: out-degree over (session, file) pairs by timestamp.
        file_sessions: dict[str, list[tuple[str, float]]] = {}
        for sid, files in session_files.items():
            for f in files:
                file_sessions.setdefault(f, []).append((sid, session_ts[sid]))
        supersedes_out: dict[str, int] = {sid: 0 for sid in session_files}
        for f, lst in file_sessions.items():
            for sid, ts in lst:
                supersedes_out[sid] += sum(1 for s2, t2 in lst if t2 < ts)

        rows = []
        baseline_covered = total_pairs = 0
        zero_payload_queries: list[str] = []
        for query in queries:
            user_id = next(
                m["user_id"] for m in bench["memories"] if m["repo"] == query["repo"]
            )
            relevant = list(query["relevant"])
            total_pairs += len(relevant)

            plan = plan_query(query["query"], None)
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

            # Shipped session order: first appearance of each session's head.
            order: list[str] = []
            head_final: dict[str, float] = {}
            pooled: dict[str, list] = {}
            for cand in reranked:
                memory = memories.get(cand.memory_id)
                sid = memory.session_id if memory else None
                if not sid:
                    continue
                if sid not in head_final:
                    order.append(sid)
                    head_final[sid] = cand.final
                pooled.setdefault(sid, []).append(cand)

            # F2: union coverage of the query's terms by pooled chunk texts.
            query_terms = content_terms(query["query"])
            plan_entities = {v for vs in plan.entities.values() for v in vs}
            union_terms: dict[str, set[str]] = {}
            for sid, members in pooled.items():
                texts = []
                for cand in members:
                    memory = memories.get(cand.memory_id)
                    if memory:
                        texts.append(memory.text)
                union_terms[sid] = set().union(*(content_terms(t) for t in texts)) if texts else set()
            ent_cov = {
                sid: (len(union_terms[sid] & plan_entities) / len(plan_entities) if plan_entities else 0.0)
                for sid in order
            }
            rare = {
                t for t in query_terms
                if sum(1 for sid in union_terms if t in union_terms[sid]) <= 2
            }
            rare_cov = {
                sid: (len(union_terms[sid] & rare) / len(rare) if rare else 0.0)
                for sid in order
            }

            # F1: query embedding vs each candidate session's first message.
            qvec = (embedder.embed([query["query"]]) or [None])[0]
            f1 = {}
            for sid in order:
                sv = first_vec.get(sid)
                if qvec and sv and len(qvec) == len(sv):
                    f1[sid] = sum(a * b for a, b in zip(qvec, sv))
                else:
                    f1[sid] = 0.0

            # F3: cause prose naming a query identifier.
            query_idents = plan_entities | query_terms
            f3 = {}
            for sid in order:
                f3[sid] = sum(
                    1 for text in session_cause_prose.get(sid, [])
                    if query_idents & content_terms(text)
                )

            top2 = order[:2]
            covered = sum(1 for sid in relevant if sid in top2)
            baseline_covered += covered
            if covered == 0:
                zero_payload_queries.append(query["instance_id"])

            rows.append({
                "query_id": query["instance_id"],
                "relevant": relevant,
                "order": order,
                "top2": top2,
                "covered": covered,
                "n_relevant": len(relevant),
                "head_final": head_final,
                "f1": f1, "ent_cov": ent_cov, "rare_cov": rare_cov,
                "f3": f3, "f4": {sid: supersedes_out.get(sid, 0) for sid in order},
                # the live Candidate objects, so the evaluation can re-run the
                # real assembler under re-ordered session blocks
                "reranked": list(reranked),
                "memories": memories,
                "plan": plan,
                "user_id": user_id,
            })

    # ---- validation + evaluation ------------------------------------------
    # The shipped payload is NOT "the first two sessions in rank order": the
    # noise gate skips weak-head sessions and later ones take the slot, which
    # is why the naive first-two replay reads 0.276 against the measured
    # 0.4746. Every variant below therefore re-runs the real assembler under
    # its own session ordering (stable block sort of the reranked list, so the
    # first-appearance order assemble sees IS the feature's ordering).
    from codemem.search.evidence import assemble as assemble_ship

    def payload_sessions(row, ordering: dict[str, float] | None) -> list[str]:
        reranked = row["reranked"]
        if ordering is not None:
            block_rank = {sid: i for i, sid in enumerate(ordering)}
            pos = {id(c): i for i, c in enumerate(reranked)}

            def block_of(cand):
                memory = row["memories"].get(cand.memory_id)
                return block_rank.get(memory.session_id, 99) if memory else 99

            reranked = sorted(reranked, key=lambda c: (block_of(c), pos[id(c)]))
        items = assemble_ship(
            settings, store, row["user_id"], row["plan"],
            list(reranked), row["memories"], top_k=100,
        )
        seen: list[str] = []
        for item in items:
            memory = row["memories"].get(item.memory_id)
            if memory and memory.session_id not in seen:
                seen.append(memory.session_id)
        return seen

    base_sessions = {r["query_id"]: payload_sessions(r, None) for r in rows}
    # two aggregations: pair-micro (the funnel's 27.6 %) and query-macro (what
    # run_benchmark's recall@10 reports — per-query recall averaged over the
    # 89 queries). The measured 0.4746 is the query-macro number.
    baseline_micro = sum(
        sum(1 for sid in r["relevant"] if sid in base_sessions[r["query_id"]])
        for r in rows
    ) / total_pairs
    baseline_macro = sum(
        sum(1 for sid in r["relevant"] if sid in base_sessions[r["query_id"]])
        / len(r["relevant"])
        for r in rows
    ) / len(rows)
    print(f"replay validation: baseline payload recall micro={baseline_micro:.4f} "
          f"macro={baseline_macro:.4f} (measured recall@10 = 0.4746)")
    if abs(baseline_macro - 0.4746) > 0.02:
        print("  WARNING: replay does not match the measured pipeline!")

    losing = sum(
        sum(1 for sid in r["relevant"] if sid not in base_sessions[r["query_id"]])
        for r in rows
    )
    zero_ids = [r["query_id"] for r in rows
                if not set(r["relevant"]) & set(base_sessions[r["query_id"]])]
    print(f"losing (query, relevant session) pairs: {losing}; "
          f"zero-payload queries: {len(zero_ids)}")

    features = ("f1", "ent_cov", "rare_cov", "f3", "f4")
    print(f"\n{'feature':>10} {'net_pairs':>10} {'zero-fixed':>11} {'recall':>8}")
    for feat in features:
        net = 0
        fixed = 0
        covered = 0
        covered_sessions = {}
        for row in rows:
            vals = row[feat]
            ordering = sorted(row["order"], key=lambda s: -vals.get(s, 0.0))
            sessions = payload_sessions(row, ordering)
            covered_sessions[row["query_id"]] = sessions
            covered += sum(1 for sid in row["relevant"] if sid in sessions)
            base = base_sessions[row["query_id"]]
            for gold in row["relevant"]:
                if gold in base and gold not in sessions:
                    net -= 1  # a baseline win lost by the feature
                if gold not in base and gold in sessions:
                    net += 1  # a baseline loss won by the feature
            if not set(row["relevant"]) & set(base) and set(row["relevant"]) & set(sessions):
                fixed += 1
        macro = sum(
            sum(1 for sid in r["relevant"] if sid in covered_sessions[r["query_id"]])
            / len(r["relevant"])
            for r in rows
        ) / len(rows)
        print(f"{feat:>10} {net:>+10} {fixed:>11} micro={covered / total_pairs:.4f} macro={macro:.4f}")

    # rank-fusion blends: session score = head rank + w * feature rank
    print("\nrank-fusion blends (head + w*feature), payload recall:")
    for feat in features:
        line = f"{feat:>10}:"
        for w in (0.5, 1.0, 2.0):
            covered = 0
            cov_sessions = {}
            for row in rows:
                head_rank = {s: i for i, s in enumerate(row["order"])}
                vals = row[feat]
                feat_rank = {s: i for i, s in enumerate(
                    sorted(vals, key=lambda x: -vals.get(x, 0.0)))}
                score = {
                    s: 1.0 / (60 + head_rank.get(s, 99))
                    + w * 1.0 / (60 + feat_rank.get(s, 99))
                    for s in row["order"]
                }
                ordering = sorted(score, key=lambda x: -score[x])
                sessions = payload_sessions(row, ordering)
                cov_sessions[row["query_id"]] = sessions
                covered += sum(1 for sid in row["relevant"] if sid in sessions)
            macro = sum(
                sum(1 for sid in r["relevant"] if sid in cov_sessions[r["query_id"]])
                / len(r["relevant"])
                for r in rows
            ) / len(rows)
            line += f"  w={w}: micro={covered / total_pairs:.4f} macro={macro:.4f}"
        print(line)

    # stacking: head + f1 + rare_cov jointly
    print("\nstacking (head + w1*f1 + w2*rare_cov), payload recall macro:")
    for w1, w2 in ((1.0, 1.0), (1.0, 2.0), (2.0, 2.0), (1.0, 3.0), (2.0, 3.0)):
        covered = 0
        cov_sessions = {}
        for row in rows:
            head_rank = {s_: i for i, s_ in enumerate(row["order"])}
            f1v, rcv = row["f1"], row["rare_cov"]
            f1r = {s_: i for i, s_ in enumerate(sorted(f1v, key=lambda x: -f1v.get(x, 0.0)))}
            rcr = {s_: i for i, s_ in enumerate(sorted(rcv, key=lambda x: -rcv.get(x, 0.0)))}
            score = {
                s_: 1.0 / (60 + head_rank.get(s_, 99))
                + w1 * 1.0 / (60 + f1r.get(s_, 99))
                + w2 * 1.0 / (60 + rcr.get(s_, 99))
                for s_ in row["order"]
            }
            ordering = sorted(score, key=lambda x: -score[x])
            sessions = payload_sessions(row, ordering)
            cov_sessions[row["query_id"]] = sessions
            covered += sum(1 for sid in row["relevant"] if sid in sessions)
        macro = sum(
            sum(1 for sid in r["relevant"] if sid in cov_sessions[r["query_id"]])
            / len(r["relevant"])
            for r in rows
        ) / len(rows)
        print(f"  w1={w1} w2={w2}: macro={macro:.4f}")

    print(f"\nzero-payload queries (baseline): {len(zero_ids)}")
    client.__exit__(None, None, None)

    out_path = ROOT / "eval/results/session_features.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump({"rows": [
            {k: v for k, v in r.items()
             if k not in ("reranked", "memories", "plan")}
            for r in rows
        ], "baseline_recall": baseline_macro}, handle,
            ensure_ascii=False, indent=1, default=str)
    print(f"wrote {out_path}")
    return 0


def conn_files(store, user_id: str, session_id: str) -> list[tuple[str]]:
    """Distinct file_path identifiers of one session, from its entity rows."""
    with store._read() as conn:  # noqa: SLF001 - replay harness
        return conn.execute(
            "SELECT DISTINCT e.value_norm FROM chunk_entity e"
            " JOIN chunk c ON c.id = e.chunk_id"
            " JOIN memory m ON m.chunk_id = c.id"
            " WHERE e.user_id = ? AND m.session_id = ? AND e.etype = 'file_path'",
            (user_id, session_id),
        ).fetchall()


if __name__ == "__main__":
    raise SystemExit(main())
