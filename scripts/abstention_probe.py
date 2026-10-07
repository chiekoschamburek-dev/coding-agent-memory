"""Can an ABSOLUTE evidence quantity predict that we do not have the answer?

Why: measured on the claim-tune anchor today (`eval/results/e2eCT_ms2.json`), when the
answer session is absent from the payload the model scores **0.175**, which is BELOW the
recorded no-memory floor of 0.243. So on those questions our payload is worse than
returning nothing, and 40/70 questions sit in that regime. Perfectly detecting them and
withholding the payload is worth 40/70 x (0.243 - 0.175) = +0.039 accuracy.

Why this is not another re-weighting: the shipped noise gate reads a score that has been
divided by the pool maximum (`evidence.py:268-271`), so the head is always 1.0 and the gate
can never abstain on the first session. The quantities discarded by that division - the IDF
identifier sum, the BM25 magnitude, the dense cosine, the cross-encoder logit - are absolute.
This probe asks whether ANY of them predicts reach before spending an answer call on it.

Search only: no relay, no answer model. Costs one corpus build.

Run:
    PYTHONPATH=src python scripts/abstention_probe.py --out eval/results/abstention_probe.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")
logging.disable(logging.WARNING)

from codemem.add.entities import _norm_path  # noqa: E402
from codemem.api.app import Container  # noqa: E402
from codemem.core.config import Settings  # noqa: E402
from codemem.core.schemas import Message  # noqa: E402
from codemem.search.evidence import score_candidates  # noqa: E402
from codemem.search.query import plan_query  # noqa: E402

STATS = ("entity_top", "lex_top", "dense_top", "rrf_top", "rerank_logit", "rerank_prob")


def auc(values: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney probability that a positive scores above a negative."""
    pos = [(v, i) for i, (v, l) in enumerate(zip(values, labels)) if l]
    neg = [(v, i) for i, (v, l) in enumerate(zip(values, labels)) if not l]
    if not pos or not neg:
        return None
    wins = sum(1 for p in pos for q in neg if p[0] > q[0])
    ties = sum(1 for p in pos for q in neg if p[0] == q[0])
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", type=Path, default=Path("eval/data/qa_claim_tune.json"))
    ap.add_argument("--data", type=Path, default=Path("eval/data/benchmark.json"))
    ap.add_argument("--recorded", type=Path, default=Path("eval/results/e2eCT_ms2.json"))
    ap.add_argument("--floor", type=float, default=0.243, help="recorded no-memory accuracy")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    qa = json.loads(args.qa.read_text(encoding="utf-8"))["questions"]
    bench = json.loads(args.data.read_text(encoding="utf-8"))
    rec = json.loads(args.recorded.read_text(encoding="utf-8"))
    outcomes = {o["query_id"]: o for o in rec["outcomes"]["with_memory"]}

    settings = Settings(data_dir=Path(tempfile.mkdtemp()))
    container = Container(settings)
    rows: list[dict] = []
    try:
        for memory in bench["memories"]:
            container.add.handle(
                request_id=f"ap:{memory['id']}",
                user_id=memory["user_id"],
                session_id=memory["session_id"],
                messages=[Message(**m) for m in memory["messages"]],
            )
        print(f"corpus built ({len(bench['memories'])} sessions)", file=sys.stderr)

        store, search = container.store, container.search
        for q in qa:
            user_id = f"bench:{q['repo']}"
            plan = plan_query(q["question"], q.get("options") or [])
            cands = search.retriever.recall(user_id, plan)
            if not cands:
                rows.append({"query_id": q["query_id"], "empty": True})
                continue
            memories = search.retriever.load(user_id, cands)
            chunk_scores = store.entity_match_scores(user_id, plan.entities)
            ent_by_mem: dict[int, float] = {}
            for c in cands:
                m = memories.get(c.memory_id)
                if m is None or m.chunk_id is None:
                    continue
                s = chunk_scores.get(m.chunk_id, 0.0)
                if s > 0:
                    ent_by_mem[m.id] = s
            scored = score_candidates(cands, memories, plan, ent_by_mem)
            if not scored:
                rows.append({"query_id": q["query_id"], "empty": True})
                continue
            head = scored[0]
            head_mem = memories.get(head.memory_id)
            rerank_logit = rerank_prob = None
            if container.reranker is not None and head_mem is not None:
                text = f"{head_mem.structural_kind}: {search._rerank_body(head_mem, plan)}"
                raw = container.reranker.score(plan.query, [text])
                if raw:
                    rerank_logit = round(float(raw[0]), 4)
                    from codemem.rerank import sigmoid

                    rerank_prob = round(
                        sigmoid(rerank_logit / settings.rerank_temperature), 4
                    )
            emitted = None
            try:
                emitted = search.handle(
                    user_id=user_id,
                    query=q["question"],
                    options=q.get("options") or [],
                    top_k=100,
                )
            except Exception:  # pragma: no cover - measurement only
                emitted = []
            shown_sessions = set()
            answer_session = q.get("answer_session")
            # Reconstruct which sessions the payload drew from by matching content
            # back to memories (the item carries memory_id).
            for it in emitted or []:
                m = memories.get(it.memory_id)
                if m is not None:
                    shown_sessions.add(m.session_id)
            pool_sessions = {m.session_id for m in memories.values() if m.session_id}
            rows.append(
                {
                    "query_id": q["query_id"],
                    "empty": False,
                    "entity_top": round(max(ent_by_mem.values(), default=0.0), 4),
                    "lex_top": round(head.channel_scores.get("lexical", 0.0), 4),
                    "dense_top": round(
                        max(
                            (c.channel_scores.get("dense", 0.0) for c in cands),
                            default=0.0,
                        ),
                        4,
                    ),
                    "rrf_top": round(head.rrf, 6),
                    "rerank_logit": rerank_logit,
                    "rerank_prob": rerank_prob,
                    "n_admitted": len(scored),
                    "pool_sessions": len(pool_sessions),
                    "emitted_sessions": len(shown_sessions),
                    "emitted_items": len(emitted or []),
                    "tokens": sum(getattr(i, "tokens", 0) for i in (emitted or [])),
                    "probe_reach": bool(
                        answer_session in shown_sessions
                    ),
                    "answer_session": answer_session,
                    "user_id": user_id,
                }
            )
            if len(rows) % 10 == 0:
                print(f"  {len(rows)}/{len(qa)}", file=sys.stderr)
    finally:
        container.close()

    rows = [r for r in rows if not r.get("empty")]
    for r in rows:
        o = outcomes.get(r["query_id"])
        if o is None:
            continue
        r["recorded_shown"] = bool(o["answer_session_shown"])
        r["correct"] = bool(o["correct"])
    joined = [r for r in rows if "recorded_shown" in r]
    print(f"\njoined {len(joined)}/{len(rows)} queries against {args.recorded.name}")

    agree = sum(1 for r in joined if r["probe_reach"] == r["recorded_shown"])/max(1, len(joined))
    print(f"probe reach vs recorded reach agreement: {agree:.3f}")

    base_acc = sum(r["correct"] for r in joined) / max(1, len(joined))
    print(f"recorded accuracy: {base_acc:.4f}   floor: {args.floor}")
    print(f"\n{'statistic':<14}{'AUC|reach':>10}{'AUC|correct':>12}   best expected acc (drop k)")
    out: dict[str, object] = {"base_acc": round(base_acc, 4), "floor": args.floor, "rows": joined}
    for name in STATS:
        vals = [r.get(name) for r in joined]
        if any(v is None for v in vals):
            print(f"{name:<14}  (missing values)")
            continue
        a_reach = auc(vals, [1 if r["recorded_shown"] else 0 for r in joined])
        a_corr = auc(vals, [1 if r["correct"] else 0 for r in joined])
        # abstention curve: drop the LOWEST-evidence queries (they are the ones we
        # supposedly cannot answer), kept ones keep their recorded outcome, dropped
        # ones fall back to the no-memory floor.
        # Abstention curve: drop the LOWEST-evidence queries (the ones we supposedly
        # cannot answer); kept queries keep their recorded outcome, dropped ones fall
        # back to the no-memory floor.
        #
        # The k that maximises this is chosen AFTER seeing the labels, so the maximum
        # is optimistically biased - with 6 statistics over the same 70 questions it
        # is a selection effect unless reported alongside fixed-k values. So: report
        # pre-specified drop fractions, and label the maximum as an upper bound.
        order = sorted(range(len(joined)), key=lambda i: vals[i])

        def expected(drop: int) -> float:
            kept = order[drop:]
            if not kept:
                return args.floor
            acc_kept = sum(joined[i]["correct"] for i in kept) / len(kept)
            return acc_kept * (len(kept) / len(joined)) + args.floor * (drop / len(joined))

        fixed = {f"{int(p * 100)}%": round(expected(int(round(p * len(joined)))), 4)
                 for p in (0.10, 0.20, 0.30)}
        best = max(((expected(d), d) for d in range(len(joined) + 1)))
        print(
            f"{name:<14}{(a_reach if a_reach is not None else 0):>10.3f}"
            f"{(a_corr if a_corr is not None else 0):>12.3f}"
            f"   fixed {fixed}   max(upper bd) {best[0]:.4f} @drop {best[1]}"
        )
        out[name] = {
            "auc_reach": a_reach, "auc_correct": a_corr,
            "expected_acc_fixed_k": fixed,
            "expected_acc_max_upper_bound": round(best[0], 4), "drop": best[1],
        }
    if args.out:
        args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
