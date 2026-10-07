"""Where do the claim-anchor failures actually live?

The campaign has been aiming at ranking, but three different layers are being
conflated, and they need different machinery:

  A. the answer SESSION never reaches the payload        -> admission / recall
  B. the session is there, the decisive CONTENT is not   -> pool entry + intra-session
  C. the content is there and the model still answers wrong -> nothing retrievable

Two published numbers sit in tension here: on this anchor the shipped ordering seats
the answer session in the *top-8 menu* for 97.1 % of questions (`eval/README.md`, the
F1-selection section), while only 42.9 % of questions get it into the *payload*
(`e2eCT_ms2.json`). So the session-level bottleneck is downstream of the menu, i.e.
assembly - which is already measured as a reach <-> conversion dial with no free arm.
Whether an ADMISSION-side channel (direction 4, `entity_timeline`) can help at all
depends on how many failures are class A versus class B versus class C, and that split
has never been measured with exact ground truth.

It can be measured exactly here, because `qa_claim_tune.json` carries `gold_claim`: the
literal corpus sentence the correct option quotes. That is a leak-free label - it is
never used as a query, only to locate which memory holds it.

Search only: no relay. Costs one corpus build.

Run:
    PYTHONPATH=src python scripts/claim_bottleneck.py --out eval/results/claim_bottleneck.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")
logging.disable(logging.WARNING)

from codemem.api.app import Container  # noqa: E402
from codemem.core.config import Settings  # noqa: E402
from codemem.core.schemas import Message  # noqa: E402
from codemem.search.evidence import score_candidates  # noqa: E402
from codemem.search.query import plan_query  # noqa: E402

_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return _WS.sub(" ", (text or "")).strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", type=Path, default=Path("eval/data/qa_claim_tune.json"))
    ap.add_argument("--data", type=Path, default=Path("eval/data/benchmark.json"))
    ap.add_argument("--recorded", type=Path, default=Path("eval/results/e2eCT_ms2.json"))
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--max-sessions", type=int, default=None,
                    help="override evidence_max_sessions (shipped default 2)")
    ap.add_argument("--full-count", type=int, default=None,
                    help="override evidence_full_count (shipped default 8)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    qa = json.loads(args.qa.read_text(encoding="utf-8"))["questions"]
    bench = json.loads(args.data.read_text(encoding="utf-8"))
    outcomes = {
        o["query_id"]: o
        for o in json.loads(args.recorded.read_text(encoding="utf-8"))["outcomes"]["with_memory"]
    }

    settings = Settings(data_dir=Path(tempfile.mkdtemp()))
    if args.max_sessions is not None:
        settings.evidence_max_sessions = args.max_sessions
    if args.full_count is not None:
        settings.evidence_full_count = args.full_count
    full_count = settings.evidence_full_count
    container = Container(settings)
    rows: list[dict] = []
    try:
        for memory in bench["memories"]:
            container.add.handle(
                request_id=f"cb:{memory['id']}",
                user_id=memory["user_id"],
                session_id=memory["session_id"],
                messages=[Message(**m) for m in memory["messages"]],
            )
        print(f"corpus built ({len(bench['memories'])} sessions)", file=sys.stderr)

        store, search = container.store, container.search
        sessions_by_user: dict[tuple[str, str], list[str]] = {
            (m["user_id"], m["session_id"]): [norm(x["content"]) for x in m["messages"]]
            for m in bench["memories"]
        }
        for q in qa:
            user_id = f"bench:{q['repo']}"
            # Locate the decisive sentence at MESSAGE level, not chunk level:
            # `gold_claim` is verbatim trajectory text (verified 70/70 against
            # benchmark.json), but one assistant message is segmented into several
            # chunks, so requiring the whole claim to sit inside a single chunk fails
            # for most questions and would silently report it as absent everywhere.
            # Memories are mapped back to their source message through chunk.msg_index.
            gold = norm(q["gold_claim"])
            answer_session = q.get("answer_session")
            gold_idx = [
                i
                for i, text in enumerate(
                    sessions_by_user.get((user_id, answer_session), [])
                )
                if gold in text
            ]
            hits = _memories_for_messages(store, user_id, answer_session, gold_idx)

            plan = plan_query(q["question"], q.get("options") or [])
            cands = search.retriever.recall(user_id, plan)
            memories = search.retriever.load(user_id, cands) if cands else {}
            pool_ids = {c.memory_id for c in cands}
            scored = score_candidates(
                cands, memories, plan, _entity_map(store, user_id, plan, cands, memories)
            )
            admitted_ids = {c.memory_id for c in scored}
            items = search.handle(
                user_id=user_id,
                query=q["question"],
                options=q.get("options") or [],
                top_k=args.top_k,
            )
            payload_ids = {it.memory_id for it in items}
            # Substantive reach: a session counts as "shown" in the e2e harness if any
            # of its items is in the window, whether that item is a full span or a
            # 110-token pointer. Position decides the form by construction
            # (`full_form = len(items) < evidence_full_count`), so record where the
            # decisive item actually landed and how much of it survived.
            hit_ids = set(hits)
            gold_positions = [
                i + 1 for i, it in enumerate(items) if it.memory_id in hit_ids
            ]
            gold_index = min(gold_positions, default=None)
            gold_tokens = next(
                (it.tokens for it in items if it.memory_id in hit_ids), None
            )

            rec = outcomes.get(q["query_id"], {})
            rows.append(
                {
                    "query_id": q["query_id"],
                    "answer_session": answer_session,
                    "n_gold_messages": len(gold_idx),
                    "n_gold_memories": len(hits),
                    "gold_in_pool": bool(set(hits) & pool_ids),
                    "gold_admitted": bool(set(hits) & admitted_ids),
                    "gold_in_payload": bool(set(hits) & payload_ids),
                    "gold_payload_index": gold_index,
                    "gold_item_tokens": gold_tokens,
                    "gold_in_payload_full": gold_index is not None
                    and gold_index <= full_count,
                    "recorded_shown": bool(rec.get("answer_session_shown")),
                    "correct": bool(rec.get("correct")),
                    "n_items": len(items),
                }
            )
            if len(rows) % 10 == 0:
                print(f"  {len(rows)}/{len(qa)}", file=sys.stderr)
    finally:
        container.close()

    n = len(rows)
    no_label = sum(1 for r in rows if r["n_gold_memories"] == 0)
    print(f"\nquestions {n}; decisive sentence located in 0 memories for {no_label}")
    scored_rows = [r for r in rows if r["n_gold_memories"] > 0]

    def rate(pred):
        return sum(1 for r in scored_rows if pred(r)) / max(1, len(scored_rows))

    print(f"\ngold sentence reach, over the {len(scored_rows)} questions where it was located:")
    print(f"  in recall pool       {rate(lambda r: r['gold_in_pool']):.3f}")
    print(f"  admitted for scoring {rate(lambda r: r['gold_admitted']):.3f}")
    print(f"  in payload           {rate(lambda r: r['gold_in_payload']):.3f}")
    print(
        f"  in payload, FULL form (index <= {full_count})"
        f" {rate(lambda r: (r['gold_in_payload'] and (r['gold_payload_index'] or 999) <= full_count)):.3f}"
    )
    print(f"  answer session shown (e2e definition) {rate(lambda r: r['recorded_shown']):.3f}")
    idx = [r["gold_payload_index"] for r in scored_rows if r["gold_payload_index"]]
    if idx:
        import statistics as _st

        print(f"  decisive item lands at mean index {_st.mean(idx):.1f} (median {int(_st.median(idx))})")

    print("\nfailure classes (the question that decides direction 4):")
    tot = len(scored_rows)
    if not tot:
        print("  no question had its decisive sentence located - the label path is broken,"
              " not the retrieval")
        return 1
    shown = [r for r in scored_rows if r["recorded_shown"]]
    absent = [r for r in scored_rows if not r["recorded_shown"]]
    cls_c = [r for r in shown if r["gold_in_payload"] and not r["correct"]]
    cls_b = [r for r in shown if not r["gold_in_payload"] and not r["correct"]]
    ok = [r for r in scored_rows if r["correct"]]
    a_side = [r for r in absent if not r["gold_in_pool"]]
    print(f"  answered correctly                     {len(ok):>3}/{tot}  {len(ok)/tot:.1%}")
    print(f"  C  shown + decisive text present, wrong {len(cls_c):>3}  {len(cls_c)/tot:.1%}  -> not retrievable")
    print(f"  B  shown, decisive text absent, wrong   {len(cls_b):>3}  {len(cls_b)/tot:.1%}  -> intra-session / pool")
    print(f"  A  session absent AND gold never pooled {len(a_side):>3}  {len(a_side)/tot:.1%}  -> admission")
    print(f"  session absent, gold WAS pooled         {len(absent)-len(a_side):>3}  {(len(absent)-len(a_side))/tot:.1%}  -> assembly")
    print(
        f"\nof the {len(absent)} questions where the answer session is not shown: "
        f"{len(a_side)} have no gold sentence in the pool at all (admission could act), "
        f"{len(absent)-len(a_side)} have it in the pool and the assembler still dropped it."
    )
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "settings": {
                        "evidence_max_sessions": settings.evidence_max_sessions,
                        "max_evidence_per_session": settings.max_evidence_per_session,
                        "evidence_full_count": settings.evidence_full_count,
                        "evidence_item_tokens": settings.evidence_item_tokens,
                        "evidence_ptr_tokens": settings.evidence_ptr_tokens,
                    },
                    "rows": rows,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


def _memories_for_messages(
    store, user_id: str, session_id: str | None, msg_indexes: list[int]
) -> list[int]:
    """Memory ids derived from the given source messages of one session."""
    if not msg_indexes or session_id is None:
        return []
    with store._read() as conn:  # noqa: SLF001 - measurement only
        return [
            int(r["id"])
            for r in conn.execute(
                "SELECT m.id AS id FROM memory m JOIN chunk c ON m.chunk_id = c.id"
                " WHERE m.user_id = ? AND m.session_id = ?"
                f" AND c.msg_index IN ({','.join('?' * len(msg_indexes))})",
                (user_id, session_id, *msg_indexes),
            )
        ]


def _entity_map(store, user_id, plan, cands, memories) -> dict[int, float]:
    chunk_scores = store.entity_match_scores(user_id, plan.entities)
    out: dict[int, float] = {}
    if not chunk_scores:
        return out
    for cand in cands:
        memory = memories.get(cand.memory_id)
        if memory is None or memory.chunk_id is None:
            continue
        score = chunk_scores.get(memory.chunk_id, 0.0)
        if score > 0:
            out[memory.id] = score
    return out


if __name__ == "__main__":
    raise SystemExit(main())
