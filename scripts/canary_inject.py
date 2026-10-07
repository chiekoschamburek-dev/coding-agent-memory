"""Canary injection: can the pipeline tell *which* session recorded a claim?

The reachability work shows the ranking machinery is not broken: query with the gold
claim's own text and the answer session reaches menu@8 on 95.7 % of the claim anchor
(`eval/results/reachability_clean.txt`). But that arm leaks the label - it puts the
answer into the query. The open question is the other half: the shipped pipeline ranks
sessions by how much they resemble the question, so does it have any notion of
**provenance** - that this claim was recorded by session A rather than merely present in
session B?

Inject the answer's own sentence into a session the label says is a distractor, then ask
the same question with the real (non-leaking) query, and watch two things:

  self      the injected claim is THIS question's gold sentence, placed in a distractor
  impact    does it enter the menu / payload, and does the true gold session move down

If the injected copy outranks or displaces the genuine recording, the ranking is
claim-surface driven and "gold" is a label distinction, not something the features can
see. If it does not, the pipeline does carry evidence beyond surface match, and the
"gold loses to noise" finding is a retrieval problem after all.

Injections accumulate across queries (the store is shared per user), which adds real
distractor text and biases the test AGAINST finding an effect - conservative, and stated
here rather than hidden. Search only, no relay, one corpus build.

Run:
    PYTHONPATH=src python scripts/canary_inject.py --limit 25 --out eval/results/canary.json
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

_WS = re.compile(r"\s+")
norm = lambda t: _WS.sub(" ", (t or "")).strip().lower()  # noqa: E731


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", type=Path, default=Path("eval/data/qa_claim_tune.json"))
    ap.add_argument("--data", type=Path, default=Path("eval/data/benchmark.json"))
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    qa = json.loads(args.qa.read_text(encoding="utf-8"))["questions"][: args.limit]
    bench = json.loads(args.data.read_text(encoding="utf-8"))

    settings = Settings(data_dir=Path(tempfile.mkdtemp()))
    container = Container(settings)
    rows: list[dict] = []
    try:
        for memory in bench["memories"]:
            container.add.handle(
                request_id=f"cy:{memory['id']}",
                user_id=memory["user_id"],
                session_id=memory["session_id"],
                messages=[Message(**m) for m in memory["messages"]],
            )
        print(f"corpus built ({len(bench['memories'])} sessions)", file=sys.stderr)
        by_user: dict[str, list[str]] = {}
        for m in bench["memories"]:
            by_user.setdefault(m["user_id"], []).append(m["session_id"])

        def seats(user_id: str, question: str, options: list[str]) -> tuple[list[str], int]:
            """Session order and payload size for one search, as the service emits it."""
            items = container.search.handle(
                user_id=user_id, query=question, options=options, top_k=args.top_k
            )
            order: list[str] = []
            for it in items:
                with container.store._read() as conn:  # noqa: SLF001 - measurement only
                    r = conn.execute(
                        "SELECT session_id FROM memory WHERE id = ?",
                        (it.memory_id,),
                    ).fetchone()
                if r and r["session_id"] not in order:
                    order.append(r["session_id"])
            return order, len(items)

        for q in qa:
            user_id = f"bench:{q['repo']}"
            gold_sess = q.get("answer_session")
            distractors = [s for s in (q.get("distractor_sessions") or []) if s != gold_sess]
            if not gold_sess or not distractors:
                continue
            target = distractors[0]
            before_order, before_items = seats(user_id, q["question"], q.get("options") or [])
            gold_rank_before = (
                before_order.index(gold_sess) + 1 if gold_sess in before_order else None
            )

            # Inject the answer's own sentence into a session the label calls a distractor.
            container.add.handle(
                request_id=f"canary:{q['query_id']}",
                user_id=user_id,
                session_id=target,
                messages=[Message(role="assistant", content=q["gold_claim"])],
            )
            after_order, after_items = seats(user_id, q["question"], q.get("options") or [])
            gold_rank_after = (
                after_order.index(gold_sess) + 1 if gold_sess in after_order else None
            )
            injected_seated = target in after_order
            rows.append(
                {
                    "query_id": q["query_id"],
                    "gold_session": gold_sess,
                    "injected_session": target,
                    "gold_rank_before": gold_rank_before,
                    "gold_rank_after": gold_rank_after,
                    "injected_seated": injected_seated,
                    "injected_beats_gold": bool(
                        injected_seated
                        and gold_rank_after is not None
                        and after_order.index(target) < after_order.index(gold_sess)
                    ),
                    "gold_lost_seat": bool(gold_rank_before) and gold_rank_after is None,
                    "items_before": before_items,
                    "items_after": after_items,
                }
            )
            if len(rows) % 5 == 0:
                print(f"  {len(rows)}/{len(qa)}", file=sys.stderr)
    finally:
        container.close()

    n = len(rows)
    print(f"\nqueries with an injection: {n}")
    seated = sum(1 for r in rows if r["injected_seated"])
    beats = sum(1 for r in rows if r["injected_beats_gold"])
    lost = sum(1 for r in rows if r["gold_lost_seat"])
    print(f"  injected claim seated in the payload : {seated}/{n} = {seated / n:.1%}")
    print(f"  injected claim outranks the true gold : {beats}/{n} = {beats / n:.1%}")
    print(f"  true gold knocked out of the payload  : {lost}/{n} = {lost / n:.1%}")
    moved = [
        (r["gold_rank_before"], r["gold_rank_after"])
        for r in rows
        if r["gold_rank_before"] and r["gold_rank_after"]
    ]
    if moved:
        worse = sum(1 for b, a in moved if a > b)
        same = sum(1 for b, a in moved if a == b)
        better = sum(1 for b, a in moved if a < b)
        print(f"  true gold's position: worse {worse}, unchanged {same}, better {better}"
              f" (of {len(moved)} where it held a seat in both)")
    print(
        "\nReading: a high seated/outrank rate means ranking is claim-surface driven and\n"
        "the gold/noise distinction lives only in the label. A near-zero rate means the\n"
        "pipeline carries evidence that is not the sentence itself, so 'gold loses to\n"
        "noise' is a retrieval finding, not a label artefact."
    )
    if args.out:
        args.out.write_text(json.dumps({"rows": rows}, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
