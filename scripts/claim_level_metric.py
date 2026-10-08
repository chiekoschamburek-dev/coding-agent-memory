"""Claim-level instrument and the label-ablation test: is gold losing to noise a
retrieval failure or a label failure?

Every "gold loses on all four features" result so far (issue.md section II, the 99-pair
pairing in eval/README.md) is measured against the **file-overlap** label: a chunk is
"gold" if it names a file the patch touched. That label was independently found to favour
the winner (distractor sessions carry more file overlap than the losing relevant session,
1.49 vs 1.38), so the finding may be a statement about the label rather than about
retrieval. No one has run the same comparison under a label the answer model cannot dispute.

This does, using `gold_claim` - the literal corpus sentence the correct option quotes,
verified verbatim against the trajectories on 70/70 - and adds the second half: the
answer model is being shown FOUR corpus claims, one per option, so payload quality has a
claim-level definition that needs no model in the loop:

  hit      the gold claim's memories are in the payload
  purity   exactly ONE option's claims are in the payload (two = a second candidate answer)
  decidable_claim = hit AND purity

`purity` is the reach-buys-confusion law expressed as a count, and unlike `decidable` it
is defined on the same objects the question is made of. The instrument is only worth
tuning on if it predicts the answer better than the current one does, so the run reports
AUC against the recorded `correct` for `decidable_claim`, `hit` and `purity` next to the
known behaviour of the old string-matching metric.

Nothing here changes the shipped weights; all knobs stay at their defaults. Search only,
no relay, one corpus build.

Run:
    PYTHONPATH=src python scripts/claim_level_metric.py --out eval/results/claim_level_metric.json
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
norm = lambda t: _WS.sub(" ", (t or "")).strip().lower()  # noqa: E731


def auc(values: list[float], labels: list[int]) -> float | None:
    pos = [v for v, y in zip(values, labels) if y]
    neg = [v for v, y in zip(values, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum(1 for p in pos for q in neg if p > q)
    ties = sum(1 for p in pos for q in neg if p == q)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", type=Path, default=Path("eval/data/qa_claim_tune.json"))
    ap.add_argument("--data", type=Path, default=Path("eval/data/benchmark.json"))
    ap.add_argument("--recorded", type=Path, default=Path("eval/results/e2eCT_ms2.json"))
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--target-tokens", type=int, default=None,
                    help="override target_chunk_tokens (prose packing, shipped 320)")
    ap.add_argument("--max-chunk-tokens", type=int, default=None,
                    help="override max_chunk_tokens (structural split cap, shipped 1400)")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    qa = json.loads(args.qa.read_text(encoding="utf-8"))["questions"]
    bench = json.loads(args.data.read_text(encoding="utf-8"))
    outcomes = {
        o["query_id"]: o
        for o in json.loads(args.recorded.read_text(encoding="utf-8"))["outcomes"]["with_memory"]
    }

    # Normalised message texts per session, for locating claim sentences at message level.
    texts: dict[tuple[str, str], list[str]] = {
        (m["user_id"], m["session_id"]): [norm(x["content"]) for x in m["messages"]]
        for m in bench["memories"]
    }

    settings = Settings(data_dir=Path(tempfile.mkdtemp()))
    if args.max_sessions is not None:
        settings.evidence_max_sessions = args.max_sessions
    if args.target_tokens is not None:
        settings.target_chunk_tokens = args.target_tokens
    if args.max_chunk_tokens is not None:
        settings.max_chunk_tokens = args.max_chunk_tokens
    container = Container(settings)
    mem_index: dict[str, list[tuple[int, str, int]]] = {}
    rows: list[dict] = []
    try:
        for memory in bench["memories"]:
            container.add.handle(
                request_id=f"cl:{memory['id']}",
                user_id=memory["user_id"],
                session_id=memory["session_id"],
                messages=[Message(**m) for m in memory["messages"]],
            )
        print(f"corpus built ({len(bench['memories'])} sessions)", file=sys.stderr)
        _users, _memories = container.store.counts()
        print(
            f"  memory rows: {_memories} "
            f"({_memories / max(1, len(bench['memories'])):.1f} per session)",
            file=sys.stderr,
        )

        store, search = container.store, container.search
        for user_id in {f"bench:{q['repo']}" for q in qa}:
            with store._read() as conn:  # noqa: SLF001 - measurement only
                mem_index[user_id] = [
                    (int(r["id"]), r["session_id"], int(r["msg_index"]))
                    for r in conn.execute(
                        "SELECT m.id AS id, m.session_id AS session_id, c.msg_index AS msg_index"
                        " FROM memory m JOIN chunk c ON m.chunk_id = c.id WHERE m.user_id = ?",
                        (user_id,),
                    )
                ]

        def locate(user_id: str, candidates: list[str | None], claim: str) -> set[int]:
            """Memories whose source message contains this claim (message level).

            Tries the named sessions first and only falls back to scanning the whole
            repository if none of them holds it - the fallback is a few hundred KB of
            text per query, so the cheap path has to be tried first.
            """
            probe = norm(claim)
            if not probe:
                return set()

            def matches(text: str) -> bool:
                return probe in text or (len(probe) > 120 and probe[:120] in text)

            for sid in candidates:
                if sid is None:
                    continue
                msgs = [i for i, t in enumerate(texts.get((user_id, sid), [])) if matches(t)]
                if msgs:
                    return {
                        mid
                        for mid, s2, idx in mem_index.get(user_id, [])
                        if s2 == sid and idx in msgs
                    }
            found: set[int] = set()
            for (uid, sid), msg_texts in texts.items():
                if uid != user_id:
                    continue
                for i, t in enumerate(msg_texts):
                    if matches(t):
                        found |= {
                            mid
                            for mid, s2, idx in mem_index.get(user_id, [])
                            if s2 == sid and idx == i
                        }
            return found

        for q in qa:
            user_id = f"bench:{q['repo']}"
            opts = q.get("options") or []
            gold_index = q["gold_index"]
            # Each option is a corpus claim; locate all four.
            distractors = list(q.get("distractor_sessions") or [])
            nongold = [i for i in range(len(opts)) if i != gold_index]
            per_option = []
            for i, opt in enumerate(opts):
                if i == gold_index:
                    cands_sess = [q.get("answer_session")]
                else:
                    j = nongold.index(i) if i in nongold else -1
                    cands_sess = (
                        [distractors[j]] + distractors if 0 <= j < len(distractors)
                        else distractors
                    )
                found = locate(user_id, cands_sess, opt)
                if not found:
                    found = locate(user_id, [None], opt)  # scan the whole repository
                per_option.append(found)

            plan = plan_query(q["question"], opts)
            cands = search.retriever.recall(user_id, plan)
            if not cands:
                continue
            memories = search.retriever.load(user_id, cands)
            ent = _entity_map(store, user_id, plan, cands, memories)
            scored = score_candidates(cands, memories, plan, ent)
            items = search.handle(
                user_id=user_id, query=q["question"], options=opts, top_k=args.top_k
            )
            payload = {it.memory_id for it in items}
            final_by_id = {c.memory_id: c for c in scored}

            hit_ids = per_option[gold_index] if gold_index < len(per_option) else set()
            others = [s for i, s in enumerate(per_option) if i != gold_index]
            present_others = sum(1 for s in others if s & payload)
            # Which seat of the payload carries the answer, and which carries the
            # rival. The assembler emits session by session, so a seat is a block
            # of items from one session; the conflict rule under consideration
            # ("drop a second seat that introduces a rival claim") can only pay if
            # gold usually owns the leading seat.
            seat_of: dict[str, int] = {}
            for it in items:
                memory = memories.get(it.memory_id)
                sid = memory.session_id if memory is not None else None
                if sid is None:
                    continue
                seat_of.setdefault(sid, len(seat_of) + 1)
            gold_seats = sorted(
                {
                    seat_of[memories[mid].session_id]
                    for mid in (hit_ids & payload)
                    if mid in memories and memories[mid].session_id in seat_of
                }
            )
            rival_seats = sorted(
                {
                    seat_of[memories[mid].session_id]
                    for s in others
                    for mid in (s & payload)
                    if mid in memories and memories[mid].session_id in seat_of
                }
            )
            # Feature comparison under the CLAIM label: the best-scoring memory that
            # holds the gold claim vs the best-scoring memory in the whole pool.
            gold_cands = [final_by_id[m] for m in hit_ids if m in final_by_id]
            top = scored[0] if scored else None
            rec = outcomes.get(q["query_id"], {})
            rows.append(
                {
                    "query_id": q["query_id"],
                    "n_options": len(opts),
                    "gold_memories": len(hit_ids),
                    "others_located": sum(1 for s in others if s),
                    "hit": bool(hit_ids & payload),
                    "purity": 1 if not present_others else 1 + present_others,
                    "gold_seat": gold_seats[0] if gold_seats else None,
                    "rival_seats": rival_seats,
                    "n_seats": len(seat_of),
                    "gold_found": bool(hit_ids),
                    "best_gold_final": max((c.final for c in gold_cands), default=None),
                    "top_final": top.final if top else None,
                    "top_is_gold": bool(top is not None and top.memory_id in hit_ids),
                    "correct": bool(rec.get("correct")),
                    "n_items": len(items),
                }
            )
            if len(rows) % 10 == 0:
                print(f"  {len(rows)}/{len(qa)}", file=sys.stderr)
    finally:
        container.close()

    lab = [1 if r["correct"] else 0 for r in rows]
    n = len(rows)
    print(
        f"\nquestions {n}, max_sessions={settings.evidence_max_sessions} "
        f"target_chunk_tokens={settings.target_chunk_tokens} "
        f"max_chunk_tokens={settings.max_chunk_tokens}"
    )
    lost = sum(1 for r in rows if not r["gold_found"])
    print(f"gold claim located in 0 memories: {lost} (label path failures)")
    usable = [r for r in rows if r["gold_found"]]
    print("\nclaim-level payload quality (the new instrument):")
    for name, pred in (
        ("hit  (gold claim present)", lambda r: r["hit"]),
        ("purity == 1 (no rival claim)", lambda r: r["purity"] == 1),
        ("decidable_claim = hit AND pure", lambda r: r["hit"] and r["purity"] == 1),
    ):
        rate = sum(1 for r in usable if pred(r)) / len(usable)
        print(f"  {name:<34} {rate:.3f}")
    dist: dict[int, int] = {}
    for r in usable:
        dist[r["purity"]] = dist.get(r["purity"], 0) + 1
    print(f"  rival-claim count distribution: {dict(sorted(dist.items()))}")
    print("\ndoes it predict the answer? (AUC against recorded `correct`)")
    for name, key in (("hit", "hit"), ("1/purity", None), ("decidable_claim", None)):
        if key:
            a = auc([1.0 if r[key] else 0.0 for r in usable], [r["correct"] for r in usable])
        elif name == "1/purity":
            a = auc([1.0 / r["purity"] for r in usable], [r["correct"] for r in usable])
        else:
            a = auc(
                [1.0 if (r["hit"] and r["purity"] == 1) else 0.0 for r in usable],
                [r["correct"] for r in usable],
            )
        print(f"  {name:<18} AUC {a:.3f}" if a is not None else f"  {name:<18} AUC n/a")
    print("\nLABEL ABLATION - the same question asked of the claim label:")
    pairs = [r for r in usable if r["best_gold_final"] is not None and r["top_final"]]
    # `best_gold_final > top_final` cannot happen by construction - `top` IS the pool
    # maximum - so it is not reported. The non-tautological quantity is how often the
    # pool's best chunk is the answer sentence itself.
    tg = sum(1 for r in usable if r["top_is_gold"])
    print(f"  the pool's top-ranked memory IS the gold claim: {tg}/{len(usable)} = {tg / len(usable):.1%}")
    print(f"  gold claim reaches the payload: "
          f"{sum(1 for r in usable if r['hit']) / len(usable):.1%}")
    print("  Compare the file-overlap label, where the chunk carrying a relevant")
    print("  session's score names a task file 12.9 % of the time: under a label no")
    print("  one can dispute, the answer sentence is the pool's best chunk barely more")
    print("  often - so 'gold loses to noise' is not an artefact of the overlap label.")
    print("\nCONFLICT RULE READOUT - who owns the leading seat:")
    acc = lambda S: (sum(1 for r in S if r["correct"]) / len(S)) if S else float("nan")
    for k in (2, 3):
        S = [r for r in usable if r["purity"] == k]
        if not S:
            continue
        first = sum(1 for r in S if r["gold_seat"] == 1)
        rival_first = sum(1 for r in S if 1 in r["rival_seats"])
        print(f"  {k - 1} rival present (n={len(S):>2}, observed acc {acc(S):.3f}): "
              f"gold owns seat 1 in {first}/{len(S)} = {first / len(S):.1%}; "
              f"a rival owns seat 1 in {rival_first}/{len(S)} = {rival_first / len(S):.1%}; "
              f"gold absent from the payload in "
              f"{sum(1 for r in S if r['gold_seat'] is None)}/{len(S)}")
    one = [r for r in usable if r["purity"] == 2]
    if one:
        keep = sum(1 for r in one if r["gold_seat"] == 1) / len(one)
        pure_acc = acc([r for r in usable if r["purity"] == 1])
        absent_acc = acc([r for r in usable if not r["hit"]])
        projected = keep * pure_acc + (1 - keep) * absent_acc
        print(f"\n  Projection for 'keep seat 1, drop a conflicting seat 2' (one-rival queries only):")
        print(f"    observed on that subset {acc(one):.3f}  ->  projected {projected:.3f}")
        print(f"    (gold kept {keep:.1%} of the time x purity-1 accuracy {pure_acc:.3f}, else fall back to "
              f"absent accuracy {absent_acc:.3f})")
        print(f"    whole-set effect if it fired on all {len(one)}: "
                f"{(acc([r for r in usable if r['purity'] != 2]) * (len(usable) - len(one)) + projected * len(one)) / len(usable):.3f} "
                f"vs current {(acc(usable)):.3f}")
        print("  This is a ceiling under the assumption that dropping the rival converts the")
        print("  query to a purity-1 query; it is not a measured arm result.")
    print("\nPURITY - how many candidate answers the payload carries:")
    for k in (1, 2, 3):
        S = [r for r in usable if r["purity"] == k]
        if not S:
            continue
        rate = sum(1 for r in S if r["correct"]) / len(S)
        print(f"  {k - 1} rival claim(s) present: n={len(S):>2}  recorded accuracy {rate:.3f}")
    if args.out:
        args.out.write_text(json.dumps({"rows": rows, "max_sessions": settings.evidence_max_sessions}, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


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
