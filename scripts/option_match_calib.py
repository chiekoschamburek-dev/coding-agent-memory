"""Calibrate a serve-time option↔item matcher against the exact verbatim label.

The conflict-collapse rule (drop the payload's claim-bearing items when it matches two
or more options) needs one thing to exist at all: the service must be able to tell, for
each item it is about to return, which of the four options that item's text belongs to.
This script measures whether an embedding cosine can do it.

Ground truth is exact and free: every option is verbatim corpus text, so "item i carries
option k's claim" is decided by locating each option's claim at message level and mapping
to memories - the same path `claim_level_metric.py` uses, validated 70/70. The candidate
serve-time feature is `cosine(embed(option_k), embed(item.content))`, which is what the
service could actually compute (the encoder is already loaded for the dense channel).

Two ways this rule can die, and both are visible in the output:

  collisions   an item that carries option k's claim also scores above tau for option j,
               because the four claims describe the same bug and sit close in the
               embedding space. Then |M| is inflated and the rule fires on almost every
               query, discarding the answer along with the rival.
  misses       the true item scores below every usable tau, so conflict goes undetected
               and the rule simply never fires.

Reported as a precision/recall sweep over tau, plus the agreement on the quantity the
rule actually branches on: the number of distinct options matched.

Run:
    PYTHONPATH=src python scripts/option_match_calib.py --out eval/results/option_match.json
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
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    qa = json.loads(args.qa.read_text(encoding="utf-8"))["questions"]
    bench = json.loads(args.data.read_text(encoding="utf-8"))
    texts = {
        (m["user_id"], m["session_id"]): [norm(x["content"]) for x in m["messages"]]
        for m in bench["memories"]
    }

    settings = Settings(data_dir=Path(tempfile.mkdtemp()))
    container = Container(settings)
    try:
        for memory in bench["memories"]:
            container.add.handle(
                request_id=f"om:{memory['id']}",
                user_id=memory["user_id"],
                session_id=memory["session_id"],
                messages=[Message(**m) for m in memory["messages"]],
            )
        store, search = container.store, container.search
        embedder = search.retriever.embedder
        if embedder is None or not embedder.available:
            print("no encoder available; this measurement needs the dense encoder")
            return 1
        mem_index: dict[str, list[tuple[int, str, int]]] = {}
        for user_id in {f"bench:{q['repo']}" for q in qa}:
            with store._read() as conn:  # noqa: SLF001 - measurement only
                mem_index[user_id] = [
                    (int(r["id"]), r["session_id"], int(r["msg_index"]))
                    for r in conn.execute(
                        "SELECT m.id AS id, m.session_id AS session_id,"
                        " c.msg_index AS msg_index FROM memory m"
                        " JOIN chunk c ON m.chunk_id = c.id WHERE m.user_id = ?",
                        (user_id,),
                    )
                ]

        def locate(user_id: str, sessions: list[str | None], claim: str) -> set[int]:
            probe = norm(claim)
            for sid in sessions:
                if sid is None:
                    continue
                msgs = {
                    i
                    for i, t in enumerate(texts.get((user_id, sid), []))
                    if probe in t or (len(probe) > 120 and probe[:120] in t)
                }
                if msgs:
                    return {
                        mid
                        for mid, s2, idx in mem_index.get(user_id, [])
                        if s2 == sid and idx in msgs
                    }
            out: set[int] = set()
            for (uid, sid), msg_texts in texts.items():
                if uid != user_id:
                    continue
                for i, t in enumerate(msg_texts):
                    if probe in t or (len(probe) > 120 and probe[:120] in t):
                        out |= {
                            mid
                            for mid, s2, idx in mem_index.get(user_id, [])
                            if s2 == sid and idx == i
                        }
            return out

        pairs: list[dict] = []
        query_states: list[dict] = []
        for q in qa:
            user_id = f"bench:{q['repo']}"
            opts = q.get("options") or []
            distractors = [s for s in (q.get("distractor_sessions") or [])]
            opt_sets = []
            for i, opt in enumerate(opts):
                # The gold option's claim lives in the answer session; the others in
                # one of the named distractor sessions. `locate` tries each in turn and
                # falls back to scanning the repository, so no option-index to
                # distractor-index mapping is assumed (the qa file does not guarantee one).
                sess = (
                    [q.get("answer_session")]
                    if i == q["gold_index"]
                    else distractors
                )
                opt_sets.append(locate(user_id, sess, opt))
            items = search.handle(
                user_id=user_id, query=q["question"], options=opts, top_k=args.top_k
            )
            if not items or not any(opt_sets):
                continue
            contents = [it.content for it in items]
            vecs = embedder.embed(list(opts) + contents) or []
            if len(vecs) != len(opts) + len(contents):
                continue
            ovec, ivec = vecs[: len(opts)], vecs[len(opts):]
            exact_M = 0
            cos_M_at = {t: 0 for t in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)}
            for oi, ov in enumerate(ovec):
                matched_exact = {
                    ii for ii, it in enumerate(items) if it.memory_id in opt_sets[oi]
                }
                matched_exact_any = bool(matched_exact)
                exact_M += 1 if matched_exact_any else 0
                for ii, iv in enumerate(ivec):
                    s = sum(a * b for a, b in zip(ov, iv))
                    pairs.append(
                        {
                            "query_id": q["query_id"],
                            "option": oi,
                            "item": ii,
                            "cos": round(s, 4),
                            "exact": ii in matched_exact,
                            "is_gold_option": oi == q["gold_index"],
                        }
                    )
                for t in cos_M_at:
                    if any(s >= t for s in (
                        sum(a * b for a, b in zip(ov, iv)) for iv in ivec
                    )):
                        cos_M_at[t] += 1
            query_states.append(
                {
                    "query_id": q["query_id"],
                    "M_exact": exact_M,
                    "M_cos": cos_M_at,
                    "n_items": len(items),
                }
            )
    finally:
        container.close()

    pos = [p for p in pairs if p["exact"]]
    neg = [p for p in pairs if not p["exact"]]
    print(
        f"\n(item, option) pairs: {len(pairs)}   positives (verbatim match): {len(pos)}"
        f"   negatives: {len(neg)}"
    )
    print(f"queries scored: {len(query_states)}")
    print("\n  tau   recall   precision    F1   | collisions: negatives scoring >= tau")
    best = None
    for t in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        tp = sum(1 for p in pos if p["cos"] >= t)
        fp = sum(1 for p in neg if p["cos"] >= t)
        fn = len(pos) - tp
        rec = tp / max(1, tp + fn)
        prec = tp / max(1, tp + fp)
        f1 = 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)
        print(f"  {t:.2f}   {rec:>6.3f}   {prec:>8.3f} {f1:>6.3f}   {fp}")
        if best is None or f1 > best[1]:
            best = (t, f1, rec, prec)
    print(f"\n  best F1 at tau={best[0]:.2f}: F1={best[1]:.3f} recall={best[2]:.3f} precision={best[3]:.3f}")

    print("\nagreement on the quantity the rule branches on (# options matched):")
    for t in (0.55, 0.65, 0.75):
        agree = sum(1 for s in query_states if s["M_cos"][t] == s["M_exact"])
        inflate = sum(1 for s in query_states if s["M_cos"][t] > s["M_exact"])
        fire_exact = sum(1 for s in query_states if s["M_exact"] >= 2)
        fire_cos = sum(1 for s in query_states if s["M_cos"][t] >= 2)
        print(
            f"  tau={t:.2f}: |M| agrees {agree}/{len(query_states)} = {agree / len(query_states):.1%}"
            f"   inflated {inflate}   fires(rule) exact {fire_exact} vs detected {fire_cos}"
        )
    print(
        "\nVerdict rule: the collapse is buildable only if some tau keeps |M| agreement"
        "\n>=95 % AND detected-fire-count within +/-3 of the exact one. Collisions (inflated"
        "\n|M|) fire the rule on queries with no conflict, which discards correct answers."
    )
    if args.out:
        args.out.write_text(
            json.dumps({"pairs": pairs, "states": query_states}, indent=1), encoding="utf-8"
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
