"""Build claim questions anchored by TOPICALITY, not file overlap.

The procedure set (`build_qa_procedure.py`) anchors relevance on
``query["relevant"]`` — sessions sharing files with the task. That anchor
has two documented weaknesses: it is our proxy, not the benchmark's answer
key, and it caps the set at 58 questions because the recorded-cause prose
under file-overlap sessions runs out. Every arm of the session-ranking
campaign was selected on that one anchor, so a second instrument anchored
differently is the generalization test (and the selection-bias cure) the
submission record calls for.

This builder anchors on claim topicality: a session is relevant to an issue
when the claims it RECORDS are about the problem the issue describes —
measured as embedding cosine between the issue text and the session's
qualifying claims (the session's first message rides along as a tie
signal). File overlap plays no part in the choice.

Discipline inherited unchanged from build_qa_procedure: the gold is a
cause-marked, identifier-bearing, unleaked, single-chunk-carriable claim;
every distractor is a real claim from another session in the same repo; at
least one distractor tracks the issue wording more closely than the gold;
the guessability gate must stay near chance.

Output is split into a tuning two-thirds and a SEALED one-third. The sealed
file exists so the final confirmation of any winner runs on questions no
lever was tuned against. Do not open the sealed file during tuning.

Usage::

    PYTHONPATH=src python eval/build_qa_claim.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_qa_procedure import (  # noqa: E402
    carriable,
    claims_of,
    content_terms,
    guessability,
    normalise,
    shingles,
)


def first_message(memory: dict) -> str:
    for message in memory.get("messages", []):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def build(benchmark: Path, *, n_distractors: int = 3, seed: int = 20261005,
          min_overlap: float = 0.15, top_affinity: float = 0.55,
          first_weight: float = 0.15, verbose: bool = True) -> tuple[dict, dict]:
    bench = json.loads(benchmark.read_text(encoding="utf-8"))
    memories = {m["session_id"]: m for m in bench["memories"]}
    repo_sessions: dict[str, list[str]] = defaultdict(list)
    for memory in bench["memories"]:
        repo_sessions[memory["repo"]].append(memory["session_id"])

    # ---- one embed pass: issues, first messages, every qualifying claim ----
    from codemem.core.config import Settings
    from codemem.embed import Instance as EmbedInstance

    embedder = EmbedInstance.get(Settings())
    if not (embedder and embedder.available):
        print("error: local embedder unavailable; claims need embeddings",
              file=sys.stderr)
        return {}, {}

    issue_texts = [q.get("query") or "" for q in bench["queries"]]
    session_claims: dict[str, list[str]] = {}
    session_first: dict[str, str] = {}
    for sid, memory in memories.items():
        session_claims[sid] = claims_of(memory, set(), min_overlap)
        session_first[sid] = first_message(memory)

    claim_index: list[tuple[str, str]] = []
    for sid, found in session_claims.items():
        for claim in found:
            claim_index.append((sid, claim))
    uniq_claims = list(dict.fromkeys(c for _, c in claim_index))

    def embed_all(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            out.extend(embedder.embed(texts[i : i + 64]) or [])
        return out

    issue_vecs = embed_all(issue_texts)
    first_vecs = {
        sid: v for sid, v in zip(
            session_first,
            embed_all([session_first[s] for s in session_first]),
        ) if any(v)
    }
    claim_vecs = dict(zip(uniq_claims, embed_all(uniq_claims)))

    rng = random.Random(seed)
    questions: list[dict] = []
    skipped: dict[str, int] = defaultdict(int)
    used_golds: set[str] = set()
    old_anchors: dict[str, set[str]] = defaultdict(set)
    for q in bench["queries"]:
        for sid in q.get("relevant") or []:
            old_anchors[q["instance_id"]].add(sid)

    anchor_overlap = 0
    for qi, query in enumerate(bench["queries"]):
        issue = query.get("query") or ""
        issue_sh = shingles(issue)
        issue_terms = content_terms(issue)
        ivec = issue_vecs[qi]

        # Topical affinity per session: best claim cosine plus a small
        # first-message signal. File overlap is never consulted.
        affinity: list[tuple[float, str]] = []
        for sid in repo_sessions[query["repo"]]:
            best = max(
                (cosine(ivec, claim_vecs[c]) for c in session_claims[sid]
                 if c in claim_vecs), default=0.0,
            )
            fvec = first_vecs.get(sid)
            first_cos = cosine(ivec, fvec) if fvec else 0.0
            affinity.append((best + first_weight * max(first_cos, 0.0), sid))
        affinity.sort(reverse=True)
        if not affinity or affinity[0][0] < top_affinity:
            skipped["no_topical_session"] += 1
            continue

        def overlap(text: str, _terms: set[str] = issue_terms) -> int:
            return len(content_terms(text) & _terms)

        anchored = [
            (sid, session_claims[sid]) for _, sid in affinity[:5]
            if session_claims[sid]
        ]
        if not anchored:
            skipped["anchor_has_no_claim"] += 1
            continue
        questions_this_query = 0
        emitted_any = False

        for anchor_id, found in anchored:
            if questions_this_query >= 2:
                break
            if anchor_id in old_anchors[query["instance_id"]]:
                anchor_overlap += 1

            # Gold = the qualifying claim least close to the issue by term
            # overlap (the procedure set's rule); the adversarial constraint
            # below handles the semantic side.
            ranked = sorted(
                enumerate(found), key=lambda p: (overlap(p[1]), -p[0])
            )
            emitted = False
            for claim_index_pos, _ in ranked:
                gold = found[claim_index_pos]
                if normalise(gold)[:80] in used_golds:
                    continue
                gold_overlap = overlap(gold)
                gold_cos = cosine(ivec, claim_vecs.get(gold, [0.0]))
                gold_sh = shingles(gold)
                pool: list[tuple[int, str, str, float]] = []
                for other_id in repo_sessions[query["repo"]]:
                    if other_id == anchor_id:
                        continue
                    other = memories.get(other_id)
                    if other is None:
                        continue
                    for claim in claims_of(other, issue_sh, min_overlap):
                        if shingles(claim) & gold_sh:
                            continue
                        pool.append((
                            overlap(claim), other_id, claim,
                            cosine(ivec, claim_vecs.get(claim, [0.0])),
                        ))
                if len(pool) < n_distractors:
                    skipped["too_few_distractors"] += 1
                    continue
                # Adversarial lead in semantic space: the designated
                # distractor must be MORE similar to the issue than the
                # gold, so the "sounds like the right diagnosis" shortcut
                # points away from the answer (the term-overlap shortcut is
                # already handled by the gold's least-overlap selection).
                decoys = [p for p in pool if p[3] > gold_cos]
                if not decoys:
                    skipped["gold_is_most_topical"] += 1
                    continue
                lead = max(decoys, key=lambda p: (p[3], p[0], p[1]))
                chosen = [lead] + rng.sample(
                    [p for p in pool if p is not lead], n_distractors - 1
                )
                options = [gold] + [claim for _, _, claim, _ in chosen]
                rng.shuffle(options)
                questions.append({
                    "query_id": (
                        f"claim::{query['instance_id']}"
                        if questions_this_query == 0
                        else f"claim::{query['instance_id']}::{questions_this_query + 1}"
                    ),
                    "instance_id": query["instance_id"],
                    "repo": query["repo"],
                    "question_type": "claim-topical recall",
                    "question": (
                        "An earlier engineering session in this repository worked on a "
                        "problem related to the issue below. That session is not visible "
                        "to you except through any memory you are given.\n\n"
                        "Issue:\n" + issue.strip()[:2500] + "\n\n"
                        "Which statement about that problem did that earlier session "
                        "record while working on it?"
                    ),
                    "options": options,
                    "gold_index": options.index(gold),
                    "gold_claim": gold,
                    "answer_session": anchor_id,
                    "anchor_affinity": round(affinity[0][0], 4),
                    "anchor_is_file_overlap": anchor_id in old_anchors[
                        query["instance_id"]
                    ],
                    "distractor_sessions": [sid for _, sid, _, _ in chosen],
                })
                used_golds.add(normalise(gold)[:80])
                emitted = True
                emitted_any = True
                questions_this_query += 1
                if questions_this_query >= 2:
                    break
        if not emitted_any:
            skipped["no_acceptable_gold"] += 1

    if verbose:
        print(f"questions: {len(questions)}")
        if skipped:
            print(f"  skipped: {dict(skipped)}")
        print(f"  anchors that coincide with the file-overlap anchor: "
              f"{anchor_overlap}/{len(questions)}")
        positions = defaultdict(int)
        for q in questions:
            positions[q["gold_index"]] += 1
        print(f"  gold position distribution: {dict(sorted(positions.items()))}")
        guess = guessability(questions)
        print(f"  ISSUE-WORD BASELINE picks the gold: {guess:.3f} "
              f"(chance {1 / (n_distractors + 1):.3f})")
        ivec_by_instance = {
            bench["queries"][i]["instance_id"]: issue_vecs[i]
            for i in range(len(bench["queries"]))
        }
        sem_hits = 0
        for q in questions:
            ivec = ivec_by_instance.get(q["instance_id"])
            if ivec is None:
                continue
            scored = [
                cosine(ivec, claim_vecs.get(o, [0.0])) for o in q["options"]
            ]
            best = max(scored)
            if best and scored.index(best) == q["gold_index"] \
                    and scored.count(best) == 1:
                sem_hits += 1
        sem = sem_hits / len(questions) if questions else 0.0
        print(f"  SEMANTIC BASELINE picks the gold: {sem:.3f} "
              f"(the no-memory floor proxy; must stay <= ~0.35)")
        lens = sorted(len(q["gold_claim"]) for q in questions)
        if lens:
            print(f"  claim length (chars): median {lens[len(lens) // 2]} "
                  f"p90 {lens[int(len(lens) * 0.9)]} max {lens[-1]}")

    def with_meta(qs: list[dict], sealed: bool) -> dict:
        return {
            "meta": {
                "source": "SWEContextBench Lite Past Experience, via "
                          "eval/data/benchmark.json",
                "created_at": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
                "task_type": "claim-topical recall, multiple choice",
                "anchor": (
                    "topical: embedding cosine between the issue and the "
                    "session's recorded claims (first message as a tie "
                    "signal); file overlap never consulted"
                ),
                "sealed": sealed,
                "scoring": "does the returned payload carry the claim verbatim",
                "validity_check": (
                    "no_memory must score near chance (0.25); the issue-word "
                    "baseline must stay near chance; anchor coincidence with "
                    "the file-overlap set is reported, not minimised"
                ),
                "caveats": [
                    "Not the scored suite: CAMBench Coding is not public.",
                    "Topical affinity is our second proxy for relevance — "
                    "the generalisation test is whether arm ordering agrees "
                    "across the two anchors, not whether either is 'true'.",
                ],
                "counts": {"questions": len(qs), "skipped": dict(skipped)},
                "issue_word_baseline": round(guessability(questions), 4),
            },
            "questions": qs,
        }

    order = list(range(len(questions)))
    random.Random(seed + 1).shuffle(order)
    n_sealed = max(1, len(questions) // 3)
    sealed_ids = set(order[:n_sealed])
    sealed_qs = [q for i, q in enumerate(questions) if i in sealed_ids]
    tune_qs = [q for i, q in enumerate(questions) if i not in sealed_ids]
    return with_meta(tune_qs, False), with_meta(sealed_qs, True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("eval/data/benchmark.json"))
    parser.add_argument("--out-tune", type=Path,
                        default=Path("eval/data/qa_claim_tune.json"))
    parser.add_argument("--out-sealed", type=Path,
                        default=Path("eval/data/qa_claim_sealed.json"))
    parser.add_argument("--n-distractors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--min-overlap", type=float, default=0.15)
    parser.add_argument("--top-affinity", type=float, default=0.55)
    parser.add_argument("--first-weight", type=float, default=0.15)
    args = parser.parse_args(argv)

    if not args.data.exists():
        print("error: run eval/build_benchmark.py first", file=sys.stderr)
        return 2

    tune, sealed = build(
        args.data, n_distractors=args.n_distractors, seed=args.seed,
        min_overlap=args.min_overlap, top_affinity=args.top_affinity,
        first_weight=args.first_weight,
    )
    if not tune:
        return 1
    for path, payload in ((args.out_tune, tune), (args.out_sealed, sealed)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
        print(f"wrote {path} ({len(payload['questions'])} questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
