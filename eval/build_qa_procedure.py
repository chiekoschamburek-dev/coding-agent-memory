"""Build questions whose answer is a *claim* a past session recorded.

The existing session-artifact questions (`build_qa_session.py`) ask which file a
session worked with. That is answerable from a (session, file) table, and it is
what `run_evidence.py` currently scores: the decisive marker is `file_path`,
which sits at the head of every tool-call record (measured: 939 of 939 Edit
messages name the file before `new_string`). So that question type cannot see
whether the *content* of a memory survives retrieval and truncation.

These questions target the layer a model cannot reconstruct:

* the answer is a verbatim causal statement an earlier session wrote down -- the
  condition under which something breaks, or what had to change in tandem;
* every distractor is also a real claim from the same repository, so the option
  set cannot be pruned by plausibility, generic Python knowledge, or overlap with
  the issue;
* the question names none of them, and each candidate is checked against the
  issue text for leakage before it is accepted.

What this makes measurable is the thing `_OPERATIVE_RES` does not currently
optimise for: prose that explains, rather than the action line that changed a
file. A claim is only recoverable if the returned item carries it whole, so the
metric also prices per-item truncation on prose, which no existing metric does.

The issue text is read from `benchmark.json`'s own `queries[].query`, so this
builder needs only the generated eval data -- unlike `build_qa_session.py`, which
re-opens the original case files for the problem statement.

Usage::

    python eval/build_qa_procedure.py --out eval/data/qa_procedure.json
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

# A cause connector, not merely a mention of failure. "Tests 1 and 2 fail with X"
# reports an event; "X happens *because* Y" is the transferable claim.
_CAUSE = re.compile(
    r"\b(because|caused by|due to|which (?:causes?|fails?|leads? to|makes? it)|"
    r"root cause|the problem is|the issue is|"
    r"(?:does not|doesn't) handle|fails? when|instead of|rather than|"
    r"must (?:also|be|set|be updated)|requires? (?:that|both|updating|the)|"
    r"in order to|otherwise|silently (?:ignored|fails?)|needed to)\b",
    re.IGNORECASE,
)
# Bare enthusiasm. "Now" and "I see" are deliberately NOT here: in this corpus
# that is how a genuine diagnosis is introduced ("Now I can see the issue! The
# problem is in the `deconstruct` method...").
_STATUS = re.compile(r"^(?:\s*(?:Perfect|Great|Excellent|Fantastic|OK|Okay|Finally|Success))\b[!.]?|^\s*(?:I|we) (?:wrote|created|added) (?:a|the) (?:test|script|file)\b", re.IGNORECASE)
# Talk about the agent's own harness, not the repository: a claim about a throwaway
# mock teaches a future task nothing.
_SCAFFOLD = re.compile(
    r"swebench_|testbed/|my (?:mock|fake|test)\w*|manual\.yaml|_preds\.json|scratch",
    re.IGNORECASE,
)
_SYMBOL = re.compile(r"`[^`]{2,}`|[a-z]+_[a-z_]+|[A-Z][a-z]+[A-Z]\w*|\w+\.\w+")
_NOT_PROSE = re.compile(r"^\[tool |^```|\[result\]", re.IGNORECASE)


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def shingles(text: str, k: int = 6) -> set[str]:
    """Word k-grams, used for both the leak test and the two-correct-answers test."""
    words = normalise(text).split()
    if len(words) <= k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


_CARRIER: dict[str, bool] = {}


def carriable(text: str) -> bool:
    """Can one stored chunk hold this claim whole?

    A returned item is a verbatim span of a single chunk, so a claim the chunker
    splits across two chunks can never be delivered -- the item would be scored
    unsolvable no matter how well retrieval worked. Checked at construction so the
    question set only contains answers that exist somewhere in the store.
    """
    key = text[:80]
    hit = _CARRIER.get(key)
    if hit is not None:
        return hit
    from codemem.add.chunker import chunk_content
    from codemem.core.config import Settings

    settings = Settings()
    pieces = chunk_content(
        text,
        target_tokens=settings.target_chunk_tokens,
        max_tokens=settings.max_chunk_tokens,
        hard_chars=settings.hard_chunk_chars,
    )
    want = normalise(text)
    ok = any(want in normalise(piece.text) for piece in pieces)
    _CARRIER[key] = ok
    return ok


def claims_of(memory: dict, issue_shingles: set[str], min_overlap: float) -> list[str]:
    """Assistant prose in one session that reads like a recorded cause.

    Rejects plans ("Let me read ...") and reports by requiring a cause marker,
    and rejects anything whose wording is already in the issue text -- a claim the
    question itself states is not evidence that memory was used.
    """
    out: list[str] = []
    seen: set[str] = set()
    for message in memory.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        text = content.strip()
        if not (80 <= len(text) <= 700):
            continue
        if _NOT_PROSE.search(text[:80]) or _STATUS.search(text) or not _CAUSE.search(text):
            continue
        if _SCAFFOLD.search(text):
            continue
        # A claim has to name something: a function, a field, a flag. Without an
        # identifier it is mood, not information.
        if not _SYMBOL.search(text):
            continue
        # Must be deliverable: one chunk has to be able to carry it whole.
        if not carriable(text):
            continue
        key = normalise(text)[:80]
        if key in seen:
            continue
        seen.add(key)
        own = shingles(text)
        if not own:
            continue
        if len(own & issue_shingles) / len(own) > min_overlap:
            continue
        out.append(text)
    return out


def content_terms(text: str, k_min: int = 4) -> set[str]:
    """Words and identifiers long enough to carry topic, minus the corpus's glue."""
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]+", normalise(text).replace(" ", " ")))
    raw = {t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_\.]+", text) if len(t) >= k_min}
    return ({w for w in raw if len(w) >= k_min}) | {w for w in words if len(w) >= k_min}


def guessability(questions: list[dict]) -> float:
    """How often the option most similar to the issue wording IS the gold.

    A verbatim-n-gram leak test does not catch topical cueing: if only the gold
    mentions `mask` for an issue about masks, a model with no memory can still
    pick it. This is the same baseline `eval/README.md` measured for the
    file-localisation type (0.500 against 0.250 chance), which is what made that
    type weak. Ties count as a miss, so this is a lower bound on guessability.
    """
    hits = 0
    for q in questions:
        issue = content_terms(q["question"].split("Issue:")[-1].split("Which statement")[0])
        scored = [len(content_terms(o) & issue) for o in q["options"]]
        best = max(scored)
        if best and scored.index(best) == q["gold_index"] and scored.count(best) == 1:
            hits += 1
    return hits / len(questions) if questions else 0.0


def build(benchmark: Path, *, n_distractors: int = 3, seed: int = 20260926,
          min_overlap: float = 0.15, max_questions: int | None = None,
          per_query: int = 1, verbose: bool = True) -> dict:
    bench = json.loads(benchmark.read_text(encoding="utf-8"))
    memories = {m["session_id"]: m for m in bench["memories"]}
    repo_sessions: dict[str, list[str]] = defaultdict(list)
    for memory in bench["memories"]:
        repo_sessions[memory["repo"]].append(memory["session_id"])

    rng = random.Random(seed)
    questions: list[dict] = []
    skipped: dict[str, int] = defaultdict(int)
    overlaps: list[float] = []
    used_golds: set[str] = set()

    for query in bench["queries"]:
        issue = query.get("query") or ""
        issue_sh = shingles(issue)
        issue_terms = content_terms(issue)
        relevant = list(query.get("relevant") or [])
        if not relevant:
            skipped["no_relevant_session"] += 1
            continue

        def overlap(text: str, _terms: set[str] = issue_terms) -> int:
            return len(content_terms(text) & _terms)

        anchored: list[tuple[str, list[str]]] = []
        for session_id in relevant:
            memory = memories.get(session_id)
            if memory is None:
                continue
            found = claims_of(memory, issue_sh, min_overlap)
            if found:
                anchored.append((session_id, found))
        if not anchored:
            skipped["no_unleaked_claim"] += 1
            continue

        # Anchor on the session that recorded the most, so the gold is a claim the
        # session genuinely centres on rather than a stray sentence. Take its
        # LEAST issue-similar claim: the answer must not be the option a reader
        # would choose from the issue wording alone. With ``per_query > 1`` the
        # remaining anchors and claims continue in the same discipline — every
        # question still gets its own adversarial distractor set and its own
        # unused gold — so the extra items are independent retrievals, not
        # paraphrases of the first.
        ranked_sessions = sorted(anchored, key=lambda pair: (-len(pair[1]), pair[0]))
        emitted = 0
        for anchor_id, found in ranked_sessions:
            if per_query and emitted >= per_query:
                break
            ranked = sorted(enumerate(found), key=lambda p: (overlap(p[1]), -p[0]))
            for claim_index, _ in ranked:
                if per_query and emitted >= per_query:
                    break
                gold = found[claim_index]
                if normalise(gold)[:80] in used_golds:
                    continue
                gold_overlap = overlap(gold)
                gold_sh = shingles(gold)
                pool: list[tuple[int, str, str]] = []
                for other_id in repo_sessions[query["repo"]]:
                    if other_id == anchor_id:
                        continue
                    other = memories.get(other_id)
                    if other is None:
                        continue
                    for claim in claims_of(other, issue_sh, min_overlap):
                        # A distractor that overlaps the gold would make two right answers.
                        if shingles(claim) & gold_sh:
                            continue
                        pool.append((overlap(claim), other_id, claim))
                if len(pool) < n_distractors:
                    skipped["too_few_distractors"] += 1
                    continue

                # Adversarial construction: at least one distractor has to track the issue
                # more closely than the gold does, so the lexical shortcut -- pick the
                # option whose wording best matches the problem statement -- points away
                # from the right answer. The rest are drawn at random, so the item does not
                # degenerate into "spot the three that sound alike".
                decoys = [p for p in pool if p[0] > gold_overlap]
                if not decoys:
                    skipped["gold_is_most_topical"] += 1
                    continue
                lead = max(decoys, key=lambda p: (p[0], p[1]))
                chosen = [lead] + rng.sample([p for p in pool if p is not lead], n_distractors - 1)
                options = [gold] + [claim for _, _, claim in chosen]
                rng.shuffle(options)
                overlaps.append(len(gold_sh & issue_sh) / len(gold_sh))

                # One claim answers at most one question globally; several queries can
                # anchor on the same busy session, and a repeated gold would turn n items
                # into n looks at one retrieval. Extra questions per query keep the
                # plain id for the first and suffix the rest, so older files line up.
                questions.append(
                    {
                        "query_id": (
                            f"proc::{query['instance_id']}"
                            if emitted == 0
                            else f"proc::{query['instance_id']}::{emitted + 1}"
                        ),
                        "instance_id": query["instance_id"],
                        "repo": query["repo"],
                        "question_type": "session-claim recall",
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
                        "distractor_sessions": [sid for _, sid, _ in chosen],
                        "claim_chars": len(gold),
                        "issue_overlap": round(overlaps[-1], 4),
                        "gold_issue_terms": gold_overlap,
                        "best_distractor_issue_terms": max(p[0] for p in chosen),
                    }
                )
                used_golds.add(normalise(gold)[:80])
                emitted += 1
                if max_questions and len(questions) >= max_questions:
                    break
            if max_questions and len(questions) >= max_questions:
                break

    if verbose:
        print(f"questions: {len(questions)}")
        if skipped:
            print(f"  skipped: {dict(skipped)}")
        positions = defaultdict(int)
        for q in questions:
            positions[q["gold_index"]] += 1
        print(f"  gold position distribution: {dict(sorted(positions.items()))}")
        if overlaps:
            overlaps.sort()
            print(f"  gold wording already in the issue: mean {sum(overlaps) / len(overlaps):.3f} "
                  f"median {overlaps[len(overlaps) // 2]:.3f} max {overlaps[-1]:.3f}")
        leaked = sum(1 for q in questions if shingles(q["gold_claim"]) & shingles(q["question"]))
        print(f"  questions where a gold 6-gram survives into the question: {leaked}")
        lens = sorted(q["claim_chars"] for q in questions)
        if lens:
            print(f"  claim length (chars): median {lens[len(lens) // 2]} "
                  f"p90 {lens[int(len(lens) * 0.9)]} max {lens[-1]}")
        guess = guessability(questions)
        print(f"  ISSUE-WORD BASELINE picks the gold: {guess:.3f} (chance {1 / (n_distractors + 1):.3f})")
        if guess > 0.40:
            print("    WARNING: above ~0.40 this type is answerable without memory; "
                  "tighten by resampling distractors that track the issue more closely.")
        used = defaultdict(int)
        for q in questions:
            used[q["repo"]] += 1
        per_repo = {r: used[r] for r in sorted(used, key=used.get, reverse=True)}
        print(f"  questions per repo: {per_repo}")

    meta = {
        "source": "SWEContextBench Lite Past Experience, via eval/data/benchmark.json",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task_type": "session-claim recall, multiple choice",
        "scoring": "does the returned payload carry the claim verbatim; no judge",
        "purpose": (
            "Score whether memory supplies an explanation the model cannot "
            "reconstruct, as opposed to a file name that a (session, file) table "
            "already answers. The gold is a recorded cause; every distractor is a "
            "real claim from another session in the same repository."
        ),
        "validity_check": (
            "The no-memory condition must score near chance (0.25 for four options), "
            "and each gold claim's wording must be filtered out of the issue text "
            "(issue_overlap <= the builder's min_overlap). Above chance means the "
            "question leaks and the comparison is void."
        ),
        "caveats": [
            "Not the scored suite: CAMBench Coding is not public.",
            "The anchored session is selected by file overlap with the task, which is "
            "our proxy for relevance, not the benchmark's own answer key.",
            "Correctness is asserted about the anchored session only. Whether that "
            "session's claim is the right fix for THIS task is a transfer question "
            "this type deliberately does not ask.",
        ],
        "counts": {"questions": len(questions), "skipped": dict(skipped)},
        "issue_word_baseline": round(guessability(questions), 4),
    }
    return {"meta": meta, "questions": questions}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("eval/data/benchmark.json"))
    parser.add_argument("--out", type=Path, default=Path("eval/data/qa_procedure.json"))
    parser.add_argument("--n-distractors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--min-overlap", type=float, default=0.15,
                        help="max fraction of the claim's 6-grams allowed in the issue")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--per-query", type=int, default=1,
                        help="up to N questions per query, each with its own "
                             "anchored session, unused gold claim and "
                             "adversarial distractor set (raises the ceiling "
                             "from one per query to per_query x queries)")
    args = parser.parse_args(argv)

    if not args.data.exists():
        print("error: run eval/build_benchmark.py first", file=sys.stderr)
        return 2

    data = build(args.data, n_distractors=args.n_distractors, seed=args.seed,
                 min_overlap=args.min_overlap, max_questions=args.max_questions,
                 per_query=args.per_query)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
