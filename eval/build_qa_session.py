"""Build questions whose answer exists ONLY in a past session.

The file-localisation questions in ``build_qa.py`` turned out to be answerable
from the issue text plus prior knowledge of well-known libraries, so they cannot
show whether a memory system contributes anything. These questions remove both
escape routes:

* the answer is an artifact of a **specific past session** (a file that session
  worked with), which no amount of library knowledge can supply;
* every candidate artifact is a real file from the same repository, and the
  question names none of them, so lexical overlap with the issue cannot decide
  it either.

The question still contains the issue text, because that is what drives
retrieval — the same input the platform's Search receives. What it does not
contain is the answer.

Consequently the no-memory condition should score near chance (0.25 for four
options). That is the validity check on the question type: if a model without
memory scores above chance, the question is leaking and must be tightened.

Usage::

    python eval/build_qa_session.py --out eval/data/qa_session.json
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

_MEANINGLESS = re.compile(
    r"(^|/)(?:\.git|node_modules|\.github|venv|\.venv|__pycache__)(/|$)"
    r"|\.(?:pyc|lock|log|png|jpg|jpeg|gif|ico|svg|whl|so|dylib)$",
    re.IGNORECASE,
)


def usable_path(path: str) -> bool:
    if not path or len(path) > 120:
        return False
    if _MEANINGLESS.search(path):
        return False
    # Must look like a source file we can plausibly ask about.
    return bool(re.search(r"\.(?:py|js|ts|tsx|jsx|rb|go|rs|java|kt|c|cc|cpp|h|hpp|cs|php|swift|scala|m|mm)$", path, re.I))


def session_text(memory: dict) -> str:
    parts = []
    for message in memory.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def build(source: Path, benchmark: Path, *, n_distractors: int = 3, seed: int = 20260921,
          verbose: bool = True) -> dict:
    bench = json.loads(benchmark.read_text(encoding="utf-8"))
    memories_by_session = {m["session_id"]: m for m in bench["memories"]}
    repo_sessions: dict[str, list[str]] = defaultdict(list)
    for memory in bench["memories"]:
        repo_sessions[memory["repo"]].append(memory["session_id"])

    cases = sorted((source / "cases" / "SWEContextBench Lite").glob("*.json"))
    tasks = {}
    for path in cases:
        with path.open(encoding="utf-8") as handle:
            task = json.load(handle)
        tasks[task["instance_id"]] = task

    rng = random.Random(seed)
    questions: list[dict] = []
    skipped = defaultdict(int)

    for query in bench["queries"]:
        task = tasks.get(query["instance_id"])
        if task is None:
            skipped["no_task"] += 1
            continue
        relevant = list(query.get("relevant") or [])
        if not relevant:
            skipped["no_relevant_session"] += 1
            continue

        issue = re.sub(r"<!--.*?-->", "", task.get("problem_statement") or "", flags=re.DOTALL)
        issue_lower = issue.lower()

        # The answer: a file the relevant session worked with, which the issue
        # never names. Prefer sessions with more candidate artifacts so the
        # question is not a coin flip between one option and noise.
        candidates: list[tuple[str, str]] = []  # (session_id, path)
        for session_id in relevant:
            memory = memories_by_session.get(session_id)
            if memory is None:
                continue
            text = session_text(memory)
            text_lower = text.lower()
            for path in memory.get("files", []):
                if not usable_path(path):
                    continue
                if path.lower() in issue_lower:
                    continue  # would leak the answer into the question
                if path.lower() not in text_lower:
                    continue  # must be recoverable from what we stored
                candidates.append((session_id, path))
        if not candidates:
            skipped["no_non_leaking_artifact"] += 1
            continue

        # Pick the artifact belonging to the session with the most candidates,
        # then the shortest path (usually the most central file).
        counts = defaultdict(int)
        for session_id, _ in candidates:
            counts[session_id] += 1
        best_session = max(counts, key=lambda s: (counts[s], s))
        pool = sorted(p for s, p in candidates if s == best_session)
        gold = pool[0]

        # Distractors: real files from other sessions in the same repository,
        # also absent from the issue so no option is favoured by the question text.
        other: list[str] = []
        for session_id in repo_sessions[query["repo"]]:
            if session_id in set(relevant):
                continue
            for path in memories_by_session[session_id].get("files", []):
                if not usable_path(path) or path == gold:
                    continue
                if path.lower() in issue_lower:
                    continue
                if path not in other:
                    other.append(path)
        if len(other) < n_distractors:
            skipped["too_few_distractors"] += 1
            continue

        options = [gold] + rng.sample(other, n_distractors)
        rng.shuffle(options)

        questions.append(
            {
                "query_id": f"session::{query['instance_id']}",
                "instance_id": query["instance_id"],
                "repo": query["repo"],
                "question": (
                    "An earlier engineering session in this repository worked on a "
                    "problem related to the issue below. That session is not visible "
                    "to you except through any memory you are given.\n\n"
                    "Issue:\n" + issue.strip()[:2500] + "\n\n"
                    "Which of these files did that earlier session work with?"
                ),
                "options": options,
                "gold_index": options.index(gold),
                "gold_file": gold,
                "answer_session": best_session,
                "relevant_sessions": relevant,
            }
        )

    if verbose:
        print(f"questions: {len(questions)}")
        if skipped:
            print(f"  skipped: {dict(skipped)}")
        positions = defaultdict(int)
        for q in questions:
            positions[q["gold_index"]] += 1
        print(f"  gold position distribution: {dict(sorted(positions.items()))}")
        leaked = sum(1 for q in questions if q["gold_file"].lower() in q["question"].lower())
        print(f"  questions where the answer leaks into the question: {leaked}")

    meta = {
        "source": "SWEContextBench Lite + Lite Past Experience",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task_type": "session-artifact recall, multiple choice",
        "scoring": "exact match on the option index; no judge",
        "purpose": (
            "Isolate whether memory supplies anything that neither lexical overlap "
            "with the issue nor prior knowledge of the library can supply. The answer "
            "is an artifact of one specific past session; the issue names none of the "
            "options."
        ),
        "validity_check": (
            "The no-memory condition must score near chance (0.25 for four options). "
            "Above chance means the question leaks and the comparison is void."
        ),
        "caveats": [
            "Not the scored suite: CAMBench Coding is not public.",
            "This measures whether retrieved session CONTENT is usable, which is a "
            "different question from whether the retrieved session was the right one.",
        ],
        "counts": {"questions": len(questions), "skipped": dict(skipped)},
    }
    return {"meta": meta, "questions": questions}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("benchmark/SWEContextBench"))
    parser.add_argument("--data", type=Path, default=Path("eval/data/benchmark.json"))
    parser.add_argument("--out", type=Path, default=Path("eval/data/qa_session.json"))
    parser.add_argument("--n-distractors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args(argv)

    if not args.source.exists() or not args.data.exists():
        print("error: run eval/build_benchmark.py first", file=sys.stderr)
        return 2

    data = build(args.source, args.data, n_distractors=args.n_distractors, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
