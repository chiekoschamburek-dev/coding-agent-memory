"""Build objectively-scored QA questions from SWEContextBench Lite.

Why this shape
--------------
Retrieval metrics (recall@k, nDCG) measure whether a session was found. The
platform scores whether the returned memory let the answer model *solve the
task*, and those are different things — a session can be relevant yet useless,
or missing yet not needed. So the end-to-end stage needs questions with a known
correct answer.

The question type here is **file localisation**: given the issue text, which file
must be changed? The gold answer comes from the task's own patch, so it is
objective and needs no judge. It is also exactly the capability the track
describes — "retrieve, filter and reuse debugging and development experience from
the same repository" — because locating the right file from an issue alone is
what a reused precedent most directly supplies.

Distractors are files taken from *other* tasks in the same repository, so they
are plausible rather than obviously wrong; a question with options from unrelated
repositories would be trivially separable and would measure nothing.

The paired no-memory condition
------------------------------
Each question is answered twice: once with no memory (the model's own prior) and
once with our retrieved memories. The **difference** is what the memory system
contributes. Absolute accuracy alone is not interpretable, since a strong model
may already know the repository.

Usage::

    python eval/build_qa.py --out eval/data/qa.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_DIFF_NEW_RE = re.compile(r"^\+\+\+ b/(.+?)\s*$", re.MULTILINE)
_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/", re.MULTILINE)


def patch_files(patch: str | None) -> list[str]:
    """Files touched by a unified diff, in first-appearance order."""
    if not patch:
        return []
    out: list[str] = []
    for match in _DIFF_NEW_RE.finditer(patch):
        name = match.group(1).strip()
        if name and name != "/dev/null" and name not in out:
            out.append(name)
    if not out:
        for match in _DIFF_GIT_RE.finditer(patch):
            name = match.group(1).strip()
            if name and name not in out:
                out.append(name)
    return out


def is_test_path(path: str) -> bool:
    lowered = path.lower()
    return (
        "/test" in lowered
        or lowered.startswith("test")
        or "/tests/" in lowered
        or lowered.endswith("_test.py")
    )


@dataclass
class Question:
    query_id: str
    instance_id: str
    repo: str
    question: str
    options: list[str]
    gold_index: int
    gold_file: str
    gold_files_all: list[str]
    # Relevant past sessions, by file overlap. Carried for reporting only.
    relevant_sessions: list[str] = field(default_factory=list)


def clean_statement(text: str) -> str:
    """Strip the issue boilerplate that SWE-bench appends to problem statements.

    The trailing comment-template block is not part of the issue and would add
    noise to the question.
    """
    text = re.sub(r"<!--.*?-->", "", text or "", flags=re.DOTALL)
    text = re.sub(r"(?im)^\s*(?:This comments are hi.*|hints?:.*)$", "", text)
    return text.strip()


def build(
    source: Path,
    *,
    n_distractors: int = 3,
    seed: int = 20260920,
    verbose: bool = True,
) -> dict:
    cases = sorted((source / "cases" / "SWEContextBench Lite").glob("*.json"))
    if not cases:
        raise FileNotFoundError(f"no Lite cases under {source}")

    tasks = []
    for path in cases:
        with path.open(encoding="utf-8") as handle:
            tasks.append(json.load(handle))

    # Distractor pool: every source file any task in this repository touches, so
    # alternatives look like real candidate locations.
    repo_files: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        for name in patch_files(task.get("patch")):
            if not is_test_path(name) and name not in repo_files[task["repo"]]:
                repo_files[task["repo"]].append(name)

    rng = random.Random(seed)
    questions: list[Question] = []
    skipped: list[str] = []

    for task in tasks:
        gold_candidates = [
            f for f in patch_files(task.get("patch")) if not is_test_path(f)
        ]
        if not gold_candidates:
            skipped.append(task["instance_id"])
            continue

        gold_file = gold_candidates[0]
        pool = [f for f in repo_files[task["repo"]] if f != gold_file]
        if len(pool) < n_distractors:
            skipped.append(task["instance_id"])
            continue

        options = [gold_file] + rng.sample(pool, n_distractors)
        rng.shuffle(options)
        gold_index = options.index(gold_file)

        questions.append(
            Question(
                query_id=task["instance_id"],
                instance_id=task["instance_id"],
                repo=task["repo"],
                question=clean_statement(task["problem_statement"]),
                options=options,
                gold_index=gold_index,
                gold_file=gold_file,
                gold_files_all=patch_files(task.get("patch")),
            )
        )

    if verbose:
        print(f"questions: {len(questions)} from {len(tasks)} tasks")
        if skipped:
            print(f"  skipped {len(skipped)} (no usable source file or too few distractors)")
        per_repo = defaultdict(int)
        for q in questions:
            per_repo[q.repo] += 1
        print(f"  repositories: {len(per_repo)}")
        # Sanity: the gold answer must not be guessable by position.
        positions = defaultdict(int)
        for q in questions:
            positions[q.gold_index] += 1
        print(f"  gold option position distribution: {dict(sorted(positions.items()))}")

    meta = {
        "source": "SWEContextBench Lite (jiayuanz3/SWEContextBench)",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task_type": "file localisation, multiple choice",
        "scoring": "exact match on the option index; no judge involved",
        "gold_definition": (
            "The first non-test file in the task's own gold patch. Distractors are "
            "source files touched by other tasks in the same repository."
        ),
        "purpose": (
            "End-to-end proxy: does the memory system let an answer model locate the "
            "fix that the task's own patch records? Compared against a no-memory "
            "condition, so the difference is the memory contribution."
        ),
        "caveats": [
            "Not the scored suite: CAMBench Coding is not public.",
            "A strong model may already know a well-known repository, which is why the "
            "no-memory condition is always reported alongside.",
            "File localisation is one facet of the task; producing a correct patch "
            "would need the SWE-bench test harness and repository Docker images.",
        ],
        "counts": {"questions": len(questions), "skipped": len(skipped)},
    }
    return {"meta": meta, "questions": [asdict(q) for q in questions]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("benchmark/SWEContextBench"))
    parser.add_argument("--out", type=Path, default=Path("eval/data/qa.json"))
    parser.add_argument("--n-distractors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"error: {args.source} not found", file=sys.stderr)
        return 2

    data = build(args.source, n_distractors=args.n_distractors, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
