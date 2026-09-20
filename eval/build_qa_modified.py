"""Build questions whose answer is internal to a session: which file did it modify?

Why this shape is not leaky
---------------------------
The earlier question types failed a validity check: the no-memory condition scored
0.689 and 0.70 against 0.25 chance, because the answer was recoverable from the
issue text (vocabulary overlap) or from prior knowledge of a well-known library's
layout. Since the gold file was *related* to the issue, no amount of path
filtering could hide it.

Here every option comes from **the same session**: the gold is a file that session
modified, and the distractors are files it merely read. They are equally related
to the issue and equally plausible as fix locations — reading a file is what you
do before or instead of editing it. So neither vocabulary overlap with the issue
nor library prior knowledge separates them; only the session's own content does.

The validity check is therefore strict and must be run before trusting anything:
**the no-memory condition must score near chance.** A model without memory cannot
know which files a particular past session happened to edit.

Usage::

    python eval/build_qa_modified.py --out eval/data/qa_modified.json
    python eval/run_endtoend.py --qa eval/data/qa_modified.json --limit 30
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

from build_benchmark import normalize_path  # noqa: E402

MODIFY_TOOLS = {"Edit", "Write", "NotebookEdit", "MultiEdit"}
READ_TOOLS = {"Read"}

# Session-harness artefacts and generated output are not repository files.
_NOISE_RE = re.compile(
    r"(^|/)(?:output|logs?|\.git|node_modules|__pycache__|\.venv|venv)(/|$)"
    r"|swebench_9_15/"
    r"|_preds\.json$|manual\.yaml$",
    re.IGNORECASE,
)
_SOURCE_RE = re.compile(
    r"\.(?:py|js|ts|tsx|jsx|rb|go|rs|java|kt|c|cc|cpp|h|hpp|cs|php|swift|scala)$",
    re.IGNORECASE,
)


def repo_file(path: str) -> bool:
    """True for a plausible repository source file (not harness output)."""
    if not path or len(path) > 120:
        return False
    if _NOISE_RE.search(path):
        return False
    return bool(_SOURCE_RE.search(path))


def session_files(path: Path) -> tuple[list[str], list[str]]:
    """(modified, read-only) repository-relative paths for one session."""
    modified: list[str] = []
    read: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if '"file_path"' not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = record.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not (isinstance(part, dict) and part.get("type") == "tool_use"):
                    continue
                name = part.get("name")
                path_value = (part.get("input") or {}).get("file_path")
                if not isinstance(path_value, str):
                    continue
                norm = normalize_path(path_value)
                if not repo_file(norm):
                    continue
                if name in MODIFY_TOOLS:
                    if norm not in modified:
                        modified.append(norm)
                elif name in READ_TOOLS:
                    if norm not in read:
                        read.append(norm)
    # A file that was both read and modified counts as modified.
    read_only = [p for p in read if p not in modified]
    return modified, read_only


def session_issue(source: Path, session_id: str) -> tuple[str | None, str | None]:
    """The issue a session was working on, from its first user message."""
    path = source / "cases" / "SWEContextBench Lite Past Experience" / f"{session_id}.jsonl"
    if not path.exists():
        return None, None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") != "user":
                continue
            message = record.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                content = "\n".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            if not isinstance(content, str):
                continue
            instance = re.search(r"instance_id:\s*([\w.\-]+)", content)
            problem = re.search(
                r"problem_statement:\s*(.*?)(?:\n\s*\w+:\s|\Z)",
                content,
                re.DOTALL,
            )
            text = problem.group(1) if problem else content
            text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()
            return text, instance.group(1) if instance else None
    return None, None


def build(source: Path, benchmark: Path, *, n_distractors: int = 3, seed: int = 20260922,
          verbose: bool = True) -> dict:
    bench = json.loads(benchmark.read_text(encoding="utf-8"))
    repo_of = {m["session_id"]: m["repo"] for m in bench["memories"]}

    questions: list[dict] = []
    skipped = defaultdict(int)
    rng = random.Random(seed)

    session_dirs = sorted(
        (source / "cases" / "SWEContextBench Lite Past Experience").glob("*.jsonl")
    )
    for path in session_dirs:
        session_id = path.stem
        modified, read_only = session_files(path)
        if not modified:
            skipped["session_modified_nothing"] += 1
            continue
        if len(read_only) < n_distractors:
            skipped["too_few_read_only"] += 1
            continue
        issue, instance = session_issue(source, session_id)
        if not issue or len(issue) < 80:
            skipped["no_issue_text"] += 1
            continue

        issue_lower = issue.lower()
        # Neither gold nor distractors may be named in the issue, so the
        # question text cannot favour an option.
        gold_pool = [p for p in modified if p.lower() not in issue_lower]
        distractor_pool = [p for p in read_only if p.lower() not in issue_lower]
        if not gold_pool:
            skipped["gold_named_in_issue"] += 1
            continue
        if len(distractor_pool) < n_distractors:
            skipped["too_few_clean_distractors"] += 1
            continue

        gold = sorted(gold_pool, key=len)[0]
        options = [gold] + rng.sample(distractor_pool, n_distractors)
        rng.shuffle(options)

        questions.append(
            {
                "query_id": f"modified::{session_id}",
                "instance_id": instance or session_id,
                "repo": repo_of.get(session_id, "unknown"),
                "question": (
                    "An earlier engineering session in this repository worked on the "
                    "issue below and made changes to the codebase. That session is not "
                    "visible to you except through any memory you are given.\n\n"
                    "Issue:\n" + issue[:2500] + "\n\n"
                    "Which of these files did that session MODIFY (as opposed to merely "
                    "reading)?"
                ),
                "options": options,
                "gold_index": options.index(gold),
                "gold_file": gold,
                "answer_session": session_id,
                "relevant_sessions": [],
            }
        )

    if verbose:
        print(f"questions: {len(questions)} from {len(session_dirs)} sessions")
        if skipped:
            print(f"  skipped: {dict(skipped)}")
        positions = defaultdict(int)
        for q in questions:
            positions[q["gold_index"]] += 1
        print(f"  gold position distribution: {dict(sorted(positions.items()))}")
        leaked = [q["query_id"] for q in questions
                  if q["gold_file"].lower() in q["question"].lower()]
        print(f"  answers leaking into the question: {len(leaked)}")
        repos = defaultdict(int)
        for q in questions:
            repos[q["repo"]] += 1
        print(f"  repositories: {len(repos)}")

    meta = {
        "source": "SWEContextBench Lite Past Experience (tool-call file operations)",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task_type": "session-internal artifact: modified versus merely read",
        "scoring": "exact match on the option index; no judge",
        "purpose": (
            "Test whether retrieved session CONTENT informs an answer that neither the "
            "issue text nor prior knowledge of the library can supply. All options come "
            "from one session, so they are equally related to the issue."
        ),
        "validity_check": (
            "The no-memory condition MUST score near chance (0.25 with four options). "
            "Above chance invalidates the comparison: the question leaks an easier route "
            "to the answer. This check is the reason this task type exists."
        ),
        "caveats": [
            "Not the scored suite: CAMBench Coding is not public.",
            "Whether a session modified or merely read a file is our inference from its "
            "tool calls, which is reliable for Edit/Write but does not capture changes "
            "made through shell redirection or a test run that rewrites a file.",
        ],
        "counts": {"questions": len(questions), "skipped": dict(skipped)},
    }
    return {"meta": meta, "questions": questions}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("benchmark/SWEContextBench"))
    parser.add_argument("--data", type=Path, default=Path("eval/data/benchmark.json"))
    parser.add_argument("--out", type=Path, default=Path("eval/data/qa_modified.json"))
    parser.add_argument("--n-distractors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260922)
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
