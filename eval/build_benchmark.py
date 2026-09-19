"""Build a retrieval benchmark from SWEContextBench.

Why this dataset
----------------
CAMBench Coding (the scored suite) is not public, so iteration needs a proxy.
SWEContextBench is a good one because it reproduces the two properties the
Coding track is described as testing:

* **same-repository noise** — its 300 "Lite Past Experience" sessions span the
  *same 12 repositories* as its 99 Lite tasks, so distractors share vocabulary,
  file paths, and style with the relevant evidence;
* **reusing engineering experience** — the memory items are real agent session
  trajectories (tool calls, diffs, test runs), not synthetic documents.

Ground truth
------------
The dataset ships **no relevance mapping**: the past-experience sessions solve
different instances from the Lite tasks (verified: zero instance_id overlap), so
there is no official answer key for "which past session is relevant to this
task". We therefore define relevance ourselves, in the most objective way the
data supports:

    A past session is relevant to a task if it touched at least one file that
    the task's gold patch or test patch touches.

That is code-grounded (it comes from the actual diffs and the actual tool calls,
not from a model's opinion), it is reproducible, and it targets exactly the
capability the competition describes: reusing prior work on the same files. It
is a **proxy**, and it is a subset of true relevance — a session that helps for
another reason (a shared technique, an architectural decision) is not credited.
This limitation is stated in the output metadata so results are never mistaken
for official numbers.

Usage::

    python eval/build_benchmark.py \
        --source benchmark/SWEContextBench \
        --out eval/data/benchmark.json

Options control corpus size, which is how noise levels are simulated.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------- parsing ----

_DIFF_NEW_RE = re.compile(r"^\+\+\+ b/(.+?)\s*$", re.MULTILINE)
_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/", re.MULTILINE)
_FILE_PATH_RE = re.compile(r'"file_path":\s*"([^"]+)"')
_TESTBED_RE = re.compile(r"testbed/(.*)$")
_INSTANCE_RE = re.compile(r"instance_id:\s*([\w.\-]+)")
_REPO_RE = re.compile(r"repo:\s*([\w.\-]+/[\w.\-]+)")


def normalize_path(raw: str) -> str:
    """Reduce a path to a repository-relative form.

    Session transcripts contain both ``./swebench_9_15/testbed/django/contrib/
    staticfiles/handlers.py`` and ``.../testbed/django__django/django/contrib/
    staticfiles/handlers.py`` for the same file, while gold patches use
    ``django/contrib/staticfiles/handlers.py``. All must collapse to one key.
    """
    path = raw.replace("\\", "/").strip()
    path = re.sub(r"^\./", "", path)
    match = _TESTBED_RE.search(path)
    if match:
        path = match.group(1)
    # Drop a leading "owner__repo/" segment when present.
    path = re.sub(r"^[\w.\-]+__[\w.\-]+/", "", path)
    return path


def patch_files(patch: str | None) -> set[str]:
    """Files touched by a unified diff."""
    if not patch:
        return set()
    out = {normalize_path(m.group(1)) for m in _DIFF_NEW_RE.finditer(patch)}
    out |= {normalize_path(m.group(1)) for m in _DIFF_GIT_RE.finditer(patch)}
    return {p for p in out if p and p != "/dev/null"}


def ts_to_ms(value: Any) -> int | None:
    if not value:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            text = value.replace("Z", "+00:00")
            return int(datetime.fromisoformat(text).timestamp() * 1000)
        except ValueError:
            return None
    return None


def _flatten_parts(content: Any, *, max_chars: int) -> str:
    """Render a message body as searchable text.

    Tool calls and their results are included because that is where the
    engineering signal lives — the file read, the edit applied, the test run —
    and dropping them would strip the transcript of exactly what a memory
    system is meant to retrieve.
    """
    if isinstance(content, str):
        return content[:max_chars]
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            chunks.append(str(part.get("text") or ""))
        elif kind == "tool_use":
            name = part.get("name") or "tool"
            payload = part.get("input")
            try:
                body = json.dumps(payload, ensure_ascii=False)
            except (TypeError, ValueError):
                body = str(payload)
            chunks.append(f"```\n[tool {name}] {body}\n```")
        elif kind == "tool_result":
            inner = part.get("content")
            if isinstance(inner, list):
                inner = " ".join(
                    str(p.get("text") or "")
                    for p in inner
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            chunks.append(f"```\n[result] {inner}\n```")
        elif kind == "thinking":
            continue  # internal reasoning is not repository experience
    text = "\n\n".join(c for c in chunks if c and c.strip())
    return text[:max_chars]


@dataclass
class Memory:
    id: str
    user_id: str
    session_id: str
    repo: str
    instance_id: str | None
    files: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Query:
    query_id: str
    instance_id: str
    repo: str
    query: str
    files: list[str] = field(default_factory=list)
    relevant: list[str] = field(default_factory=list)


@dataclass
class Benchmark:
    meta: dict[str, Any]
    memories: list[Memory]
    queries: list[Query]


# ---------------------------------------------------------------- loading ----


def load_lite(source: Path) -> list[dict[str, Any]]:
    cases = sorted((source / "cases" / "SWEContextBench Lite").glob("*.json"))
    if not cases:
        raise FileNotFoundError(f"no Lite cases under {source}")
    out = []
    for path in cases:
        with path.open(encoding="utf-8") as handle:
            out.append(json.load(handle))
    return out


def load_sessions(
    source: Path, *, max_chars_per_message: int, max_messages: int
) -> list[Memory]:
    sessions = sorted(
        (source / "cases" / "SWEContextBench Lite Past Experience").glob("*.jsonl")
    )
    if not sessions:
        raise FileNotFoundError(f"no Past Experience sessions under {source}")

    memories: list[Memory] = []
    for path in sessions:
        session_id = path.stem
        messages: list[dict[str, Any]] = []
        files: set[str] = set()
        repo = ""
        instance = ""
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "file_path" in line:
                    files.update(
                        normalize_path(m.group(1)) for m in _FILE_PATH_RE.finditer(line)
                    )

                if record.get("type") not in ("user", "assistant"):
                    continue
                message = record.get("message")
                if not isinstance(message, dict):
                    continue
                role = message.get("role")
                if role not in ("user", "assistant"):
                    continue
                body = _flatten_parts(
                    message.get("content"), max_chars=max_chars_per_message
                )
                if not body.strip():
                    continue
                if not repo:
                    match = _REPO_RE.search(body)
                    if match:
                        repo = match.group(1)
                if not instance:
                    match = _INSTANCE_RE.search(body)
                    if match:
                        instance = match.group(1)
                messages.append(
                    {
                        "role": role,
                        "content": body,
                        "timestamp": ts_to_ms(record.get("timestamp")),
                    }
                )
                if len(messages) >= max_messages:
                    break

        if not messages:
            continue
        if not repo:
            repo = session_id  # unknown; keeps same-repo grouping conservative
        memories.append(
            Memory(
                id=f"pe:{session_id}",
                user_id=f"bench:{repo}",
                session_id=session_id,
                repo=repo,
                instance_id=instance or None,
                files=sorted(files),
                messages=messages,
            )
        )
    return memories


# ------------------------------------------------------------- relevance ----


def build(
    source: Path,
    *,
    max_chars_per_message: int = 20_000,
    max_messages: int = 400,
    repos: Iterable[str] | None = None,
    min_relevant: int = 1,
    verbose: bool = True,
) -> Benchmark:
    lite = load_lite(source)
    sessions = load_sessions(
        source,
        max_chars_per_message=max_chars_per_message,
        max_messages=max_messages,
    )

    wanted = set(repos) if repos else None
    if wanted:
        sessions = [s for s in sessions if s.repo in wanted]
        lite = [t for t in lite if t["repo"] in wanted]

    by_repo: dict[str, list[Memory]] = {}
    for memory in sessions:
        by_repo.setdefault(memory.repo, []).append(memory)

    queries: list[Query] = []
    dropped_no_mapping = 0
    for task in lite:
        files = patch_files(task.get("patch")) | patch_files(task.get("test_patch"))
        query_id = task["instance_id"]
        relevant: list[str] = []
        for memory in by_repo.get(task["repo"], []):
            if set(memory.files) & files:
                # Store the plain session id, which is the same key the runner
                # recovers from the store. Using the namespaced ``Memory.id``
                # here silently never matches and reports zero recall for every
                # system, including a random baseline.
                relevant.append(memory.session_id)
        if len(relevant) < min_relevant:
            dropped_no_mapping += 1
        queries.append(
            Query(
                query_id=query_id,
                instance_id=query_id,
                repo=task["repo"],
                query=task["problem_statement"],
                files=sorted(files),
                relevant=sorted(relevant),
            )
        )

    usable = [q for q in queries if len(q.relevant) >= min_relevant]

    if verbose:
        counts = [len(q.relevant) for q in usable]
        print(f"memories: {len(sessions)} sessions across {len(by_repo)} repositories")
        print(f"queries:  {len(queries)} tasks, {len(usable)} with >= {min_relevant} relevant memory")
        if dropped_no_mapping:
            print(f"  ({dropped_no_mapping} tasks have no file-overlap match and score as misses)")
        if counts:
            print(
                f"  relevant-per-query: min={min(counts)} median={statistics.median(counts)} "
                f"max={max(counts)} mean={statistics.fmean(counts):.1f}"
            )
        sizes = [len(v) for v in by_repo.values()]
        print(f"  memories per repository: min={min(sizes)} max={max(sizes)}")

    meta = {
        "source": "SWEContextBench (jiayuanz3/SWEContextBench)",
        "source_commit": _git_commit(source),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": "proxy",
        "relevance_definition": (
            "A past session is relevant to a task if it touched at least one file "
            "that the task's gold or test patch touches (normalized "
            "repository-relative paths)."
        ),
        "relevance_caveats": (
            "This is OUR proxy, not an official answer key: SWEContextBench ships no "
            "task-to-session mapping (past-experience instances and Lite instances do "
            "not overlap). A session that would help for a reason other than shared "
            "files is not credited, so recall is a lower bound."
        ),
        "not_the_scored_benchmark": (
            "CAMBench Coding is the scored AML suite and is not public. These numbers "
            "are for internal iteration only and must never be presented as official."
        ),
        "memory_isolation": "user_id = bench:<repo>, so retrieval is same-repository",
        "counts": {
            "memories": len(sessions),
            "queries_scored": len(usable),
            "queries_total": len(sessions) and len(queries),
        },
    }
    return Benchmark(meta=meta, memories=sessions, queries=queries)


def _git_commit(source: Path) -> str | None:
    head = source / ".git" / "HEAD"
    try:
        text = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if text.startswith("ref: "):
        ref = source / ".git" / text[5:].strip()
        try:
            return ref.read_text(encoding="utf-8").strip()
        except OSError:
            return None
    return text or None


# ------------------------------------------------------------------ io -----


def save(benchmark: Benchmark, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": benchmark.meta,
        "memories": [asdict(m) for m in benchmark.memories],
        "queries": [asdict(q) for q in benchmark.queries],
    }
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    size_mb = out.stat().st_size / 1_048_576
    print(f"\nwrote {out} ({size_mb:.1f} MB)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("benchmark/SWEContextBench"),
        help="SWEContextBench checkout",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("eval/data/benchmark.json")
    )
    parser.add_argument(
        "--repos",
        default="",
        help="comma-separated repo subset (default: all present in the data)",
    )
    parser.add_argument(
        "--max-chars-per-message",
        type=int,
        default=20_000,
        help="cap per message; the corpus is 122 MB raw",
    )
    parser.add_argument(
        "--max-messages", type=int, default=400, help="cap messages per session"
    )
    parser.add_argument(
        "--min-relevant",
        type=int,
        default=1,
        help="a query needs at least this many relevant memories to be scored",
    )
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"error: {args.source} not found", file=sys.stderr)
        print(
            "clone it first:\n"
            "  git clone --depth 1 https://github.com/jiayuanz3/SWEContextBench "
            "benchmark/SWEContextBench",
            file=sys.stderr,
        )
        return 2

    repos = [r.strip() for r in args.repos.split(",") if r.strip()] or None
    benchmark = build(
        args.source,
        max_chars_per_message=args.max_chars_per_message,
        max_messages=args.max_messages,
        repos=repos,
        min_relevant=args.min_relevant,
    )
    if not benchmark.memories:
        print("error: no memories parsed", file=sys.stderr)
        return 1
    save(benchmark, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
