"""Audit the chunker's kind labels against the real corpus.

Not a test: a diagnostic that answers "does each kind actually occur, and is the
label right?" -- which the classifier cannot answer from its own rules. Kind
matters downstream because it drives the intent bonus in scoring, and because a
codeish kind unlocks the aggressive symbol/command/package entity extractors.

Run:  PYTHONPATH=src python scripts/audit_chunk_kinds.py
      PYTHONPATH=src python scripts/audit_chunk_kinds.py --data eval/data/benchmark.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")

from codemem.add.chunker import CODE, DIFF, PROSE, TEST, chunk_content, segment  # noqa: E402
from codemem.core.config import Settings  # noqa: E402
from codemem.search.evidence import _operative_score  # noqa: E402

KINDS = (PROSE, CODE, DIFF, "log", "stacktrace", "cmd", "config", TEST)
FENCE_RE = re.compile(r"^\s*(```+|~~~+)\s*([A-Za-z0-9_+#.\-]*)\s*$")
DIFF_HEADER_RE = re.compile(r"^(diff --git |@@ |--- a/|\+\+\+ b/)")
SHELL_PROMPT_RE = re.compile(r"^\s*(\$|>|>>>)\s*\S")


def audit(data: dict, settings: Settings) -> dict:
    counts: collections.Counter = collections.Counter()
    seg_counts: collections.Counter = collections.Counter()
    langs: collections.Counter = collections.Counter()
    fence_langs: collections.Counter = collections.Counter()
    operative: collections.Counter = collections.Counter()
    samples: dict[str, list[str]] = {}
    suspects: dict[str, int] = {"config_task_prompt": 0, "diff_no_header": 0, "cmd_no_prompt": 0}
    kind_totals: dict[str, int] = {"config": 0, "diff": 0, "cmd": 0}
    sessions = messages = chunks = 0
    lines = fenced_lines = tool_lines = 0

    for memory in data["memories"]:
        sessions += 1
        for message in memory["messages"]:
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            messages += 1
            in_fence = False
            for line in content.splitlines():
                lines += 1
                fence = FENCE_RE.match(line)
                if fence:
                    fence_langs[(fence.group(2) or "<none>").strip().lower()] += 1
                    in_fence = not in_fence
                    continue
                if in_fence:
                    fenced_lines += 1
                if "[tool " in line:
                    tool_lines += 1
            for seg in segment(content):
                seg_counts[seg.kind] += 1
            for chunk in chunk_content(
                content,
                target_tokens=settings.target_chunk_tokens,
                max_tokens=settings.max_chunk_tokens,
                hard_chars=settings.hard_chunk_chars,
            ):
                chunks += 1
                counts[chunk.kind] += 1
                langs[chunk.lang or "-"] += 1
                samples.setdefault(chunk.kind, []).append(chunk.text)
                if sum(_operative_score(ln) for ln in chunk.text.splitlines()):
                    operative[chunk.kind] += 1
                body = [ln for ln in chunk.text.splitlines() if ln.strip()]
                if chunk.kind == "config":
                    kind_totals["config"] += 1
                    if chunk.text.startswith("Fix this bug to solve the issue based on"):
                        suspects["config_task_prompt"] += 1
                elif chunk.kind == DIFF:
                    kind_totals["diff"] += 1
                    changed = sum(1 for ln in body if ln[:1] in ("+", "-"))
                    if not any(DIFF_HEADER_RE.match(ln) for ln in body) and changed >= 2:
                        suspects["diff_no_header"] += 1
                elif chunk.kind == "cmd":
                    kind_totals["cmd"] += 1
                    if not SHELL_PROMPT_RE.match(chunk.text):
                        suspects["cmd_no_prompt"] += 1

    return {
        "sessions": sessions,
        "messages": messages,
        "lines": lines,
        "lines_in_fences": fenced_lines,
        "fences": sum(fence_langs.values()),
        "fences_without_lang": fence_langs["<none>"],
        "tool_lines": tool_lines,
        "segments": sum(seg_counts.values()),
        "chunks": chunks,
        "kinds": dict(counts.most_common()),
        "segment_kinds": dict(seg_counts.most_common()),
        "langs": dict(langs.most_common(12)),
        "operative_chunks_by_kind": dict(operative.most_common()),
        "suspects": suspects,
        "kind_totals": kind_totals,
        "samples": {k: v[:2] for k, v in samples.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("eval/data/benchmark.json"))
    parser.add_argument("--show-samples", action="store_true")
    args = parser.parse_args(argv)

    data = json.loads(args.data.read_text(encoding="utf-8"))
    report = audit(data, Settings())
    chunks = report["chunks"]

    print(f"sessions={report['sessions']} messages={report['messages']} "
          f"lines={report['lines']} segments={report['segments']} chunks={chunks}")
    print(f"lines inside fences={report['lines_in_fences']} "
          f"({report['lines_in_fences'] / max(report['lines'], 1) * 100:.1f}%)  "
          f"fences={report['fences']} without a language tag={report['fences_without_lang']}  "
          f"[tool*] lines={report['tool_lines']}")

    print("\nkind distribution (chunk granularity):")
    for kind in KINDS:
        n = report["kinds"].get(kind, 0)
        print(f"  {kind:<11} {n:>7}  {n / max(chunks, 1) * 100:6.3f}%")
    missing = [k for k in KINDS if not report["kinds"].get(k)]
    print(f"  kinds absent from this corpus: {missing or 'none'}")

    print(f"\nsegment granularity (before packing): {report['segment_kinds']}")
    print(f"operative-evidence chunks by kind:    {report['operative_chunks_by_kind']}")
    print(f"top langs: {report['langs']}")

    print("\nlabels worth inspecting (count, and how many look wrong):")
    for kind, key, why in (
        ("config", "config_task_prompt", "is the task prompt, i.e. prose"),
        ("diff", "diff_no_header", "no diff header at all, i.e. a bullet list"),
        ("cmd", "cmd_no_prompt", "no shell prompt, i.e. prose starting with a command word"),
    ):
        total = report["kind_totals"].get(kind, 0)
        bad = report["suspects"].get(key, 0)
        print(f"  {kind:<7} total={total:<6} suspect={bad:<6} "
              f"({bad / max(total, 1) * 100:.0f}%) {why}")

    if args.show_samples:
        for kind, texts in report["samples"].items():
            print(f"\n--- {kind} ---")
            for text in texts:
                print(repr(text[:200]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
