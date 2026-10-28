"""Self-contained repair case packets.

The unit Search returns to a tool-using agent is a **case packet**, not a raw
chunk and not a human-facing summary.  The packet must be actionable without
dereferencing anything: the Add/Search API only exposes ``content``, so a
chunk id is audit metadata, never the meaning of an item.

Two rules drive the schema:

* **Conditions are facts, applicability is a relation.**  A packet stores the
  admission constraints it can prove (version, module, symbol, error string),
  never an ``applicable: true`` verdict.  Applicability is evaluated per query
  by the listwise stage (``codemem.applicability``).
* **One piece of information appears once, in its most executable form.**
  The historical diff is stored as diff text (the current repo cannot supply
  it), the test as an exact command plus expected output, the problem as a
  bounded quote.  There is no prose restatement and no chunk excerpt block.

Extraction is deterministic: files come from ``+++ b/...`` headers, symbols
from added ``def``/``class`` lines, verification from command/test-result
chunks.  The model only supplies what it alone can: the problem statement,
the implement description, the pitfalls and the declared conditions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .store import StoredChunk

CASE_PAYLOAD_VERSION = 4

# Hard ceilings.  Callers may lower them per request; these values only stop a
# pathological chunk from producing an unbounded item.
MAX_FIELD_CHARS = 320
MAX_PATCH_LINES = 60
MAX_FILES = 8
MAX_SYMBOLS = 12
MAX_CONDITIONS = 8
MAX_PITFALLS = 3
MAX_PITFALL_CHARS = 120
MAX_VERIFY_CHARS = 160
MAX_CONDITION_CHARS = 120

# A condition is a *typed anchor*, not free text: its type decides which query
# channel can match it (``_anchor_kinds``)
# search.  An open enum lets the extractor invent types such as ``file`` or
# ``line`` that no channel ever queries, so they only bloat ``case_anchors``.
# ``test_name`` and ``frame`` are produced by ``derive_conditions`` and are not
# offered to the model, which only writes the five semantic kinds.
CONDITION_TYPES = frozenset(
    {"version", "module", "symbol", "error", "config", "test_name", "frame"}
)

__all__ = [
    "CASE_PAYLOAD_VERSION",
    "CONDITION_TYPES",
    "CasePacket",
    "Condition",
    "VerifyRecipe",
    "build_packet",
    "packet_summary",
    "render_packet",
    "extract_patch",
    "extract_files",
    "extract_symbols",
    "extract_verify",
    "derive_conditions",
]


# --------------------------------------------------------------------------
# value objects
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """One admission constraint, with the evidence it came from."""

    type: str
    value: str
    basis: str = ""

    def to_dict(self) -> dict:
        out = {"type": self.type, "value": self.value}
        if self.basis:
            out["basis"] = self.basis
        return out

    @staticmethod
    def from_dict(raw: object) -> "Condition | None":
        if not isinstance(raw, dict):
            return None
        ctype = raw.get("type")
        value = raw.get("value")
        if not isinstance(ctype, str) or not isinstance(value, str):
            return None
        ctype = ctype.strip().lower()[:32]
        value = " ".join(value.split())[:MAX_CONDITION_CHARS]
        if not ctype or not value:
            return None
        basis = raw.get("basis")
        return Condition(ctype, value, basis if isinstance(basis, str) else "")

    def key(self) -> tuple[str, str]:
        return (self.type, self.value.lower())


@dataclass(frozen=True)
class VerifyRecipe:
    cmd: str = ""
    expect: str = ""
    observed: str = ""

    def is_empty(self) -> bool:
        return not (self.cmd or self.expect or self.observed)

    def to_dict(self) -> dict:
        out = {}
        for name in ("cmd", "expect", "observed"):
            value = getattr(self, name)
            if value:
                out[name] = value
        return out

    @staticmethod
    def from_dict(raw: object) -> "VerifyRecipe | None":
        if not isinstance(raw, dict):
            return None
        values = {}
        for name in ("cmd", "expect", "observed"):
            value = raw.get(name)
            values[name] = value[:MAX_VERIFY_CHARS] if isinstance(value, str) else ""
        recipe = VerifyRecipe(**values)
        return recipe if not recipe.is_empty() else None


@dataclass(frozen=True)
class CasePacket:
    """The reusable unit: a historical repair, self-contained."""

    experience_id: str
    user_id: str = ""
    session_id: str = ""
    kind: str = "debug"
    problem: str = ""
    mechanism: str = ""
    conditions: tuple[Condition, ...] = ()
    files: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    locate: str = ""
    implement: str = ""
    implement_shape: str = ""
    implement_boundaries: str = ""
    implement_acceptance: str = ""
    patch: str = ""
    verify: VerifyRecipe = field(default_factory=VerifyRecipe)
    pitfalls: tuple[str, ...] = ()
    provenance: dict = field(default_factory=dict)
    created_at: str = ""
    source_chunk_ids: tuple[str, ...] = ()

    @property
    def stable_id(self) -> str:
        return f"experience:{self.experience_id}"

    def has_actionable_content(self) -> bool:
        return bool(self.problem or self.files or self.patch or self.verify.cmd or self.implement)

    # -- serialisation ----------------------------------------------------
    def to_payload(self) -> dict:
        payload = {
            "payload_version": CASE_PAYLOAD_VERSION,
            "experience_id": self.experience_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "kind": self.kind,
            "problem": self.problem,
            "conditions": [c.to_dict() for c in self.conditions],
            "files": list(self.files),
            "symbols": list(self.symbols),
            "locate": self.locate,
            "implement": self.implement,
            "patch": self.patch,
            "verify": self.verify.to_dict(),
            "pitfalls": list(self.pitfalls),
            "provenance": dict(self.provenance),
            "created_at": self.created_at,
            "source_chunk_ids": list(self.source_chunk_ids),
        }
        return payload

    @staticmethod
    def from_payload(raw: dict) -> "CasePacket":
        conditions = tuple(
            c for c in (Condition.from_dict(v) for v in raw.get("conditions") or []) if c
        )
        files = tuple(v for v in raw.get("files") or [] if isinstance(v, str) and v)
        symbols = tuple(v for v in raw.get("symbols") or [] if isinstance(v, str) and v)
        pitfalls = tuple(
            v for v in raw.get("pitfalls") or [] if isinstance(v, str) and v
        )
        source_ids = tuple(
            v for v in raw.get("source_chunk_ids") or [] if isinstance(v, str) and v
        )
        provenance = raw.get("provenance")
        return CasePacket(
            experience_id=str(raw.get("experience_id", "")),
            user_id=str(raw.get("user_id", "")),
            session_id=str(raw.get("session_id", "")),
            kind=str(raw.get("kind", "debug")),
            problem=_clean(str(raw.get("problem") or "")),
            conditions=conditions,
            files=files,
            symbols=symbols,
            locate=_clean(str(raw.get("locate") or "")),
            implement=_clean(str(raw.get("implement") or "")),
            patch=str(raw.get("patch") or ""),
            verify=VerifyRecipe.from_dict(raw.get("verify")) or VerifyRecipe(),
            pitfalls=pitfalls,
            provenance=provenance if isinstance(provenance, dict) else {},
            created_at=str(raw.get("created_at") or ""),
            source_chunk_ids=source_ids,
        )


# --------------------------------------------------------------------------
# deterministic extraction
# --------------------------------------------------------------------------

_DIFF_GIT = re.compile(r"^diff --git a/(\S+) b/(\S+)\s*$", re.M)
_DIFF_PLUS = re.compile(r"^\+\+\+ b/(\S+)\s*$", re.M)
_DEF_LINE = re.compile(r"^\+\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
_CLASS_LINE = re.compile(r"^\+\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
_CMD_LINE = re.compile(r"^\s*(?:\$|>|PS [A-Z]:)\s*(\S.*)$", re.M)
_TEST_HINT = re.compile(
    r"\b(pytest|py\.test|python\s+-m\s+pytest|python\s+\S*(?:test|repro|check)\S*\.py|"
    r"npm\s+(?:run\s+)?test|yarn\s+test|go\s+test|cargo\s+test|mvn\s|gradle\s|tox|nox|"
    r"make\s+test|unittest)\b",
    re.I,
)
_RESULT_LINE = re.compile(
    r"(?m)^\s*(?:=+\s*)?(\d+\s+(?:passed|failed|error|errors|skipped)\b[^\n]*)$"
)
_SCRIPT_RESULT = re.compile(r"(?m)^(?:SUCCESS|PASSED|OK)\b[^\n]{0,120}$")
_ERROR_NAME = re.compile(r"\b([A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning))\b")
_TEST_NODE = re.compile(r"\b((?:[\w./-]+\.py)::[A-Za-z0-9_\[\]-]+)\b")
_FAILED_TEST = re.compile(r"(?m)^\s*FAILED\s+(\S+)")
_TRACEBACK_FRAME = re.compile(r'File "([^"\n]+)", line (\d+)')
_VERSION_NEAR = re.compile(r"\b(?:version|v)\s*([0-9]+(?:\.[0-9]+){1,2})\b", re.I)
_MODULE_PATH = re.compile(r"^([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")
_TOOL_PATH = re.compile(r'"(?:file_path|path|file)"\s*:\s*"([^"\n]{3,200})"')
_TOOL_EDIT = re.compile(r"\[工具调用\]\s*Edit\s+(\{.*\})")
_TOOL_BASH = re.compile(r'"command"\s*:\s*"((?:[^"\\]|\\.)*)"')
_REPO_EXTS = frozenset(
    {
        "py", "pyi", "js", "jsx", "ts", "tsx", "go", "rs", "java", "kt", "rb",
        "php", "c", "h", "cc", "cpp", "hpp", "cs", "swift", "scala", "sh",
        "sql", "json", "yaml", "yml", "toml", "ini", "cfg", "conf", "gradle",
    }
)

# Paths that appear in a trajectory but are not part of the repository fix:
# virtualenvs, the harness testbed prefix, agent scratch tests, editor junk.
_SCRATCH_MARKERS = (
    "site-packages/",
    "dist-packages/",
    "miniconda",
    "anaconda",
    "/lib/python",
    "/home/",
    "home/",
    ".ds_store",
    "manual.yaml",
)
_SCRATCH_BASENAMES = frozenset(
    {"test_issue.py", "conftest_issue.py", "test_comprehensive.py",
     "reproduce_issue.py", "repro.py"}
)
# Historical trajectories name the repo under a harness prefix.  Strip it so
# the packet hands the agent a repository-relative path it can `read` directly.
_PATH_PREFIXES = ("swebench_9_15/testbed/", "swebench_9_15/", "testbed/")


def _clean(text: str) -> str:
    return " ".join(text.split())


def _norm_path(value: str) -> str:
    value = value.strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    if value.startswith(("a/", "b/")):
        value = value[2:]
    low = value.lower()
    for prefix in _PATH_PREFIXES:
        index = low.find(prefix)
        if index != -1:
            value = value[index + len(prefix):]
            low = value.lower()
            break
    return value.strip("/")


def _is_diff_chunk(chunk: StoredChunk) -> bool:
    """True only for a real unified diff, not for `ls -l`-looking output.

    The chunker classifies any line starting with `-` or `+` as a diff line, so
    an `ls -l` listing (`-rw-r--r-- ...`) used to be treated as a patch.  A real
    patch must carry either a `diff --git` header or a `---`/`+++` pair plus a
    `@@` hunk marker.
    """
    text = chunk.text
    if "diff --git " in text:
        return True
    has_headers = re.search(r"(?m)^--- .+\n\+\+\+ .+", text) is not None
    has_hunk = re.search(r"(?m)^@@ ", text) is not None
    return has_headers and has_hunk


def _is_scratch_path(path: str) -> bool:
    low = path.lower().replace("\\", "/")
    base = low.rsplit("/", 1)[-1]
    if base in _SCRATCH_BASENAMES:
        return True
    # Agent-created root-level test scripts (`test_simple.py`, `test_fix.py`);
    # repository tests live under a package directory and are kept.
    if "/" not in low and re.match(r"test_.*\.py$", base):
        return True
    return any(marker in low for marker in _SCRATCH_MARKERS)


_PATCH_FILE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)


def files_from_patch(patch: str, limit: int = MAX_FILES) -> tuple[str, ...]:
    """Files named by the historical patch itself."""
    found: list[str] = []
    for match in _PATCH_FILE.finditer(patch or ""):
        for value in (match.group(1), match.group(2)):
            path = _norm_path(value)
            if path and not _is_scratch_path(path) and path not in found:
                found.append(path)
    return tuple(found[:limit])


def _patch_from_edit_tools(chunks: Sequence[StoredChunk], max_lines: int) -> str:
    """Synthesize a readable change snippet from `Edit` tool calls.

    Trajectories often contain search/replace edits instead of a unified diff.
    The packet should still carry *what changed*: old line(s) and new line(s),
    with the exact path, rather than a prose summary.
    """
    blocks: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for chunk in chunks:
        for match in _TOOL_EDIT.finditer(chunk.text):
            try:
                payload = json.loads(match.group(1))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            raw_path = payload.get("file_path") or payload.get("path")
            old = payload.get("old_string") or payload.get("old_str") or ""
            new = payload.get("new_string") or payload.get("new_str") or ""
            if not isinstance(raw_path, str) or not isinstance(old, str) or not isinstance(new, str):
                continue
            path = _norm_path(raw_path)
            if not path or _is_scratch_path(path) or not (old.strip() or new.strip()):
                continue
            key = (path, old, new)
            if key in seen:
                continue
            seen.add(key)
            block = [f"--- a/{path}", f"+++ b/{path}", "@@"]
            block.extend(f"-{line}" for line in old.splitlines() or [])
            block.extend(f"+{line}" for line in new.splitlines() or [])
            blocks.extend(block)
            if len(blocks) >= max_lines:
                blocks.append(f"... [diff 截断，共 {len(blocks)} 行]")
                return "\n".join(blocks[:max_lines])
    return "\n".join(blocks)


_HUNK_HEADER = re.compile(
    r"^@@\s+-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@\s*(.*)$"
)


def _change_anchor(removed: list[str]) -> str:
    """A distinctive line the agent can grep for in the current file.

    The hunk header is the best location signal, but edits synthesized from a
    trajectory's Edit tool call carry only ``@@``.  The removed side still
    contains the exact text the historical change matched, so it can serve as a
    search anchor even when no line number exists.
    """
    candidates = [line.strip() for line in removed if line.strip()]
    for cand in candidates:
        if len(cand) >= 8 and not cand.startswith(("#", "//", "/*", "*")):
            return cand[:120]
    return candidates[0][:120] if candidates else ""


def format_change_for_agent(patch: str, max_chars: int = 1400) -> str:
    """Turn a unified/diff-like patch into before/after blocks.

    Weak models copy raw ``+`` lines verbatim, including their indentation, into
    unrelated contexts.  The same information is safer as an explicit
    before/after pair with a location label: the model still sees the change,
    but cannot mistake the ``+`` marker for literal code.

    The location is load-bearing.  A tool-loop agent often reads a large file
    truncated (head + tail), so a packet that says only "this file" sends it to
    the first plausible edit in the visible region; it must also get the hunk's
    start line or an exact line to search for.
    """
    if not patch.strip():
        return ""
    sections = re.split(r"(?m)^(?=diff --git )", patch)
    chunks: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        lines = section.splitlines()
        path = ""
        match = re.search(r"^diff --git a/(\S+) b/(\S+)", section, re.M)
        if match:
            path = _norm_path(match.group(2))
        hunk: list[str] = []
        removed: list[str] = []
        added: list[str] = []
        position = ""
        start_line = 0

        def flush() -> None:
            if not removed and not added:
                return
            head = [f"文件：{path or '(unknown)'}"]
            location = []
            if start_line:
                location.append(f"第 {start_line} 行")
            if position:
                location.append(position)
            if location:
                head.append("位置：" + " · ".join(location))
            anchor = _change_anchor(removed)
            if anchor and anchor not in position:
                head.append(f"定位锚点（在目标文件中检索这一行）：{anchor}")
            head.append("修改前：")
            head.extend(removed if removed else ["(无)"])
            head.append("修改后：")
            head.extend(added if added else ["(无)"])
            chunks.append("\n".join(head))

        for line in lines:
            if line.startswith("diff --git ") or line.startswith("index "):
                continue
            if line.startswith("--- a/") and not path:
                path = _norm_path(line[6:])
                continue
            if line.startswith("--- ") or line.startswith("+++ "):
                continue
            if line.startswith("@@"):
                flush()
                removed, added = [], []
                header = _HUNK_HEADER.match(line)
                if header:
                    start_line = int(header.group(1))
                    position = header.group(3).strip()[:120]
                else:
                    start_line = 0
                    # A synthesized header may carry an anchor (`@@ dtype_orig
                    # = ...`); plain `@@` carries nothing.
                    position = line[2:].strip().lstrip("@").strip()[:120]
                continue
            if not hunk:
                hunk.append(line)
            if line.startswith("+"):
                added.append(line[1:])
            elif line.startswith("-"):
                removed.append(line[1:])
            elif line.startswith(" ") or not line:
                # Context is implicit in before/after; dropping it keeps the
                # blocks compact.  The hunk header carries the location.
                pass
        flush()
    text = "\n\n".join(chunks).strip()
    if not text:
        return ""
    return text[:max_chars] + ("\n...[改法截断]" if len(text) > max_chars else "")


def _diff_sections(text: str) -> list[str]:
    """Split a text blob into diff sections, keeping only real diffs."""
    sections = re.split(r"(?m)^(?=diff --git )", text)
    out: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if not (section.startswith("diff --git ") or re.search(r"(?m)^--- .+\n\+\+\+ ", section)):
            continue
        match = re.search(r"^diff --git a/(\S+) b/(\S+)", section, re.M)
        if match and _is_scratch_path(_norm_path(match.group(2))):
            continue
        # Cut trailing prose/test output that followed the patch in the same
        # message.  Once a hunk has started, only diff-shaped lines belong.
        lines = section.splitlines()
        kept: list[str] = []
        in_hunk = False
        for line in lines:
            if line.startswith("@@"):
                in_hunk = True
            if in_hunk and line and not re.match(r"^[ +\-@\\]", line):
                break
            kept.append(line)
        out.append("\n".join(kept).rstrip())
    return out


def extract_patch(chunks: Sequence[StoredChunk], max_lines: int = MAX_PATCH_LINES) -> str:
    """Historical change, verbatim when a diff exists, else from edit tools.

    Diffs are reconstructed **per message**: the chunker splits a patch into
    several chunks, and a hunk-body chunk does not itself contain ``diff --git``
    or ``@@``.  Grouping by message keeps the body attached to its header while
    still rejecting an ``ls -l`` listing (which has no header/hunk in its
    message).
    """
    by_message: dict[str, list[StoredChunk]] = {}
    for chunk in chunks:
        by_message.setdefault(chunk.message_id, []).append(chunk)

    sections: list[str] = []
    seen: set[str] = set()
    for group in by_message.values():
        text = "\n".join(c.text.strip("\n") for c in sorted(group, key=lambda c: c.ord))
        if "diff --git " not in text and not re.search(r"(?m)^@@ ", text):
            continue
        for section in _diff_sections(text):
            key = re.sub(r"\s+", " ", section)[:500]
            if key in seen:
                continue
            seen.add(key)
            sections.append(section)

    if not sections:
        text = _patch_from_edit_tools(chunks, max_lines)
    else:
        text = "\n".join(sections)
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        kept = lines[:max_lines]
        kept.append(f"... [diff 截断，共 {len(lines)} 行]")
        text = "\n".join(kept)
    return text


def extract_files(chunks: Sequence[StoredChunk], limit: int = MAX_FILES) -> tuple[str, ...]:
    """Exact paths touched by the historical change, from diff headers."""
    found: list[str] = []
    for chunk in chunks:
        if not _is_diff_chunk(chunk):
            continue
        for match in _DIFF_GIT.finditer(chunk.text):
            for value in (match.group(1), match.group(2)):
                path = _norm_path(value)
                if path and not _is_scratch_path(path) and path not in found:
                    found.append(path)
        for match in _DIFF_PLUS.finditer(chunk.text):
            path = _norm_path(match.group(1))
            if path and not _is_scratch_path(path) and path not in found:
                found.append(path)
    if not found:
        # Trajectories frequently carry tool calls instead of a unified diff:
        # `[工具调用] Read {"file_path": "src/a.py"}`.  Those paths are the only
        # navigation hint the agent gets, so index them as a fallback.
        for chunk in chunks:
            for match in _TOOL_PATH.finditer(chunk.text):
                path = _norm_path(match.group(1))
                ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
                if "/" not in path and "\\" not in path:
                    continue
                if ext not in _REPO_EXTS or _is_scratch_path(path):
                    continue
                if path not in found:
                    found.append(path)
                if len(found) >= limit:
                    break
            if len(found) >= limit:
                break
    return tuple(found[:limit])


def extract_symbols(
    chunks: Sequence[StoredChunk],
    limit: int = MAX_SYMBOLS,
) -> tuple[str, ...]:
    """Symbols introduced or modified by the historical diff."""
    found: list[str] = []
    for chunk in chunks:
        if not (_is_diff_chunk(chunk) or chunk.kind in ("code", "command")):
            continue
        for pattern in (_DEF_LINE, _CLASS_LINE):
            for match in pattern.finditer(chunk.text):
                value = match.group(1)
                if value not in found:
                    found.append(value)
        if len(found) >= limit:
            break
    return tuple(found[:limit])


def _is_scratch_command(cmd: str) -> bool:
    low = cmd.lower().replace("\\", "/")
    for base in _SCRATCH_BASENAMES:
        if base in low:
            return True
    # Root-level `test_*.py` scripts are agent scratch, not repository tests.
    return bool(re.search(r"(?:^|[\s;&|])test_[a-z0-9_]+\.py\b", low)) and "tests/" not in low


def extract_verify(chunks: Sequence[StoredChunk]) -> VerifyRecipe:
    """The final command and observable result the history actually used."""
    cmd = ""
    for chunk in chunks:
        if chunk.kind not in ("command", "test_result", "code", "log"):
            continue
        for match in _CMD_LINE.finditer(chunk.text):
            candidate = _clean(match.group(1))[:MAX_VERIFY_CHARS]
            if _TEST_HINT.search(candidate) and not _is_scratch_command(candidate):
                cmd = candidate
    if not cmd:
        # A bare command chunk (no prompt prefix) is still a verification hint.
        for chunk in chunks:
            if chunk.kind != "command":
                continue
            first = next((ln.strip() for ln in chunk.text.splitlines() if ln.strip()), "")
            if first and _TEST_HINT.search(first) and not _is_scratch_command(first):
                cmd = first[:MAX_VERIFY_CHARS]
    if not cmd:
        # Tool-call form: `[工具调用] Bash {"command": "pytest ..."}`.  The
        # command is JSON-escaped, so decode the string rather than regexing
        # newlines and quotes by hand.
        for chunk in chunks:
            for match in _TOOL_BASH.finditer(chunk.text):
                try:
                    candidate = json.loads(f'"{match.group(1)}"')
                except Exception:
                    candidate = match.group(1).replace('\\"', '"')
                candidate = _clean(str(candidate))[:MAX_VERIFY_CHARS]
                if _TEST_HINT.search(candidate) and not _is_scratch_command(candidate):
                    cmd = candidate

    if not cmd:
        # Test-centric trajectories often show a failing node id without the
        # command that produced it.  Reconstruct the minimal pytest invocation.
        for chunk in chunks:
            for pattern in (_TEST_NODE, _FAILED_TEST):
                match = pattern.search(chunk.text)
                if not match:
                    continue
                node = _clean(match.group(1))
                node_path, _, node_name = node.partition("::")
                node = f"{_norm_path(node_path)}::{node_name}" if node_path else node
                if not node or _is_scratch_path(node.split("::", 1)[0]):
                    continue
                cmd = f"pytest {node}"[:MAX_VERIFY_CHARS]

    def _result_lines(kinds: tuple[str, ...]) -> list[str]:
        lines: list[str] = []
        for chunk in chunks:
            if kinds and chunk.kind not in kinds:
                continue
            for match in _RESULT_LINE.finditer(chunk.text):
                lines.append(_clean(match.group(1))[:MAX_VERIFY_CHARS])
            for match in _SCRIPT_RESULT.finditer(chunk.text):
                lines.append(_clean(match.group(0))[:MAX_VERIFY_CHARS])
        return lines

    def _is_success(line: str) -> bool:
        low = line.lower()
        if "failed" in low or "error" in low:
            return False
        return any(marker in low for marker in ("passed", "ok", "success"))

    results = _result_lines(("test_result", "command", "log"))
    if not results:
        # Fallback: the result line may sit in a prose tool-output chunk.
        results = _result_lines(())
    successes = [line for line in results if _is_success(line)]
    expect = successes[-1] if successes else (results[-1] if results else "")
    observed = expect
    if not cmd:
        # A result without a runnable command is not a verification recipe.
        return VerifyRecipe()
    return VerifyRecipe(cmd=cmd, expect=expect, observed=observed)


def derive_conditions(
    chunks: Sequence[StoredChunk],
    files: Sequence[str],
    problem: str = "",
    extra: Iterable[Condition] = (),
    limit: int = MAX_CONDITIONS,
    source_ids: Sequence[str] = (),
) -> tuple[Condition, ...]:
    """Facts that constrain whether this experience transfers.

    These are conditions, not verdicts: the query-time judge decides whether
    the current issue actually meets them (or leaves them unknown).
    """
    out: list[Condition] = []
    seen: set[tuple[str, str]] = set()

    def add(condition: Condition) -> None:
        key = condition.key()
        if key in seen or len(out) >= limit:
            return
        seen.add(key)
        out.append(condition)

    for condition in extra:
        add(condition)

    for path in files[:3]:
        match = _MODULE_PATH.match(path)
        if match:
            add(Condition("module", match.group(1), "diff"))

    version = _VERSION_NEAR.search(problem or "")
    if version:
        add(Condition("version", version.group(1), "problem"))

    # Official trajectories are test-heavy and diff-light: failing test node
    # ids and traceback frames are the transferable anchors there.  Prefer the
    # chunks the card actually cites; scanning the whole session would turn
    # unrelated environment errors into "conditions".
    wanted = set(source_ids)
    primary = [chunk for chunk in chunks if chunk.chunk_id in wanted] or list(chunks)
    tests: list[str] = []
    frames: list[str] = []
    for chunk in primary:
        for match in _TEST_NODE.finditer(chunk.text):
            value = match.group(1)
            node_path, _, node_name = value.partition("::")
            value = f"{_norm_path(node_path)}::{node_name}" if node_path else value
            if _is_scratch_path(value.split("::", 1)[0]):
                continue
            if value not in tests:
                tests.append(value)
        for match in _FAILED_TEST.finditer(chunk.text):
            value = _clean(match.group(1))
            node_path, _, node_name = value.partition("::")
            value = f"{_norm_path(node_path)}::{node_name}" if node_path else value
            if value and not _is_scratch_path(value.split("::", 1)[0]) and value not in tests:
                tests.append(value)
        for match in _TRACEBACK_FRAME.finditer(chunk.text):
            path = _norm_path(match.group(1))
            if not path or path.startswith("<") or _is_scratch_path(path):
                continue
            value = f"{path}:{match.group(2)}"
            if value not in frames:
                frames.append(value)
        if len(tests) >= 3 and len(frames) >= 2:
            break
    for value in tests[:3]:
        add(Condition("test_name", value, "chunk"))
    for value in frames[:2]:
        add(Condition("frame", value, "chunk"))

    problem_errors = [
        match.group(1) for match in _ERROR_NAME.finditer(problem or "")
    ]
    errors: list[str] = []
    for chunk in primary:
        for match in _ERROR_NAME.finditer(chunk.text):
            value = match.group(1)
            if problem_errors and value not in problem_errors:
                continue
            if value not in errors:
                errors.append(value)
            if len(errors) >= 3:
                break
        if len(errors) >= 3:
            break
    for value in errors:
        add(Condition("error", value, "chunk"))

    return tuple(out)


# --------------------------------------------------------------------------
# packet construction
# --------------------------------------------------------------------------


def _card_attr(card: object, name: str, default=None):
    return getattr(card, name, default)


def build_packet(
    card: object,
    chunks: Sequence[StoredChunk] = (),
    *,
    max_patch_lines: int = MAX_PATCH_LINES,
    allow_patch: bool = True,
    persisted: bool = False,
) -> CasePacket:
    """Merge model-written fields with deterministic extraction.

    Model fields win for problem/implement/pitfalls (it alone can phrase
    them); structural fields fall back to extraction so a packet is never a
    bare summary.

    ``persisted`` marks a card reloaded from the store, whose files/symbols/
    patch/verify were already derived at write time from the whole session and
    are what the search index was built from.  Search re-fetches only the cited
    chunks, so on that path the stored structure is authoritative and pool
    extraction is merely the fallback for a payload that lacks it.  A card on
    the write path never carries these fields
    (``experience._new_card_from_raw`` forces patch/verify to None), so the
    fallback cannot let a model-written diff bypass grounding.
    """
    problem = _clean(str(_card_attr(card, "problem", "") or ""))
    problem = problem[:MAX_FIELD_CHARS]
    implement = _clean(str(_card_attr(card, "implement", "") or ""))
    implement = implement[:MAX_FIELD_CHARS]
    source_ids = tuple(_card_attr(card, "source_chunk_ids", ()) or ())
    wanted_ids = set(source_ids)
    grounding_chunks = [
        chunk for chunk in chunks if chunk.chunk_id in wanted_ids
    ] or list(chunks)
    source_text = "\n".join(chunk.text for chunk in grounding_chunks).lower()

    def path_is_grounded(path: str) -> bool:
        normalized = _norm_path(path).lower()
        return bool(normalized) and normalized in source_text

    # Model-declared structure is accepted only when it occurs in the evidence;
    # a hallucinated path/symbol would send the agent to the wrong file.  A card
    # reloaded from the store already cleared that gate at write time and was
    # enriched from the whole session, so its persisted structure outranks
    # anything the cited-chunk subset can reproduce -- re-deriving from the
    # subset narrows the packet, and a subset that happens to contain a bare
    # ``diff --git`` header yields a hunk-less patch that hides the real one.
    declared_files = tuple(_card_attr(card, "files", ()) or ())
    stored_files = [_norm_path(path) for path in declared_files if persisted and path]
    clean_declared = [
        _norm_path(path)
        for path in declared_files
        if path and not _is_scratch_path(_norm_path(str(path)))
        and path_is_grounded(str(path))
    ]
    files = tuple(stored_files) or tuple(clean_declared) or extract_files(chunks)
    files = tuple(dict.fromkeys(_norm_path(p) for p in files if p))[:MAX_FILES]

    declared_symbols = tuple(_card_attr(card, "symbols", ()) or ())
    stored_symbols = tuple(s for s in declared_symbols if persisted and str(s).strip())
    symbols = stored_symbols or tuple(
        symbol for symbol in declared_symbols
        if str(symbol).strip() and str(symbol).lower() in source_text
    ) or extract_symbols(chunks)
    symbols = tuple(dict.fromkeys(s for s in symbols if s))[:MAX_SYMBOLS]

    # The historical diff is the only accepted patch. A model-written diff
    # would bypass path grounding and could invent files; ``persisted`` is set
    # only on the read path, where the diff was derived from real chunks and
    # stored by our own write path.
    patch = str(_card_attr(card, "patch", "") or "") if persisted else ""
    if not patch:
        patch = extract_patch(chunks, max_patch_lines)
    if not allow_patch:
        patch = ""
    if patch:
        lines = patch.splitlines()
        if len(lines) > max_patch_lines:
            patch = "\n".join(lines[:max_patch_lines] + [f"... [diff 截断，共 {len(lines)} 行]"])
        # The patch is the authoritative statement of what changed; declared
        # or tool-call paths must not push a scratch test file ahead of it.
        patch_files = files_from_patch(patch)
        if patch_files:
            files = tuple(dict.fromkeys((*patch_files, *files)))[:MAX_FILES]

    # The extraction prompt does not ask for a verification recipe; only the
    # deterministic command/result from real chunks is trustworthy.
    verify = _card_attr(card, "verify", None) if persisted else None
    if verify is None or verify.is_empty():
        verify = extract_verify(chunks)

    pitfalls = tuple(
        _clean(str(p))[:MAX_PITFALL_CHARS]
        for p in (_card_attr(card, "pitfalls", ()) or ())
        if str(p).strip()
    )[:MAX_PITFALLS]

    conditions = tuple(_card_attr(card, "conditions", ()) or ())
    conditions = derive_conditions(
        chunks, files, problem, conditions, source_ids=source_ids,
    )

    provenance = dict(_card_attr(card, "provenance", {}) or {})
    provenance.setdefault("session", str(_card_attr(card, "session_id", "") or ""))
    if _card_attr(card, "created_at", ""):
        provenance.setdefault("created_at", str(_card_attr(card, "created_at")))
    if source_ids:
        provenance.setdefault("chunk_count", len(source_ids))
        provenance.setdefault("chunks", list(source_ids[:3]))

    locate = _clean(str(_card_attr(card, "locate", "") or ""))
    if not locate:
        locate = "；".join(dict.fromkeys((*files, *symbols)))
    locate = locate[:MAX_FIELD_CHARS]

    return CasePacket(
        experience_id=str(_card_attr(card, "experience_id", "") or ""),
        user_id=str(_card_attr(card, "user_id", "") or ""),
        session_id=str(_card_attr(card, "session_id", "") or ""),
        kind=str(_card_attr(card, "kind", "debug") or "debug"),
        problem=problem,
        mechanism=str(_card_attr(card, "mechanism", "") or "")[:MAX_FIELD_CHARS],
        conditions=conditions,
        files=files,
        symbols=symbols,
        locate=locate,
        implement=implement,
        implement_shape=str(_card_attr(card, "implement_shape", "") or "")[:MAX_FIELD_CHARS],
        implement_boundaries=str(_card_attr(card, "implement_boundaries", "") or "")[:MAX_FIELD_CHARS],
        implement_acceptance=str(_card_attr(card, "implement_acceptance", "") or "")[:MAX_FIELD_CHARS],
        patch=patch,
        verify=verify,
        pitfalls=pitfalls,
        provenance=provenance,
        created_at=str(_card_attr(card, "created_at", "") or ""),
        source_chunk_ids=source_ids,
    )




# --------------------------------------------------------------------------
# rendering (self-contained, budget aware)
# --------------------------------------------------------------------------


def _header(packet: CasePacket) -> str:
    repo = str(packet.provenance.get("repo") or "")
    issue = str(packet.provenance.get("issue") or "")
    commit = str(packet.provenance.get("commit") or "")
    where = " ".join(v for v in (repo, issue) if v)
    tail = f" · {packet.kind}"
    if where:
        tail += f" · {where}"
    if commit:
        tail += f" @ {commit}"
    return f"[历史修复案例]{tail}"


def _condition_line(packet: CasePacket) -> str:
    # ``test_name`` conditions render in their own 测试 line; repeating them
    # here would show the same anchor twice.
    shown = [c for c in packet.conditions if c.type != "test_name"]
    if not shown:
        return ""
    parts = [f"{c.type}={c.value}" for c in shown]
    return "条件：" + " · ".join(parts)


def _verify_lines(packet: CasePacket) -> list[str]:
    if packet.verify.is_empty():
        return []
    lines = []
    if packet.verify.cmd:
        lines.append(f"验证：`{packet.verify.cmd}`")
    result = packet.verify.observed or packet.verify.expect
    if result:
        lines.append(f"结果：{result}")
    return lines


def _provenance_line(
    packet: CasePacket,
    include_chunk_provenance: bool = True,
) -> str:
    # Chunk ids stay in the stored payload/provenance for audit.  The answer
    # render omits them: the packet is self-contained and the answering agent
    # cannot dereference a chunk id.
    parts = []
    commit = packet.provenance.get("commit")
    if commit:
        parts.append(str(commit))
    if include_chunk_provenance:
        chunks = packet.provenance.get("chunks") or []
        count = packet.provenance.get("chunk_count") or len(packet.source_chunk_ids)
        if chunks:
            shown = ", ".join(str(c) for c in list(chunks)[:2])
            suffix = f"（{count}）" if count else ""
            parts.append(f"chunks {shown}{suffix}")
        elif packet.session_id:
            parts.append(packet.session_id)
    return f"来源：{' · '.join(parts)}" if parts else ""


def _historicalize(text: str, files: Sequence[str]) -> str:
    """Downgrade historical paths/test names inside card prose.

    The gate used to suppress only the patch block, leaving the problem statement and
    test names as if they were current facts.  On a sibling-file task that is
    exactly the negative-transfer signal; when the gate rejects the change we
    keep the information but label it as historical.
    """
    if not text:
        return text
    out = text
    for path in files:
        path = str(path).strip()
        if not path:
            continue
        out = out.replace(path, "（历史文件）")
        base = path.replace("\\", "/").rsplit("/", 1)[-1]
        if base and base != path:
            out = out.replace(base, "（历史文件）")
    out = re.sub(r"\b[\w./-]+\.py::test_[A-Za-z0-9_\[\]-]+", "（历史测试）", out)
    out = re.sub(r"\btest_[A-Za-z0-9_]{2,80}\b", "（历史测试）", out)
    return out


def _patch_summary(packet: CasePacket) -> str:
    """A reference-only description of the historical change.

    Deliberately emits no ``+/-`` code: the answer model gets the location,
    symbols and semantic delta, but must implement the edit against the
    current file itself.
    """
    files = ", ".join(packet.files) or "(未标注文件)"
    patch = packet.patch or ""
    positions: list[str] = []
    changed: list[str] = []
    for line in patch.splitlines():
        if line.startswith("@@"):
            header = re.sub(r"^@@[^@]*@@\s*", "", line).strip()
            if header and header not in positions:
                positions.append(header[:100])
        elif line[:1] in ("+", "-") and not line.startswith(("+++", "---")):
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", line[1:]):
                if token not in changed and token not in {"None", "True", "False"}:
                    changed.append(token)
    parts = [f"涉及文件：{files}"]
    if positions:
        parts.append("改动位置：" + "；".join(positions[:4]))
    if changed:
        parts.append("变更涉及的标识符：" + ", ".join(changed[:12]))
    if packet.symbols:
        parts.append("符号：" + ", ".join(packet.symbols[:8]))
    parts.append("只提供结构与语义摘要，不提供可复制代码；请在当前仓库自行实现。")
    return "\n".join(parts)


def render_packet(packet: CasePacket, counter, max_tokens: int,
                  evidence: Sequence[str] = (), allow_patch: bool = True,
                  first_action_hint: str = "", patch_render: str = "before_after",
                  include_chunk_provenance: bool = True) -> str:
    """Render one self-contained packet under a hard token budget.

    Sections are dropped in reverse priority when space runs out; the patch is
    shortened before the problem statement is, because ``where/what changed``
    is the part the agent cannot reconstruct.
    """
    if max_tokens <= 0 or not packet.has_actionable_content():
        return ""

    core: list[str] = [_header(packet)]
    if first_action_hint:
        core.append(first_action_hint)
    if packet.problem:
        if allow_patch:
            core.append(f"问题：{packet.problem}")
        else:
            core.append("问题（历史线索，需在当前仓库确认）：" + _historicalize(packet.problem, packet.files))
    if packet.mechanism:
        label = "机制：" if allow_patch else "历史机制理解（仅供参考）："
        core.append(label + packet.mechanism)
    if packet.files and allow_patch:
        core.append("文件：" + ", ".join(packet.files))
    tests = [c.value for c in packet.conditions if c.type == "test_name"]
    if packet.locate:
        label = "定位：" if allow_patch else "历史定位线索（请先在当前仓库确认）："
        locate = packet.locate if allow_patch else _historicalize(packet.locate, packet.files)
        core.append(label + locate)
    if not allow_patch:
        anchors = list(dict.fromkeys(
            [*packet.symbols, *tests, *(c.value for c in packet.conditions if c.type == "error")]
        ))[:6]
        if anchors:
            core.append(
                "确认方式：在当前仓库检索 " + "、".join(anchors)
                + "，确认它们现在所在的文件后再编辑；不要直接使用历史文件。"
            )
    if packet.implement:
        implement = packet.implement if allow_patch else _historicalize(packet.implement, packet.files)
        prefix = "改法：" if allow_patch else "历史改法思路（仅供参考）："
        core.append(prefix + implement)
    if packet.implement_shape or packet.implement_boundaries or packet.implement_acceptance:
        spec_prefix = "验收过的修复规格" if allow_patch else "历史修复规格（需按当前 issue 校准）："
        core.append(spec_prefix)
        for label, value in (("形态", packet.implement_shape),
                             ("边界", packet.implement_boundaries),
                             ("验收", packet.implement_acceptance)):
            if value:
                core.append(f"  {label}：{value}")
    tests = [c.value for c in packet.conditions if c.type == "test_name"]
    if tests:
        label = "测试：" if allow_patch else "历史测试（需在当前仓库确认）："
        core.append(label + ", ".join(tests))
    change_text = ""
    if packet.patch and allow_patch and patch_render != "off":
        if patch_render == "summary":
            core.append(
                "历史改法（仅作参考，不提供可复制代码；请基于当前仓库自行实现）：\n"
                + _patch_summary(packet)
            )
        else:
            change_text = format_change_for_agent(packet.patch)
            if change_text:
                if "定位锚点" in change_text:
                    hint = ("历史改法（参考；先在目标文件中检索下方锚点，确认存在相同代码后，"
                            "再改写为当前文件的 SEARCH/REPLACE，禁止整段粘贴）：")
                else:
                    hint = "历史改法（参考；请改写为当前文件的 SEARCH/REPLACE，禁止整段粘贴）："
                core.append(hint + "\n" + change_text)
    core.extend(_verify_lines(packet))

    optional: list[str] = []
    cond = _condition_line(packet)
    if cond:
        optional.append(cond)
    if packet.symbols:
        optional.append("符号：" + ", ".join(packet.symbols))
    for snippet in evidence:
        optional.append("相关证据：" + snippet.replace("\n", " ⏎ ")[:400])
    for pitfall in packet.pitfalls:
        optional.append(f"坑：{pitfall}")
    provenance = _provenance_line(packet, include_chunk_provenance)
    if provenance:
        optional.append(provenance)

    def build(with_optional: list[str]) -> str:
        return "\n".join(core + with_optional)

    text = build(optional)
    while optional and counter.count(text) > max_tokens:
        optional.pop()
        text = build(optional)

    # Drop the reference change before touching the problem: a partial patch is
    # worse than none, while files/verify still tell the agent where to look.
    if counter.count(text) > max_tokens and change_text:
        core = [line for line in core if not line.startswith("历史改法（参考")]
        text = build(optional)

    if counter.count(text) > max_tokens and packet.problem:
        problem_line = next((ln for ln in core if ln.startswith("问题：")), None)
        if problem_line is not None:
            keep = len(problem_line)
            while keep > 12 and counter.count(text) > max_tokens:
                keep = max(12, keep - 24)
                shortened = problem_line[:keep].rstrip() + " …"
                core = [shortened if ln is problem_line else ln for ln in core]
                text = build(optional)

    if counter.count(text) > max_tokens:
        text = counter.truncate(text, max_tokens)
    return text.strip()


def packet_summary(packet: CasePacket, max_chars: int = 700) -> str:
    """Compact view for the listwise judge (never the full diff)."""
    lines = [f"id={packet.stable_id}"]
    if packet.problem:
        lines.append(f"问题：{packet.problem[:200]}")
    cond = _condition_line(packet)
    if cond:
        lines.append(cond[:240])
    if packet.files:
        lines.append("文件：" + ", ".join(packet.files[:6]))
    if packet.symbols:
        lines.append("符号：" + ", ".join(packet.symbols[:8]))
    if packet.implement:
        lines.append(f"改法：{packet.implement[:200]}")
    if packet.implement_acceptance:
        lines.append(f"验收：{packet.implement_acceptance[:160]}")
    if packet.verify.cmd:
        lines.append(f"验证：{packet.verify.cmd[:160]}")
    return "\n".join(lines)[:max_chars]
