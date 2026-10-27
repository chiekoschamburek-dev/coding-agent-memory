"""Structured chunking and exact entity extraction.

plan.md section 5/A3: split on paragraph, code fence, diff, stack trace, command
output and test boundaries; keep operation/result adjacency; never lose the
exact paths, symbols, error strings, config keys and test names.

The chunker is deliberately structure-first and lossless: every character of
the source message ends up in exactly one chunk, so the concatenation of a
message's chunks reproduces the original string. Overlap between chunks is
deliberately *not* used at storage level -- overlapping text would break that
invariant and double-count in BM25. Adjacency is preserved through the `ord`
column and ``Store.neighbour_chunks`` instead, which is the "bounded adjacent
expansion" plan.md asks for.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .config import Settings, get_settings
from .store import ChunkRecord, EntityRecord, MessageInput, chunk_id_for, message_id_for, tokenize

KIND_PROSE = "prose"
KIND_CODE = "code"
KIND_DIFF = "diff"
KIND_STACK = "stack"
KIND_LOG = "log"
KIND_COMMAND = "command"
KIND_TEST = "test_result"

# Rough importance ordering, used only to break ties when a chunk would be
# truncated by the token budget.
KIND_PRIORITY = {
    KIND_DIFF: 0,
    KIND_CODE: 1,
    KIND_STACK: 2,
    KIND_TEST: 3,
    KIND_COMMAND: 4,
    KIND_LOG: 5,
    KIND_PROSE: 6,
}


@dataclass(frozen=True)
class Segment:
    kind: str
    text: str
    start: int
    end: int


# --------------------------------------------------------------------------
# entity extraction (exact, rule-based -- no model)
# --------------------------------------------------------------------------

_ENTITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Paths: require a separator or a known extension so prose is not captured.
    (
        "path",
        re.compile(
            r"(?<![\w./\\-])"
            r"(?:[A-Za-z]:)?(?:\.{0,2}[/\\])?"
            r"(?:[\w.+-]+[/\\])*[\w.+-]+\.(?:py|pyi|js|jsx|mjs|cjs|ts|tsx|go|rs|java|kt|"
            r"rb|php|c|h|cc|cpp|hpp|cs|swift|scala|sh|bash|zsh|ps1|sql|json|ya?ml|toml|ini|"
            r"cfg|conf|env|md|txt|csv|xml|html|css|scss|vue|svelte|proto|gradle|mod|lock)"
            r"(?![\w./\\-])"
        ),
    ),
    (
        "path",
        re.compile(
            r"(?<![\w./\\-])"
            r"(?:[\w.+-]+[/\\]){1,}[\w.+-]+"
            r"(?![\w./\\-])"
        ),
    ),
    # Test identifiers: expect(/describe(/it(/test( and *_test / test_* names.
    (
        "test_name",
        re.compile(
            r"\b(?:expect|describe|it|test|Should|Verify|Assert)\(\s*[\"']([^\"'\n]{1,120})[\"']"
        ),
    ),
    ("test_name", re.compile(r"\b(?:def\s+)?(test_[A-Za-z0-9_]{2,80})\b")),
    ("test_name", re.compile(r"\b([A-Za-z0-9_]{2,80}_test)\b")),
    # Exception / error symbols: `FooError`, `FooException`, plus errno-ish codes.
    ("error", re.compile(r"\b([A-Z][A-Za-z0-9]*(?:Error|Exception|Warning|Fault))\b")),
    ("error", re.compile(r"\b((?:E|W|C|TS|HTTP|errno|ERR_)[A-Z0-9_]{2,20})\b")),
    (
        "error",
        re.compile(r"\b([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):\s*[^\n]{3,160})"),
    ),
    # Config keys: dotted.key.path or ALL_CAPS_ENV_STYLE.
    ("config_key", re.compile(r"\b([a-z][a-z0-9_]*(?:\.[a-z0-9_]+){1,6})\b")),
    ("config_key", re.compile(r"\b([A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)+)\b")),
)

# Symbol-ish identifiers are extracted from code-ish chunks only, to keep
# ordinary prose from contributing thousands of useless "symbols".
_SYMBOL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{3,60})\b")

_NOISE_SYMBOLS = frozenset(
    {
        "true", "false", "none", "null", "self", "this", "return", "import",
        "from", "class", "def", "function", "const", "let", "var", "public",
        "private", "static", "async", "await", "print", "printf", "error",
        "value", "string", "number", "object", "array", "result", "data",
        "test", "tests", "should", "assert", "expect", "describe", "https",
        "http", "localhost", "the", "and", "for", "with", "that",
        "readme", "license", "changelog", "contributing", "makefile",
        "dockerfile", "pyproject", "setup",
    }
)


def _norm_entity(value: str) -> str:
    return value.strip().strip("\"'`").strip()


def extract_entities(chunks: Sequence[ChunkRecord]) -> list[EntityRecord]:
    """Exact-entity index. Rule-based, deterministic, reproducible."""
    found: dict[tuple[str, str, str], EntityRecord] = {}
    for chunk in chunks:
        for kind, pattern in _ENTITY_PATTERNS:
            for match in pattern.finditer(chunk.text):
                value = _norm_entity(match.group(1) if match.groups() else match.group(0))
                if kind == "path":
                    value = value.rstrip(".,;:")
                if not value or len(value) > 200 or len(value) < 3:
                    continue
                if kind == "path" and not _looks_like_path(value):
                    continue
                if kind == "config_key" and value.lower() in _NOISE_SYMBOLS:
                    continue
                key = (kind, value.lower(), chunk.chunk_id)
                found.setdefault(
                    key, EntityRecord(chunk_id=chunk.chunk_id, kind=kind, value=value.lower())
                )
        if chunk.kind in (KIND_CODE, KIND_DIFF, KIND_COMMAND, KIND_STACK):
            for match in _SYMBOL_RE.finditer(chunk.text):
                value = match.group(1)
                low = value.lower()
                if low in _NOISE_SYMBOLS or len(low) < 4:
                    continue
                # Only identifiers with an intra-word case/underscore switch, or
                # that look like a real API name, are worth indexing.
                if not (any(ch.isupper() for ch in value[1:]) or "_" in value or value.isupper()):
                    continue
                key = ("symbol", low, chunk.chunk_id)
                found.setdefault(
                    key, EntityRecord(chunk_id=chunk.chunk_id, kind="symbol", value=low)
                )
    return list(found.values())


def _looks_like_path(value: str) -> bool:
    if "/" in value or "\\" in value:
        return True
    _, _, ext = value.rpartition(".")
    return 1 <= len(ext) <= 6 and ext.isalpha()


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")
_DIFF_HUNK_RE = re.compile(r"^[ \t]*(diff --git|@@ |--- |\+\+\+ |Index: )")
_DIFF_LINE_RE = re.compile(r"^[ \t]*[+-][^+-]")
_STACK_RE = re.compile(
    r"^\s*(?:Traceback \(most recent call last\)|"
    r"(?:File \"[^\"]+\", line \d+|at [\w.$<>]+\([^)]*\)|"
    r"\s+at .+:\d+:\d+)|"
    r"[A-Za-z_][\w.$]*(?:Error|Exception)[: ])"
)
_LOG_RE = re.compile(
    r"^\s*(?:\[?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}|"
    r"(?:INFO|DEBUG|WARN|WARNING|ERROR|TRACE|FATAL|CRITICAL)\b|"
    r"\d{2}:\d{2}:\d{2}\b)"
)
_COMMAND_RE = re.compile(r"^\s*(?:\$|>|PS [A-Z]:|#\s|\w+@[\w.-]+:.*[$#])\s*\S")
_TEST_LINE_RE = re.compile(
    r"^\s*(?:\d+\s+(?:passed|failed|skipped|error)|"
    r"=+\s*\d+\s+(?:passed|failed)|"
    r"(?:PASSED|FAILED|OK|FAIL)\b|"
    r"(?:Tests?|Test Suites?):\s*\d+)"
)

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_ROLE_HEADING_RE = re.compile(
    r"^\s{0,3}(?:User|Assistant|Human|AI|System|Q|A)\s*[:：]\s*$", re.IGNORECASE
)


def _classify_line(line: str, in_code: bool) -> str:
    if in_code:
        return KIND_CODE
    if _DIFF_HUNK_RE.match(line) or _DIFF_LINE_RE.match(line):
        return KIND_DIFF
    if _STACK_RE.match(line):
        return KIND_STACK
    if _TEST_LINE_RE.match(line):
        return KIND_TEST
    if _COMMAND_RE.match(line):
        return KIND_COMMAND
    if _LOG_RE.match(line):
        return KIND_LOG
    return KIND_PROSE


def segment_message(text: str, default_kind: str = KIND_PROSE) -> list[Segment]:
    """Split one message into typed, contiguous, non-overlapping segments.

    Contiguity is exact: ``"".join(segment.text)`` equals ``text``.
    """
    if not text:
        return []

    lines = text.splitlines(keepends=True)
    segments: list[Segment] = []
    buffer: list[tuple[str, int, int]] = []
    cursor = 0
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        if not buffer:
            return
        kind = buffer[0][0]
        start = buffer[0][1]
        end = buffer[-1][2]
        segments.append(Segment(kind=kind, text=text[start:end], start=start, end=end))
        buffer.clear()

    def push(kind: str, start: int, end: int) -> None:
        if buffer and buffer[0][0] != kind:
            flush()
        buffer.append((kind, start, end))

    for line in lines:
        line_start = cursor
        line_end = cursor + len(line)
        cursor = line_end

        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if not in_fence:
                flush()
                in_fence = True
                fence_marker = marker
                push(KIND_CODE, line_start, line_end)
            elif marker == fence_marker:
                push(KIND_CODE, line_start, line_end)
                flush()
                in_fence = False
                fence_marker = ""
            else:
                push(KIND_CODE, line_start, line_end)
            continue

        if in_fence:
            push(KIND_CODE, line_start, line_end)
            continue

        kind = _classify_line(line, in_code=False)
        if kind == KIND_PROSE and default_kind != KIND_PROSE:
            kind = default_kind
        if _HEADING_RE.match(line) or _ROLE_HEADING_RE.match(line):
            # A heading starts a fresh prose block, keeping topical units intact.
            flush()
        push(kind, line_start, line_end)

    flush()
    return segments


# --------------------------------------------------------------------------
# chunk assembly
# --------------------------------------------------------------------------


def _split_long_segment(segment: Segment, max_chars: int) -> list[Segment]:
    """Split an oversized segment on natural boundaries.

    Character ranges stay exact because every part is a slice of the parent.
    """
    if len(segment.text) <= max_chars:
        return [segment]

    parts: list[Segment] = []
    text = segment.text
    cursor = 0
    while cursor < len(text):
        remaining = len(text) - cursor
        if remaining <= max_chars:
            parts.append(
                Segment(segment.kind, text[cursor:], segment.start + cursor, segment.start + len(text))
            )
            break
        window_end = cursor + max_chars
        boundary = -1
        # Prefer a blank line, then a newline, then whitespace.
        for probe in ("\n\n", "\n", " "):
            idx = text.rfind(probe, cursor + max_chars // 2, window_end)
            if idx > boundary:
                boundary = idx + len(probe)
        if boundary <= cursor:
            boundary = window_end
        parts.append(
            Segment(segment.kind, text[cursor:boundary], segment.start + cursor, segment.start + boundary)
        )
        cursor = boundary
    return parts


def _merge_small_segments(segments: Sequence[Segment], target: int, minimum: int) -> list[Segment]:
    """Pack small adjacent same-kind segments up to ``target`` characters."""
    out: list[Segment] = []
    buf: list[Segment] = []

    def buf_len() -> int:
        return sum(len(s.text) for s in buf)

    def flush() -> None:
        if not buf:
            return
        out.append(
            Segment(
                kind=buf[0].kind,
                text="".join(s.text for s in buf),
                start=buf[0].start,
                end=buf[-1].end,
            )
        )
        buf.clear()

    for segment in segments:
        if not buf:
            buf.append(segment)
            continue
        same_kind = buf[0].kind == segment.kind
        # Prose merges freely; structural kinds only merge with themselves.
        if same_kind and buf_len() + len(segment.text) <= target:
            buf.append(segment)
        elif same_kind and buf_len() < minimum:
            # Under-filled structural block: absorb the next same-kind piece
            # rather than emit a stub.
            buf.append(segment)
        else:
            flush()
            buf.append(segment)
    flush()
    return out


def build_chunks(
    *,
    user_id: str,
    request_id: str,
    messages: Sequence[MessageInput],
    settings: Settings | None = None,
) -> list[ChunkRecord]:
    """Produce the chunk set for one Add request.

    Guarantees:
      * every message's chunks concatenate back to the original string;
      * chunk ids are stable across retries of the same request;
      * no cross-message chunk (a chunk never spans two messages), so
        operation/result association is never fabricated.
    """
    cfg = settings or get_settings()
    records: list[ChunkRecord] = []

    for message_index, message in enumerate(messages):
        message_id = message_id_for(user_id, request_id, message_index)
        segments = segment_message(message.content)
        expanded: list[Segment] = []
        for segment in segments:
            expanded.extend(_split_long_segment(segment, cfg.chunk_max_chars))
        packed = _merge_small_segments(expanded, cfg.chunk_target_chars, cfg.chunk_min_chars)

        for ord_, segment in enumerate(packed):
            chunk_id = chunk_id_for(user_id, request_id, message_index, ord_, segment.start, segment.end)
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    message_id=message_id,
                    message_index=message_index,
                    ord=ord_,
                    char_start=segment.start,
                    char_end=segment.end,
                    kind=segment.kind,
                    text=segment.text,
                    parent_chunk_id=None,
                    tokens=tuple(tokenize(segment.text)),
                )
            )
    return records


def content_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()