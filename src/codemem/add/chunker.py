"""Structure-aware chunking.

Memory arrives as ordered messages of free text that mix prose, code fences,
unified diffs, stack traces, logs, commands and config. Splitting naively by
size destroys exactly the signal the Coding track rewards, so segmentation is
driven by structure first and size second:

* fenced blocks are never split mid-line;
* a diff / stack trace / log burst / config stays in one piece;
* prose paragraphs are packed up to a target token budget.

The result is deterministic: identical input always yields identical chunks,
which is what makes reindexing and score reproduction safe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.tokens import count_tokens

# ---------------------------------------------------------------- kinds ----

PROSE = "prose"
CODE = "code"
DIFF = "diff"
LOG = "log"
STACKTRACE = "stacktrace"
CMD = "cmd"
CONFIG = "config"
TEST = "test"

# Kinds that carry a complete artefact and must not be merged with prose.
STRUCTURAL = frozenset({CODE, DIFF, LOG, STACKTRACE, CMD, CONFIG, TEST})

_FENCE_RE = re.compile(r"^\s*(```+|~~~+)\s*([A-Za-z0-9_+#.\-]*)\s*$")
_DIFF_HEADER_RE = re.compile(r"^(diff --git |--- a/|\+\+\+ b/|@@ )")
_DIFF_ANY_RE = re.compile(r"^(diff --git |@@ |index [0-9a-f]{7,})", re.MULTILINE)
_TRACEBACK_RE = re.compile(
    r"^(Traceback \(most recent call last\):|"
    r"\s+File \"[^\"]+\", line \d+|"
    r"\s+at [\w$.<>]+\s*\(|"
    r"\s+at .*:\d+:\d+\)?$|"
    r"\s*Caused by: )"
)
_EXC_TAIL_RE = re.compile(
    r"^[A-Za-z_][\w.]*(Error|Exception|Warning|Fault|Panic|Failure)\b.*:?.*$"
)
_LOG_LINE_RE = re.compile(
    r"^(\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?|"
    r"\d{2}:\d{2}:\d{2}[.,]\d{3}|"
    r"\[\s*(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|TRACE)\s*\]|"
    r"(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|TRACE)[:\s])"
)
_CMD_LINE_RE = re.compile(
    r"^\s*(\$|>|>>>|PS\s*[A-Za-z]:\\)\s*\S|"
    r"^\s*(npm|npx|yarn|pnpm|bun|pip3?|python3?|pytest|poetry|uv|conda|go|cargo|rustc|"
    r"make|cmake|gradle|mvn|bazel|dotnet|docker|kubectl|helm|terraform|git|gh|curl|wget|"
    r"node|deno|tsc|eslint|prettier|jest|vitest|mocha|ruff|mypy|black|flake8|tox|"
    r"apt|apt-get|yum|apk|brew|chmod|mkdir|cp|mv|rm|cat|grep|sed|awk|find|export|source|"
    r"pytest|unittest|nosetests|sbt|swift|php|composer|bundle|rake|gcc|g\+\+|clang)"
    r"(\s|$)"
)
_CONFIG_HINT_RE = re.compile(
    r"^\s*(\[[\w.\-]+\]\s*$|[\w.\-]+\s*[:=]\s*\S|\"[\w.\-]+\"\s*:|[\w.\-]+\s*\{)",
    re.MULTILINE,
)
_JSONISH_RE = re.compile(r"^\s*[\[{]")
_TEST_HINT_RE = re.compile(
    r"\b(test_|_test\b|\.test\.|\.spec\.|describe\(|it\(|assert|expect\(|"
    r"PASSED|FAILED|ERRORS?|passed|failed|xfail|pytest|jest|junit|"
    r"FAIL_TO_PASS|PASS_TO_PASS)\b"
)
_SHELL_PREFIX_RE = re.compile(r"^\s*(\$|>|>>>)\s+")

_LANG_BY_EXT = {
    "py": "python", "pyi": "python", "js": "javascript", "mjs": "javascript",
    "cjs": "javascript", "jsx": "javascript", "ts": "typescript", "tsx": "typescript",
    "java": "java", "kt": "kotlin", "go": "go", "rs": "rust", "rb": "ruby",
    "php": "php", "c": "c", "h": "c", "cc": "cpp", "cpp": "cpp", "hpp": "cpp",
    "cs": "csharp", "swift": "swift", "scala": "scala", "sh": "bash", "bash": "bash",
    "zsh": "bash", "ps1": "powershell", "sql": "sql", "yaml": "yaml", "yml": "yaml",
    "json": "json", "toml": "toml", "ini": "ini", "cfg": "ini", "xml": "xml",
    "html": "html", "css": "css", "scss": "scss", "md": "markdown", "dockerfile": "dockerfile",
}


@dataclass(slots=True)
class Segment:
    kind: str
    text: str
    lang: str | None = None
    line_start: int = 0
    line_end: int = 0


@dataclass(slots=True)
class Chunk:
    kind: str
    text: str
    lang: str | None
    line_start: int
    line_end: int
    part_index: int = 0
    part_count: int = 1
    meta: dict[str, str] = field(default_factory=dict)


# ------------------------------------------------------------------ scan ----


def _classify_unfenced(lines: list[str]) -> str:
    """Classify a run of non-fenced lines."""
    body = [ln for ln in lines if ln.strip()]
    if not body:
        return PROSE

    def ratio(pred) -> float:
        return sum(1 for ln in body if pred(ln)) / len(body)

    if ratio(lambda ln: bool(_DIFF_HEADER_RE.match(ln)) or ln.startswith(("+", "-"))) >= 0.5:
        return DIFF
    if ratio(lambda ln: bool(_TRACEBACK_RE.match(ln) or _EXC_TAIL_RE.match(ln))) >= 0.4:
        return STACKTRACE
    if ratio(lambda ln: bool(_LOG_LINE_RE.match(ln))) >= 0.5:
        return LOG
    if ratio(lambda ln: bool(_CMD_LINE_RE.match(ln))) >= 0.6:
        return CMD
    text = "\n".join(lines)
    if len(body) >= 3 and _CONFIG_HINT_RE.findall(text):
        if ratio(lambda ln: bool(_CONFIG_HINT_RE.match(ln))) >= 0.6 or _JSONISH_RE.match(text):
            if not _TEST_HINT_RE.search(text):
                return CONFIG
    # A single prose sentence that happens to mention "pytest" is not a test
    # artefact, so require at least two lines before promoting to TEST.
    if len(body) >= 2 and _TEST_HINT_RE.search(text) and ratio(
        lambda ln: bool(_CMD_LINE_RE.match(ln)) or bool(_TEST_HINT_RE.search(ln))
    ) >= 0.5:
        return TEST
    return PROSE


def _split_paragraphs(lines: list[str]) -> list[tuple[int, list[str]]]:
    """Split unfenced lines into blank-line separated paragraphs."""
    out: list[tuple[int, list[str]]] = []
    buf: list[str] = []
    start = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            if buf:
                out.append((start, buf))
                buf = []
        else:
            if not buf:
                start = i
            buf.append(line)
    if buf:
        out.append((start, buf))
    return out


def _split_on_diff_headers(lines: list[str]) -> list[tuple[int, list[str]]]:
    """Split a paragraph so a diff starts its own segment."""
    out: list[tuple[int, list[str]]] = []
    buf: list[str] = []
    start = 0
    for i, line in enumerate(lines):
        if _DIFF_HEADER_RE.match(line) and buf and not _DIFF_HEADER_RE.match(buf[0]):
            out.append((start, buf))
            buf = [line]
            start = i
        else:
            if not buf:
                start = i
            buf.append(line)
    if buf:
        out.append((start, buf))
    return out


def segment(content: str) -> list[Segment]:
    """Segment one message body into typed, non-overlapping segments."""
    lines = content.splitlines()
    segments: list[Segment] = []
    i = 0
    n = len(lines)

    while i < n:
        match = _FENCE_RE.match(lines[i])
        if match:
            fence = match.group(1)
            lang = (match.group(2) or "").strip().lower() or None
            close = re.compile(r"^\s*" + re.escape(fence[:3]) + r"+\s*$")
            j = i + 1
            body: list[str] = []
            while j < n and not close.match(lines[j]):
                body.append(lines[j])
                j += 1
            inner = "\n".join(body)
            kind = _classify_fenced(inner, lang)
            segments.append(
                Segment(kind=kind, text=inner, lang=lang, line_start=i, line_end=j)
            )
            i = j + 1 if j < n else n
            continue

        # Gather a run of unfenced lines up to the next fence.
        j = i
        run: list[str] = []
        while j < n and not _FENCE_RE.match(lines[j]):
            run.append(lines[j])
            j += 1
        if not run:
            run = [lines[i]]
            j = i + 1

        for p_start, para in _split_paragraphs(run):
            for d_start, piece in _split_on_diff_headers(para):
                kind = _classify_unfenced(piece)
                text = "\n".join(piece).strip("\n")
                if not text.strip():
                    continue
                segments.append(
                    Segment(
                        kind=kind,
                        text=text,
                        lang=_lang_for_unfenced(piece, kind),
                        line_start=i + d_start,
                        line_end=i + d_start + len(piece),
                    )
                )
        i = j

    return segments


def _classify_fenced(inner: str, lang: str | None) -> str:
    """A fenced block keeps its content type, not the fence language."""
    if _DIFF_ANY_RE.search(inner):
        return DIFF
    lines = inner.splitlines()
    body = [ln for ln in lines if ln.strip()]
    if body:
        trace = sum(1 for ln in body if _TRACEBACK_RE.match(ln) or _EXC_TAIL_RE.match(ln))
        if trace / len(body) >= 0.3:
            return STACKTRACE
        logs = sum(1 for ln in body if _LOG_LINE_RE.match(ln))
        if len(body) >= 3 and logs / len(body) >= 0.5:
            return LOG
        cmds = sum(1 for ln in body if _CMD_LINE_RE.match(ln) or _SHELL_PREFIX_RE.match(ln))
        if cmds / len(body) >= 0.6:
            return CMD
    if lang in {"json", "yaml", "yml", "toml", "ini", "cfg", "xml", "properties", "env"}:
        return CONFIG
    if lang in {"bash", "sh", "shell", "zsh", "console", "ps1", "powershell", "text"}:
        if body and sum(1 for ln in body if _CMD_LINE_RE.match(ln)) / len(body) >= 0.5:
            return CMD
    if lang in {"diff", "patch"}:
        return DIFF
    return CODE


def _lang_for_unfenced(lines: list[str], kind: str) -> str | None:
    if kind == DIFF:
        return "diff"
    if kind == LOG:
        return "log"
    if kind == CMD:
        return "bash"
    if kind == PROSE:
        # Prose that merely mentions "foo.py" is not Python; the file path is
        # captured as an entity instead.
        return None
    joined = "\n".join(lines[:40])
    exts = re.findall(r"[\w./\\-]+\.([A-Za-z0-9_+#]{1,12})\b", joined)
    if exts:
        from collections import Counter

        ext = Counter(e.lower() for e in exts).most_common(1)[0][0]
        if ext in _LANG_BY_EXT:
            return _LANG_BY_EXT[ext]
    if kind == CONFIG:
        return "config"
    return None


# ------------------------------------------------------------------ pack ----


def _split_oversize(text: str, max_tokens: int, hard_chars: int) -> list[str]:
    """Split text at line boundaries so no piece exceeds the limits.

    Whole lines are preserved; a single pathological line is the only case
    where a hard character cut happens.
    """
    if count_tokens(text) <= max_tokens and len(text) <= hard_chars:
        return [text]

    pieces: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    buf_chars = 0
    for line in text.splitlines():
        line_tokens = count_tokens(line)
        line_chars = len(line)
        if buf and (buf_tokens + line_tokens > max_tokens or buf_chars + line_chars > hard_chars):
            pieces.append("\n".join(buf))
            buf = []
            buf_tokens = 0
            buf_chars = 0
        buf.append(line)
        buf_tokens += line_tokens
        buf_chars += line_chars
    if buf:
        pieces.append("\n".join(buf))

    out: list[str] = []
    for piece in pieces:
        if len(piece) > hard_chars:
            for offset in range(0, len(piece), hard_chars):
                out.append(piece[offset : offset + hard_chars])
        else:
            out.append(piece)
    return [p for p in out if p.strip()]


def chunk_segments(
    segments: list[Segment],
    *,
    target_tokens: int = 320,
    max_tokens: int = 1400,
    hard_chars: int = 24_000,
) -> list[Chunk]:
    """Group segments into retrievable chunks."""
    chunks: list[Chunk] = []
    prose_buf: list[str] = []
    prose_start = 0
    prose_end = 0
    prose_lang: str | None = None

    def flush_prose() -> None:
        nonlocal prose_buf, prose_start, prose_end, prose_lang
        if not prose_buf:
            return
        text = "\n\n".join(prose_buf).strip()
        if text:
            for part_index, piece in enumerate(_split_oversize(text, max_tokens, hard_chars)):
                chunks.append(
                    Chunk(
                        kind=PROSE,
                        text=piece,
                        lang=prose_lang,
                        line_start=prose_start,
                        line_end=prose_end,
                    )
                )
        prose_buf = []
        prose_lang = None

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue

        if seg.kind == PROSE:
            cost = count_tokens(text)
            current = count_tokens("\n\n".join(prose_buf)) if prose_buf else 0
            if prose_buf and current + cost > target_tokens:
                flush_prose()
            if not prose_buf:
                prose_start = seg.line_start
                prose_lang = seg.lang
            prose_buf.append(text)
            prose_end = seg.line_end
            continue

        flush_prose()
        pieces = _split_oversize(text, max_tokens, hard_chars)
        for piece in pieces:
            chunks.append(
                Chunk(
                    kind=seg.kind,
                    text=piece,
                    lang=seg.lang,
                    line_start=seg.line_start,
                    line_end=seg.line_end,
                )
            )

    flush_prose()

    # Merge consecutive tiny structural chunks of the same kind so we do not
    # fragment one artefact into many near-duplicate memories.
    merged: list[Chunk] = []
    for chunk in chunks:
        if (
            merged
            and merged[-1].kind == chunk.kind
            and chunk.kind in STRUCTURAL
            and count_tokens(merged[-1].text) < target_tokens // 4
            and count_tokens(chunk.text) < target_tokens // 4
            and count_tokens(merged[-1].text + "\n" + chunk.text) <= target_tokens
        ):
            merged[-1].text = merged[-1].text + "\n" + chunk.text
            merged[-1].line_end = chunk.line_end
            continue
        merged.append(chunk)

    for chunk in merged:
        chunk.part_count = 1
    return merged


def chunk_content(content: str, **kwargs) -> list[Chunk]:
    return chunk_segments(segment(content), **kwargs)
