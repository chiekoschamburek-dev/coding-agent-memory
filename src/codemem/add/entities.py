"""Deterministic metadata extraction (L2, no LLM).

Code retrieval is dominated by literal identifiers: a file path, an exception
class or a symbol name is far more discriminative than sentence similarity,
and — critically — it does not drift when the memory pool is full of
same-repository distractors that share vocabulary and style. This module
extracts those identifiers with regexes only, so results are reproducible and
cost nothing.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

# ------------------------------------------------------------- vocab -------

CODE_EXTS = {
    "py", "pyi", "pyx", "js", "mjs", "cjs", "jsx", "ts", "tsx", "java", "kt",
    "kts", "go", "rs", "rb", "php", "c", "h", "cc", "cpp", "cxx", "hpp", "hh",
    "cs", "swift", "scala", "m", "mm", "sh", "bash", "zsh", "fish", "ps1",
    "bat", "sql", "yaml", "yml", "json", "json5", "toml", "ini", "cfg", "conf",
    "xml", "html", "htm", "css", "scss", "sass", "less", "vue", "svelte", "md",
    "rst", "txt", "lock", "gradle", "cmake", "mk", "dockerfile", "env", "tf",
    "proto", "graphql", "gql", "ipynb", "patch", "diff", "csv", "tsv", "log",
}

# Extensions that look like versions/numbers rather than source files.
_VERSIONISH_RE = re.compile(r"^\d+(\.\d+)+$")

# Trackers we accept for issue ids, plus a guard against "UTF-8"-style tokens.
_ISSUE_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9]{1,9})-(\d{1,6})\b")
_ISSUE_STOP_PREFIX = {
    "UTF", "ISO", "SHA", "MD", "AES", "TLS", "ASCII", "UTC", "RFC", "IPV",
    "SHA1", "SHA2", "SHA256", "X86", "ARM", "UTC", "BOM", "HTTP", "HTTPS",
}
_HASH_ISSUE_RE = re.compile(r"(?<![\w#])#(\d{2,6})\b")

_EXCEPTION_RE = re.compile(
    r"\b((?:[a-z_][\w]*\.)*[A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning|Fault|Panic|Failure))\b"
)
_DEF_RE = re.compile(
    r"^\s*(?:async\s+)?(?:def|class|function|func|fn|struct|interface|enum|trait|impl|type)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\(")
_BACKTICK_RE = re.compile(r"`([^`\n]{1,120})`")
_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import|import\s+([A-Za-z_][\w.]*)|"
    r"#include\s*[<\"]([\w./\-]+)[>\"]|"
    r"require\(\s*['\"]([^'\"]+)['\"]|"
    r"use\s+([A-Za-z_][\w\\]*)|"
    r"using\s+([A-Za-z_][\w.]*))",
    re.MULTILINE,
)
_TESTNAME_RE = re.compile(r"\b(test_[A-Za-z0-9_]{2,}|[A-Za-z0-9_]{3,}_test)\b")
_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/])?(?:[\w.@\-]+[\\/]){1,12}[\w.@\-]+\.[A-Za-z0-9_+#]{1,12}"
    r"|[\w.@\-]+\.[A-Za-z0-9_+#]{1,12}"
)
_CMD_FIRST_RE = re.compile(r"^\s*(?:\$|>|>>>)\s+([A-Za-z][\w.\-]*)")
_CMD_BARE_RE = re.compile(
    r"^\s*((?:npm|npx|yarn|pnpm|bun|pip3?|python3?|pytest|poetry|uv|conda|go|cargo|"
    r"make|cmake|gradle|mvn|bazel|dotnet|docker|kubectl|helm|terraform|git|gh|curl|"
    r"wget|node|deno|tsc|eslint|prettier|jest|vitest|mocha|ruff|mypy|black|flake8|"
    r"tox|sbt|swift|php|composer|bundle|rake|gcc|g\+\+|clang)(?=\s|$))"
)

# Identifiers too generic to be useful as a retrieval key.
_SYMBOL_STOP = {
    "print", "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "range", "open", "read", "write", "main", "self", "cls", "init", "test",
    "get", "set", "add", "run", "new", "type", "value", "name", "data", "item",
    "items", "keys", "values", "append", "format", "join", "split", "map",
    "filter", "sum", "min", "max", "abs", "all", "any", "next", "iter", "input",
    "exception", "error", "none", "true", "false", "null", "string", "object",
    "array", "return", "raise", "assert", "if", "else", "for", "while", "with",
    "console", "log", "require", "import", "from", "def", "class", "function",
    "fn", "func", "struct", "impl", "trait", "interface", "enum", "public",
    "private", "static", "void", "final", "var", "let", "const", "async", "await",
    "yield", "lambda", "echo", "printf", "toString", "valueOf", "equals", "hash",
    "main", "args", "kwargs", "here", "there", "case", "when", "then", "that",
    "with", "this", "into", "your", "have", "been", "will", "should", "would",
}

_SNAKE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_CAMEL_RE = re.compile(r"^[a-z]+[A-Z][A-Za-z0-9]*$")
_PASCAL_RE = re.compile(r"^[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+$")

# Per-chunk caps keep the inverted index from exploding on large blobs.
MAX_PER_TYPE = {
    "file_path": 40,
    "file_name": 40,
    "dir": 20,
    "symbol": 60,
    "exception": 20,
    "test": 20,
    "cmd": 15,
    "pkg": 25,
    "issue_id": 15,
    "lang": 5,
}

# Relative importance when matching a query entity. Exactness decays down the
# list: a full path match is stronger evidence than a shared language tag.
ENTITY_WEIGHT = {
    "file_path": 1.00,
    "file_name": 0.85,
    "symbol": 0.80,
    "exception": 0.75,
    "test": 0.70,
    "dir": 0.55,
    "pkg": 0.50,
    "cmd": 0.35,
    "issue_id": 0.65,
    "lang": 0.15,
}


@dataclass(frozen=True, slots=True)
class Entity:
    etype: str
    value_norm: str
    value_raw: str


def _norm_path(raw: str) -> str:
    value = raw.replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value.strip("/") if value.startswith("/") and not re.match(r"^[A-Za-z]:", value) else value


def _looks_like_identifier(token: str) -> bool:
    low = token.lower()
    if low in _SYMBOL_STOP:
        return False
    if len(token) < 3:
        return False
    if _VERSIONISH_RE.match(token):
        return False
    parts = token.split(".")
    tail = parts[-1]
    if tail.lower() in _SYMBOL_STOP:
        return False
    return bool(_SNAKE_RE.match(tail) or _CAMEL_RE.match(tail) or _PASCAL_RE.match(tail))


def _extract_paths(text: str) -> tuple[list[str], list[str]]:
    full: list[str] = []
    names: list[str] = []
    for match in _PATH_RE.finditer(text):
        raw = match.group(0)
        if raw.startswith(("http://", "https://")):
            continue
        ext = raw.rsplit(".", 1)[-1].lower()
        if ext not in CODE_EXTS:
            continue
        if _VERSIONISH_RE.match(raw):
            continue
        norm = _norm_path(raw)
        if not norm:
            continue
        full.append(norm)
        base = norm.rsplit("/", 1)[-1]
        if base != norm:
            names.append(base)
    return full, names


def _extract_dirs(text: str) -> list[str]:
    dirs: list[str] = []
    for match in re.finditer(r"((?:[\w.@\-]+/){1,12})", text):
        raw = match.group(1)
        if len(raw) < 3 or raw.startswith(("http", "//")):
            continue
        if any(ch in raw for ch in (" ", "\t")):
            continue
        dirs.append(raw.rstrip("/"))
    return dirs


def _extract_symbols(text: str) -> list[str]:
    out: list[str] = []
    for match in _DEF_RE.finditer(text):
        out.append(match.group(1))
    for match in _BACKTICK_RE.finditer(text):
        body = match.group(1).strip()
        if not body:
            continue
        candidate = body.split("(")[0].strip()
        if " " in candidate or candidate.endswith((".py", ".js", ".ts", ".go", ".java")):
            continue
        for piece in re.split(r"[,\s]+", candidate):
            piece = piece.strip(".'\"():;")
            if _looks_like_identifier(piece):
                out.append(piece)
    for match in _CALL_RE.finditer(text):
        token = match.group(1)
        if _looks_like_identifier(token):
            out.append(token)
    return out


def _extract_pkgs(text: str) -> list[str]:
    out: list[str] = []
    for match in _IMPORT_RE.finditer(text):
        raw = next((g for g in match.groups() if g), None)
        if not raw:
            continue
        raw = raw.strip().strip("<>\"'")
        if raw.startswith((".", "/")):
            continue
        # Keep the import root: `foo.bar.baz` -> `foo`.
        root = re.split(r"[./\\]", raw)[0]
        if root:
            out.append(root)
    return out


def _extract_cmds(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        for regex in (_CMD_FIRST_RE, _CMD_BARE_RE):
            match = regex.match(line)
            if match:
                token = match.group(1)
                if token:
                    out.append(token.lower())
                break
    return out


def _extract_issues(text: str) -> list[str]:
    out: list[str] = []
    for match in _ISSUE_RE.finditer(text):
        prefix = match.group(1).upper()
        if prefix in _ISSUE_STOP_PREFIX:
            continue
        out.append(f"{prefix}-{match.group(2)}")
    for match in _HASH_ISSUE_RE.finditer(text):
        out.append(f"#{match.group(1)}")
    return out


def _dedupe(values: Iterable[str], cap: int) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        if value and value not in seen:
            seen[value] = None
            if len(seen) >= cap:
                break
    return list(seen)


def extract_entities(text: str, *, kind: str | None = None, lang: str | None = None) -> list[Entity]:
    """Extract deterministic entities from one chunk.

    ``kind``/``lang`` come from the chunker and only narrow which extractors
    run; every extractor is literal and reproducible.
    """
    if not text:
        return []

    entities: list[Entity] = []

    paths, names = _extract_paths(text)
    entities += [Entity("file_path", p, p) for p in _dedupe(paths, MAX_PER_TYPE["file_path"])]
    entities += [Entity("file_name", n, n) for n in _dedupe(names, MAX_PER_TYPE["file_name"])]

    entities += [
        Entity("dir", d, d) for d in _dedupe(_extract_dirs(text), MAX_PER_TYPE["dir"])
    ]

    is_codeish = kind in {"code", "diff", "stacktrace", "test", "config"}
    if kind != "prose" or "`" in text:
        symbols = _extract_symbols(text) if is_codeish else _extract_symbols_backticks_only(text)
        entities += [
            Entity("symbol", s, s) for s in _dedupe(symbols, MAX_PER_TYPE["symbol"])
        ]

    entities += [
        Entity("exception", e, e)
        for e in _dedupe(_EXCEPTION_RE.findall(text), MAX_PER_TYPE["exception"])
    ]
    entities += [
        Entity("test", t, t) for t in _dedupe(_TESTNAME_RE.findall(text), MAX_PER_TYPE["test"])
    ]
    if is_codeish:
        entities += [
            Entity("cmd", c, c) for c in _dedupe(_extract_cmds(text), MAX_PER_TYPE["cmd"])
        ]
        entities += [
            Entity("pkg", p, p) for p in _dedupe(_extract_pkgs(text), MAX_PER_TYPE["pkg"])
        ]
    entities += [
        Entity("issue_id", i, i) for i in _dedupe(_extract_issues(text), MAX_PER_TYPE["issue_id"])
    ]

    if lang:
        entities.append(Entity("lang", lang.lower(), lang.lower()))

    # De-duplicate on (etype, value_norm) while preserving first-seen order.
    seen: set[tuple[str, str]] = set()
    unique: list[Entity] = []
    for entity in entities:
        key = (entity.etype, entity.value_norm)
        if key in seen:
            continue
        seen.add(key)
        unique.append(entity)
    return unique


def _extract_symbols_backticks_only(text: str) -> list[str]:
    out: list[str] = []
    for match in _BACKTICK_RE.finditer(text):
        candidate = match.group(1).split("(")[0].strip()
        if " " in candidate:
            continue
        if _looks_like_identifier(candidate):
            out.append(candidate)
    return out


def entity_counts(entities: Iterable[Entity]) -> Counter[tuple[str, str]]:
    return Counter((e.etype, e.value_norm) for e in entities)
