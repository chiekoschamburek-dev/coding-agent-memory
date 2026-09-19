"""Sparse-text preparation for FTS5.

FTS5's ``unicode61`` tokenizer splits on punctuation but keeps ``camelCase`` and
keeps ``snake_case`` intact (as one token with the underscore). That is wrong
for code retrieval in both directions:

* a query for ``read_token`` should also reach text that wrote ``readToken``;
* a query for ``read`` should reach ``read_token``.

So alongside the verbatim text we build a ``sparse`` field that appends the
sub-tokens of every code-like identifier found. Everything is literal and
deterministic; no stemming, no synonyms, so nothing here can be mistaken for
generated content.
"""

from __future__ import annotations

import re

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_WORDLIKE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Sub-tokens this short are noise ("id", "of", "a").
_MIN_SUBTOKEN = 3
_MAX_TOKENS = 4000


def identifier_subtokens(token: str) -> list[str]:
    """Split an identifier into its searchable sub-tokens."""
    parts: list[str] = []
    for dot_piece in token.split("."):
        if not dot_piece:
            continue
        parts.append(dot_piece.lower())
        for camel in _CAMEL_BOUNDARY.split(dot_piece):
            if camel:
                parts.append(camel.lower())
        if "_" in dot_piece or "-" in dot_piece:
            parts.extend(p.lower() for p in re.split(r"[_\-]+", dot_piece) if p)
    seen: dict[str, None] = {}
    for part in parts:
        if len(part) >= _MIN_SUBTOKEN and part.isalnum() and part not in seen:
            seen[part] = None
    return list(seen)


def build_sparse(*texts: str | None) -> str:
    """Build the auxiliary FTS column: identifier sub-tokens, de-duplicated."""
    out: dict[str, None] = {}
    budget = _MAX_TOKENS
    for text in texts:
        if not text or budget <= 0:
            continue
        for match in _IDENT_RE.finditer(text):
            token = match.group(0)
            if not _WORDLIKE.match(part := token.split(".")[0]) and len(token.split(".")) == 1:
                continue
            for sub in identifier_subtokens(token):
                if sub not in out:
                    out[sub] = None
                    budget -= 1
                    if budget <= 0:
                        break
            if budget <= 0:
                break
    return " ".join(out)


def fts_query_terms(text: str, *, max_terms: int = 24, min_len: int = 2) -> list[str]:
    """Extract literal query terms (plus identifier sub-tokens) for FTS/BM25.

    Terms are OR-combined by the caller; this only decides the vocabulary.
    """
    terms: dict[str, None] = {}
    for match in _IDENT_RE.finditer(text or ""):
        token = match.group(0)
        if len(token) >= min_len:
            terms.setdefault(token.lower(), None)
        for sub in identifier_subtokens(token):
            terms.setdefault(sub, None)
        if len(terms) >= max_terms:
            return list(terms)
    # Also keep standalone CJK runs and long numbers, which carry real signal
    # in stack traces and issue references.
    for match in re.finditer(r"[\u4e00-\u9fff]{2,}|\d{3,}", text or ""):
        terms.setdefault(match.group(0), None)
        if len(terms) >= max_terms:
            break
    return list(terms)


def escape_fts_token(token: str) -> str:
    """Quote a token so FTS5 treats it as a literal, never as syntax."""
    return '"' + token.replace('"', '""') + '"'
