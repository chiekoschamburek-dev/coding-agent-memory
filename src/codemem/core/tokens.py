"""Token accounting.

The platform truncates our returned evidence at a token-counted prefix, so we
must budget in tokens rather than characters. We prefer ``tiktoken`` when it is
installed and fall back to a deterministic heuristic otherwise; the heuristic
is deliberately conservative (over-estimates) so we never overshoot a budget.
"""

from __future__ import annotations

import functools
import re

_WORD_RE = re.compile(r"\w+|[^\w\s]")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


@functools.lru_cache(maxsize=1)
def _encoder():  # pragma: no cover - depends on optional dependency
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Approximate the token count of ``text``."""
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:  # pragma: no cover - optional path
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    return _heuristic(text)


def _heuristic(text: str) -> int:
    cjk = len(_CJK_RE.findall(text))
    words = len(_WORD_RE.findall(text))
    # Code and prose both average roughly 1.3 tokens per lexical unit, and CJK
    # runs near 1 token per character. Take the larger estimate.
    return max(words, cjk) + (words // 4)


def truncate_to_tokens(text: str, budget: int) -> str:
    """Truncate ``text`` so that ``count_tokens`` stays within ``budget``.

    Splits on line boundaries when possible so that code stays readable, and
    hard-truncates as a last resort. Never raises.
    """
    if budget <= 0:
        return ""
    if count_tokens(text) <= budget:
        return text

    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    used = 0
    for line in lines:
        cost = count_tokens(line)
        if used + cost > budget:
            break
        kept.append(line)
        used += cost

    if kept:
        result = "".join(kept).rstrip()
        if result and result != text.strip():
            result += "\n…"
        return result

    # A single line exceeds the budget: hard-cut by characters.
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + "…" if lo else ""
