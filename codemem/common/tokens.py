"""Token counting.

plan.md section 2.4 requires a counter that matches the target tokenizer and
forbids claiming exact control over platform truncation. We therefore measure
with tiktoken when available and otherwise fall back to a deliberately
*conservative* (over-counting) character heuristic, and we mark which mode was
used so the difference is auditable.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .config import Settings, get_settings

_MODE_TIKTOKEN = "tiktoken"
_MODE_ESTIMATE = "estimate"


@dataclass(frozen=True)
class TokenMeasurement:
    tokens: int
    mode: str


@functools.lru_cache(maxsize=8)
def _encoding(name: str):
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except Exception:
        return None


_TRUNCATION_MARKER = " …"


class TokenCounter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._encoder = _encoding(self.settings.tokenizer_encoding)
        self.mode = _MODE_TIKTOKEN if self._encoder is not None else _MODE_ESTIMATE

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text, disallowed_special=()))
        # Conservative: assume fewer characters per token than reality, so we
        # reserve more room than we need rather than less.
        return int(len(text) / self.settings.fallback_chars_per_token) + 1

    def _hard_cut(self, text: str, max_tokens: int) -> str:
        """Cut to at most ``max_tokens`` with no boundary adjustment."""
        if max_tokens <= 0:
            return ""
        if self._encoder is not None:
            ids = self._encoder.encode(text, disallowed_special=())
            return self._encoder.decode(ids[:max_tokens])
        keep = int(max_tokens * self.settings.fallback_chars_per_token)
        return text[:keep]

    def truncate(self, text: str, max_tokens: int) -> str:
        """Trim to a token budget without splitting mid-line when avoidable.

        The ellipsis marker is charged against the budget, so the result is
        genuinely bounded rather than bounded-plus-a-suffix.
        """
        if max_tokens <= 0:
            return ""
        if self.count(text) <= max_tokens:
            return text

        marker_cost = self.count(_TRUNCATION_MARKER)
        body_budget = max_tokens - marker_cost
        if body_budget <= 0:
            # No room for a marker: cut hard rather than silently overshoot.
            return self._hard_cut(text, max_tokens)

        cut = self._hard_cut(text, body_budget)
        # Prefer a clean boundary so a truncated excerpt does not end mid-token.
        for separator in ("\n", " "):
            idx = cut.rfind(separator)
            if idx > len(cut) * 0.6:
                return cut[:idx].rstrip() + _TRUNCATION_MARKER
        return cut.rstrip() + _TRUNCATION_MARKER


_counter: TokenCounter | None = None


def get_counter(settings: Settings | None = None) -> TokenCounter:
    global _counter
    if _counter is None or settings is not None:
        _counter = TokenCounter(settings)
    return _counter


def reset_counter_for_tests() -> None:
    global _counter
    _counter = None