"""Listwise reranking via an LLM.

Why listwise, when a cross-encoder already reranks
--------------------------------------------------
The cross-encoder scores each candidate *independently*: it sees the question and
one memory and answers "how relevant". A listwise pass sees the whole candidate
set at once and can therefore reason comparatively — it knows that these forty
memories all mention this repository, so the one that actually explains the
failure is the informative one and the rest are incidental matches. That is
precisely the discrimination a same-repository corpus demands, and it is the one
thing a per-pair scorer cannot see.

Generation boundary
-------------------
This module **only produces numbers**. The model is asked for a relevance JSON
array, which is parsed into floats and blended into the existing score. No model
output ever reaches `data[].content`; that remains a verbatim span of stored
memory text. Rule 1 forbids generating an answer or presenting generated text as
a memory record, and scoring existing memories is not generation.

Failure is always non-fatal: an unavailable endpoint, a malformed response, or a
timeout leaves the fused ranking untouched.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Sequence

from ..core.config import Settings
from ..core.logging import get_logger

log = get_logger("codemem.listwise")

# Deliberately terse. The model is asked for a single array of integers, which
# keeps the response small and trivially parseable; anything richer invites the
# model to produce prose, and prose here would be generation.
_SYSTEM = (
    "You judge the relevance of candidate memory records to a software-engineering "
    "question. All candidates come from the same repository, so surface similarity "
    "(shared file paths, shared vocabulary) is NOT evidence of relevance. Judge "
    "whether each record would actually help answer the question. "
    "Reply with ONLY a JSON array of integers, one per candidate, in order, each "
    "0-10 where 0 means irrelevant and 10 means directly answers or resolves the "
    "question. Do not add commentary, keys, or any other text."
)


@dataclass
class ListwiseReranker:
    """An optional LLM that scores a candidate list comparatively."""

    model: str
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 60.0
    max_candidates: int = 40
    # Characters of each memory shown to the judge. Enough to tell a diagnosis
    # from an incidental mention, small enough to fit many candidates.
    excerpt_chars: int = 700
    _client: object | None = None
    _failed: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def available(self) -> bool:
        if self._failed:
            return False
        if self._client is not None:
            return True
        return self._load()

    def _load(self) -> bool:
        with self._lock:
            if self._client is not None:
                return True
            if self._failed:
                return False
            if not self.base_url or not self.api_key:
                log.warning(
                    "listwise rerank unavailable: base_url/api_key not configured"
                )
                self._failed = True
                return False
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    base_url=self.base_url, api_key=self.api_key, timeout=self.timeout
                )
                log.info(
                    "listwise reranker ready",
                    extra={"ctx": {"model": self.model, "base_url": self.base_url}},
                )
                return True
            except Exception as exc:
                log.warning(
                    "listwise reranker unavailable",
                    extra={"ctx": {"error": str(exc)[:200]}},
                )
                self._failed = True
                return False

    def score(self, question: str, documents: Sequence[str]) -> list[float] | None:
        """Return one relevance score per document, or None on any failure.

        The returned list always has the same length as ``documents`` so the
        caller can zip it safely; a short or malformed response is rejected
        outright rather than partially applied.
        """
        if not documents:
            return []
        if not self.available:
            return None

        shown = documents[: self.max_candidates]
        numbered = "\n\n".join(
            f"[{i}] {doc[: self.excerpt_chars]}" for i, doc in enumerate(shown)
        )
        user = (
            f"Question:\n{question}\n\n"
            f"Candidate memory records ({len(shown)}):\n{numbered}\n\n"
            f"Return a JSON array of exactly {len(shown)} integers."
        )

        try:
            response = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=64 + 8 * len(shown),
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            log.warning("listwise request failed", extra={"ctx": {"error": str(exc)[:200]}})
            return None

        parsed = _parse_scores(raw, len(shown))
        if parsed is None:
            log.warning(
                "listwise response unparseable; keeping fused order",
                extra={"ctx": {"raw": raw[:200]}},
            )
            return None

        # Candidates beyond max_candidates are not judged; they keep a neutral
        # score rather than being penalised for being unexamined.
        if len(documents) > len(shown):
            parsed = parsed + [None] * (len(documents) - len(shown))
        return parsed  # type: ignore[return-value]

    def close(self) -> None:
        self._client = None


_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*\]")


def _parse_scores(raw: str, expected: int) -> list[float] | None:
    """Extract exactly ``expected`` scores from a model response.

    Tolerant about wrapping (models sometimes fence or prefix output) but strict
    about the count and the value type: a partial or padded answer is discarded,
    because silently mixing judged and invented scores would corrupt the ranking
    in a way that is invisible downstream.
    """
    if not raw:
        return None
    candidates = _JSON_ARRAY_RE.findall(raw)
    for chunk in candidates:
        try:
            values = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if not isinstance(values, list) or len(values) != expected:
            continue
        scores: list[float] = []
        ok = True
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                ok = False
                break
            scores.append(float(max(0.0, min(10.0, float(value)))))
        if ok:
            return scores
    return None


def normalise(scores: Sequence[float | None]) -> list[float | None]:
    """Map 0-10 relevance onto 0-1, leaving unjudged entries as None."""
    return [None if s is None else max(0.0, min(1.0, s / 10.0)) for s in scores]
