"""Query understanding.

Extracts the retrieval probes for Search. Two rules constrain this module:

* it never produces answer text — only search keys;
* if ``options`` are present they are used solely to widen the set of things
  the *question asks about* (e.g. option wording mentioning a subsystem), never
  to decide which option a memory supports. Option-level filtering would shade
  into "disguising an answer as memory" and is explicitly out of scope.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..add.entities import extract_entities
from ..core.text import STOPWORDS as _STOPWORDS

_OPTION_PREFIX_RE = re.compile(r"^\s*(?:\(?[A-Za-z]\)|[A-Za-z][.):]|\d+[.):])\s*")
_CODEISH_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


@dataclass(slots=True)
class QueryPlan:
    """The retrieval probes derived from one Search request."""

    query: str
    options: list[str]
    intent: str
    probes: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)

    @property
    def has_entities(self) -> bool:
        return any(self.entities.values())


def _strip_option(label: str) -> str:
    return _OPTION_PREFIX_RE.sub("", label or "").strip()


def _keywords(text: str, limit: int = 40) -> list[str]:
    seen: dict[str, None] = {}
    for match in _CODEISH_RE.finditer(text or ""):
        token = match.group(0)
        low = token.lower()
        if low in _STOPWORDS or (len(token) < 3 and not token.isdigit()):
            continue
        if low not in seen:
            seen[low] = None
            if len(seen) >= limit:
                break
    return list(seen)


def _detect_intent(text: str, entities: dict[str, list[str]]) -> str:
    """Classify the question as a debugging or development task.

    Exception classes are checked separately because a word-boundary search for
    "error" does not match camel-case ``IndexError`` (the character before
    ``Error`` is a word character, so there is no boundary).
    """
    low = (text or "").lower()
    if entities.get("exception"):
        return "debug"
    if re.search(r"\b(fix|fixed|bug|crash|fails?|failing|failed|failure|broken|"
                 r"regression|traceback|exception|error|debug|root cause|patch|"
                 r"throws?|raising|raises)\b", low):
        return "debug"
    if re.search(r"\b(add|implement|feature|support|introduce|enhance|extend|"
                 r"refactor|migrate|upgrade|design|architecture|should we)\b", low):
        return "develop"
    return "general"


def plan_query(query: str, options: list[str] | None = None) -> QueryPlan:
    """Build retrieval probes for a Search request."""
    opts = [o for o in (options or []) if o]
    combined = query + "\n" + "\n".join(opts) if opts else query

    # Identifiers from the question are the highest-value probes; when the
    # question is vague, probe the option wording instead of the raw query so
    # the retriever sees the candidate topic space.
    probe_sources: list[str] = [query]
    if opts:
        probe_sources.extend(_strip_option(o) for o in opts[:6])
    probes: list[str] = []
    seen_probes: set[str] = set()
    for source in probe_sources:
        source = source.strip()
        if not source or source in seen_probes:
            continue
        seen_probes.add(source)
        probes.append(source)

    entities: dict[str, list[str]] = {}
    # Entities from the question are exact; option-derived entities are weaker
    # context, so they are added only to widen recall, never to filter.
    for source in probes:
        for entity in extract_entities(source, kind="prose", lang=None):
            bucket = entities.setdefault(entity.etype, [])
            if entity.value_norm not in bucket:
                bucket.append(entity.value_norm)

    intent = _detect_intent(combined, entities)
    keywords = _keywords(combined)

    return QueryPlan(
        query=query,
        options=opts,
        intent=intent,
        probes=probes,
        entities=entities,
        keywords=keywords,
    )
