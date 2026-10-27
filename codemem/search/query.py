"""Query feature extraction (plan.md section 6.1).

Two separate channels are produced:

* the **query channel** -- the task statement itself, full weight;
* the **options channel** -- balanced expansions distilled from the option
  list, de-weighted at fusion time.

``api_guide.md`` guarantees options carry no gold answer, and plan.md section
6.1/6.2 forbids treating them as facts, gold labels, or the answer target. They
are used strictly as an additional lexical/entity probe: a path or symbol that
appears in an option is a legitimate hint about which part of the codebase the
question concerns. Nothing from options is ever stored back into memory.

The query itself is **denoised before feature extraction**.  Task prompts in
the code track are PR-issue text and routinely carry an HTML comment block, a
markdown checklist, and documentation URLs.  Those tokens are not evidence
about the bug and they used to dominate both BM25 and the exact-entity
channel: a real prompt yielded ``path=n/a``, ``config_key=docs.github.com``
and ``symbol=documenting_mpl``, none of which exist in any repository.  The
raw text is kept for the vector channel (semantics), the clean text drives
lexical/entity/anchors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..common.chunking import extract_entities
from ..common.store import ChunkRecord, tokenize

_OPTION_PREFIX_RE = re.compile(r"^\s*(?:[A-Za-z]|[0-9]{1,2})\s*[.):、]\s*")

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "to",
        "in", "on", "at", "for", "with", "by", "from", "as", "that", "this",
        "it", "its", "which", "what", "when", "where", "how", "why", "and",
        "or", "not", "no", "do", "does", "did", "should", "would", "could",
        "can", "will", "code", "following", "best", "most", "likely", "task",
        "answer", "option", "options", "question", "memory", "historical",
    }
)

# -- denoising --------------------------------------------------------------

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_URL_RE = re.compile(r"\b(?:https?|ftp)://\S+", re.I)
_MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_CHECKBOX_RE = re.compile(r"^\s*[-*]\s*\[[ xX]\]\s*.*$", re.M)
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>\n]{0,80}>")
_HEADING_MARK_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
_PR_BOILERPLATE_RE = re.compile(
    r"^\s*(?:##\s*(?:PR summary|PR checklist|Checklist|Description)\s*)$",
    re.M | re.I,
)

# -- anchors ----------------------------------------------------------------

_ISSUE_REF_RE = re.compile(r"(?<![\w#])#(\d{2,6})\b")
_CODE_PATH_RE = re.compile(
    r"(?<![\w./\\-])"
    r"((?:[A-Za-z0-9_.+-]+[/\\])*[A-Za-z0-9_.+-]+"
    r"\.(?:py|pyi|js|jsx|mjs|cjs|ts|tsx|go|rs|java|kt|rb|php|c|h|cc|cpp|hpp|cs|"
    r"swift|scala|sh|bash|zsh|ps1|sql|json|ya?ml|toml|ini|cfg|conf|gradle|mod|lock))"
    r"(?![A-Za-z0-9_])(?::(\d+))?"
)
_CODE_SYMBOL_RE = re.compile(r"\b(?:def|class|function|func|struct|interface)\s+([A-Za-z_][A-Za-z0-9_]*)")
_BACKTICK_RE = re.compile(r"`([^`\n]{2,80})`")

_JUNK_ENTITY_VALUES = frozenset({"n/a", "na", "none", "null", "http", "https", "www"})
_DOMAIN_RE = re.compile(r"^(?:[\w-]+\.)+(?:com|org|net|io|ai|dev|cn|edu|gov)$", re.I)
_DOC_EXTS = frozenset({".html", ".md", ".rst", ".txt", ".css", ".scss"})

_MAX_ANCHORS_PER_KIND = 12


def denoise_query(text: str) -> str:
    """Strip PR-template noise before tokenising/entity extraction.

    The original query is still sent to the embedding channel, so semantic
    meaning is preserved; this only removes characters that cannot be repo
    evidence.
    """
    if not text:
        return ""
    cleaned = _HTML_COMMENT_RE.sub(" ", text)
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _MD_LINK_RE.sub(r"\1", cleaned)
    cleaned = _CHECKBOX_RE.sub("", cleaned)
    cleaned = _PR_BOILERPLATE_RE.sub("", cleaned)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = _HEADING_MARK_RE.sub("", cleaned)
    # Collapse the blank runs left behind, but keep paragraph structure.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _is_junk_entity(kind: str, value: str) -> bool:
    low = value.strip().lower()
    if not low or low in _JUNK_ENTITY_VALUES or len(low) < 3:
        return True
    if "://" in low or low.startswith("www."):
        return True
    if _DOMAIN_RE.match(low):
        return True
    if kind == "path":
        ext = low.rsplit(".", 1)[-1]
        if f".{ext}" in _DOC_EXTS:
            return True
        if "/" not in low and ext in {"html", "md", "rst", "css", "txt"}:
            return True
    if kind == "config_key" and low.startswith(("www.", "docs.")):
        return True
    return False


def extract_anchors(text: str) -> dict[str, list[str]]:
    """Structured, repository-relevant anchors in a query.

    These are the bridges between a current issue and a historical case:
    file paths, symbols, error literals, test names and issue numbers.  The
    generic rule-based extractor is reused to keep index and query vocabularies
    identical, then filtered for URL/documentation noise.
    """
    grouped: dict[str, list[str]] = {}

    def add(kind: str, value: str) -> None:
        value = value.strip().strip(".,:;()[]{}'\"")
        if not value or _is_junk_entity(kind, value):
            return
        bucket = grouped.setdefault(kind, [])
        if value.lower() not in {v.lower() for v in bucket} and len(bucket) < _MAX_ANCHORS_PER_KIND:
            bucket.append(value)

    probe = ChunkRecord(
        chunk_id="query",
        message_id="query",
        message_index=0,
        ord=0,
        char_start=0,
        char_end=len(text),
        kind="code",
        text=text,
        parent_chunk_id=None,
        tokens=(),
    )
    for entity in extract_entities([probe]):
        add(entity.kind, entity.value)

    for match in _ISSUE_REF_RE.finditer(text):
        add("issue", f"#{match.group(1)}")
    for match in _CODE_PATH_RE.finditer(text):
        add("path", match.group(1))
    for match in _CODE_SYMBOL_RE.finditer(text):
        add("symbol", match.group(1))
    for match in _BACKTICK_RE.finditer(text):
        token = match.group(1).strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{2,60}", token):
            add("symbol", token)
    return grouped


def _content_tokens(tokens: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for token in tokens:
        if token in _STOPWORDS or len(token) < 2:
            continue
        seen.setdefault(token, None)
    return list(seen)


@dataclass
class QueryFeatures:
    raw_query: str
    options: list[str] = field(default_factory=list)
    # Query-channel artefacts
    query_text: str = ""
    denoised_text: str = ""
    tokens: list[str] = field(default_factory=list)
    content_tokens: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    anchors: dict[str, list[str]] = field(default_factory=dict)
    # Options-channel artefacts (never facts, only retrieval probes)
    options_text: str = ""
    option_tokens: list[str] = field(default_factory=list)
    option_entities: dict[str, list[str]] = field(default_factory=dict)

    def entity_values(self, kinds: tuple[str, ...] = ("path", "symbol", "error", "test_name", "config_key")) -> list[str]:
        out: list[str] = []
        for kind in kinds:
            out.extend(self.entities.get(kind, ()))
        return out

    def option_entity_values(self, kinds: tuple[str, ...] = ("path", "symbol", "error", "test_name", "config_key")) -> list[str]:
        out: list[str] = []
        for kind in kinds:
            out.extend(self.option_entities.get(kind, ()))
        return out

    def anchor_values(self, kinds: tuple[str, ...] | None = None) -> list[str]:
        kinds = kinds or tuple(self.anchors)
        out: list[str] = []
        for kind in kinds:
            out.extend(self.anchors.get(kind, ()))
        return out


def _strip_option_prefix(option: str) -> str:
    return _OPTION_PREFIX_RE.sub("", option).strip()


def _extract_from_text(text: str) -> tuple[dict[str, list[str]], list[str]]:
    """Run the shared rule-based extractor over a query string.

    Reusing ``chunking.extract_entities`` keeps query-side and index-side
    entity vocabularies identical -- a mismatch there would silently kill the
    entity channel.
    """
    probe = ChunkRecord(
        chunk_id="query",
        message_id="query",
        message_index=0,
        ord=0,
        char_start=0,
        char_end=len(text),
        kind="code",
        text=text,
        parent_chunk_id=None,
        tokens=(),
    )
    grouped: dict[str, list[str]] = {}
    for entity in extract_entities([probe]):
        if _is_junk_entity(entity.kind, entity.value):
            continue
        bucket = grouped.setdefault(entity.kind, [])
        if entity.value.lower() not in {value.lower() for value in bucket}:
            bucket.append(entity.value)
    return grouped, tokenize(text)


def build_query_features(
    query: str,
    options: list[str] | None = None,
    *,
    denoise: bool = True,
) -> QueryFeatures:
    options = list(options or [])
    features = QueryFeatures(raw_query=query, options=options)

    features.query_text = query
    denoised = denoise_query(query) if denoise else query
    # A prompt can be almost entirely template boilerplate (a PR checklist with
    # no description).  Falling back keeps such a query searchable instead of
    # reducing it to one line.
    if len(denoised) < 20:
        denoised = query
    features.denoised_text = denoised
    probe_text = features.denoised_text or query
    features.tokens = tokenize(probe_text)
    features.content_tokens = _content_tokens(features.tokens)
    features.entities, _ = _extract_from_text(probe_text)
    features.anchors = extract_anchors(probe_text)

    if options:
        # Balanced: every option contributes, prefixes stripped, order kept.
        # Joined by newline so entity extraction cannot pair a trailing token of
        # one option with the leading token of the next.
        normalised = [_strip_option_prefix(option) for option in options]
        features.options_text = "\n".join(normalised)
        features.option_tokens = _content_tokens(tokenize(features.options_text))
        features.option_entities, _ = _extract_from_text(features.options_text)

    return features
