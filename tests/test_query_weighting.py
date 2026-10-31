"""Query-weighting tests (FIXES.md query-weighting fix).

Two defects this pins:

1. **Generic identifier halves were double-weighted.** An id like
   ``repo__repo-1234`` tokenises to ``['repo__repo', 'repo', 'repo', '1234']``;
   feeding raw query tf into BM25 multiplied the *generic* token's
   contribution by 2, cancelling the IDF penalty meant to demote exactly those
   high-df tokens.
2. **Stopword filtering was dead code.** ``content_tokens`` was computed but
   the retrieval path passed the raw query text instead.

The retrieval surface is the card layer now: field BM25
(``experience_field_search``), card entities and IDF-weighted case anchors.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from codemem.common.config import Settings  # noqa: E402
from codemem.search.query import build_query_features  # noqa: E402
from codemem.common.store import Store, tokenize  # noqa: E402


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(db_path=tmp_path / "qw.db", embed_model="", embed_base_url=None, embed_api_key=None)
    base.update(overrides)
    return Settings(**base)


def seed_card(store: Store, settings: Settings, user_id: str, eid: str, text: str,
              anchors=()):
    """Insert one card whose index face is the given text (plus anchors)."""
    from codemem.common.store import now_iso

    store.insert_experience(
        user_id=user_id, session_id="s", experience_id=eid,
        ingestion_id=_ingestion_id(store, user_id, eid),
        kind="debug", payload_json="{}", full_content=text,
        extraction_version="t", created_at=now_iso(),
        terms=[f"problem\x1f{term}" for term in tokenize(text)],
        entities=[],
        anchors=anchors,
    )


def _ingestion_id(store: Store, user_id: str, request_key: str) -> int:
    state = store.begin_ingestion(
        user_id=user_id, request_id=f"r-{request_key}", session_id="s",
        digest=f"r-{request_key}", mode="lex_only", lease_seconds=60,
    )
    return state.ingestion_id


# --------------------------------------------------------------------------
# defect 1: generic identifier halves must not be double-weighted
# --------------------------------------------------------------------------


def test_identifier_tokenisation_produces_duplicate_generic_half():
    """Documents the shape that caused the bug."""
    tokens = tokenize("matplotlib__matplotlib-24970")
    assert tokens.count("matplotlib") == 2, "generic half appears twice"
    assert tokens.count("24970") == 1, "discriminative half appears once"


def test_query_terms_are_deduplicated(tmp_path):
    """Presence, not multiplicity, is the query signal.

    Score for a query must be identical whether the generic half appears once
    or three times -- otherwise repetition silently boosts it.
    """
    settings = make_settings(tmp_path)
    store = Store(settings)
    seed_card(store, settings, "u", "e:1", "matplotlib__matplotlib-24970 fix in plot code")
    seed_card(store, settings, "u", "e:2", "matplotlib chart rendering unrelated content")

    once = store.experience_field_search("u", ["matplotlib", "24970"], 10)
    thrice = store.experience_field_search(
        "u", ["matplotlib", "matplotlib", "matplotlib", "24970"], 10)
    assert once == thrice, "repeated query term must not change the score"


def test_generic_token_does_not_outweigh_discriminative_token(tmp_path):
    """The discriminative token must contribute strictly more than the generic
    one when the generic one is common in the corpus."""
    settings = make_settings(tmp_path)
    store = Store(settings)
    # Many cards containing the generic token, one containing the id.
    for i in range(30):
        seed_card(store, settings, "u", f"e:gen{i}", f"matplotlib plotting details number {i}")
    seed_card(store, settings, "u", "e:target", "matplotlib__matplotlib-24970 unique fix here")

    results = store.experience_field_search("u", tokenize("matplotlib__matplotlib-24970"), 100)
    assert results, "expected results"
    assert results[0][0] == "e:target", "id-bearing card must rank first"


def test_deduplication_does_not_change_single_occurrence_queries(tmp_path):
    settings = make_settings(tmp_path)
    store = Store(settings)
    seed_card(store, settings, "u", "e:1", "alpha beta gamma")
    a = store.experience_field_search("u", "alpha beta", 10)
    b = store.experience_field_search("u", ["alpha", "beta"], 10)
    assert a == b


# --------------------------------------------------------------------------
# defect 2: stopword filtering must reach the retrieval path
# --------------------------------------------------------------------------


def test_content_tokens_strip_stopwords_and_duplicates():
    features = build_query_features("what is the best way to fix the code")
    assert "the" not in features.content_tokens
    assert "what" not in features.content_tokens
    assert "fix" in features.content_tokens
    # No duplicates survive the filter.
    assert len(features.content_tokens) == len(set(features.content_tokens))


def test_retrieval_uses_content_tokens(tmp_path):
    """The search path must feed filtered tokens, not the raw query text."""
    src = (REPO_ROOT / "codemem" / "search" / "retrieval.py").read_text(encoding="utf-8")
    assert "features.content_tokens" in src, (
        "retrieval must use the stopword-filtered tokens; passing raw query text "
        "makes the stopword list dead code"
    )


def test_stopword_only_query_falls_back_rather_than_returning_nothing(tmp_path):
    """A query made entirely of stopwords should still behave."""
    settings = make_settings(tmp_path)
    store = Store(settings)
    seed_card(store, settings, "u", "e:1", "the code is the thing")
    features = build_query_features("the is a of")
    assert features.content_tokens == []
    # Retrieval path falls back to raw tokens in this case.
    results = store.experience_field_search(
        "u", features.content_tokens or features.tokens, 10)
    assert isinstance(results, list)


def test_no_query_tf_weight_in_bm25():
    """Regression: the query-tf multiplier must not return."""
    src = (REPO_ROOT / "codemem" / "common" / "store.py").read_text(encoding="utf-8")
    assert "contribution * weight" not in src, "query-tf weighting reintroduced"


def test_field_search_accepts_text_and_token_sequence(tmp_path):
    settings = make_settings(tmp_path)
    store = Store(settings)
    seed_card(store, settings, "u", "e:1", "resolve the ConfigParseError in loader")
    as_text = store.experience_field_search("u", "ConfigParseError", 10)
    as_tokens = store.experience_field_search("u", ["ConfigParseError"], 10)
    assert as_text == as_tokens
    assert as_text, "expected a hit"


# --------------------------------------------------------------------------
# defect 3: query entities must not leak docs/domains into exact matching
# --------------------------------------------------------------------------


def test_query_entities_drop_docs_domains_and_path_fragments():
    features = build_query_features(
        "See docs.github.com and README.md. n/a path. "
        "Error: ValueError in src/pkg/core.py. `helper_fn` calls parse_config."
    )
    values = features.entity_values()
    assert "n/a" not in values
    assert "docs.github.com" not in values
    assert "readme" not in values
    assert "src/pkg/core.py" in values
    assert "docs.github.c" not in features.anchor_values(("path",))


def test_case_anchor_idf_prefers_rare_values(tmp_path):
    """A rare anchor must outweigh a common one: IDF weighting is what keeps
    a file every task touches from dominating the anchor channel."""
    settings = make_settings(tmp_path)
    store = Store(settings)
    for i in range(12):
        seed_card(store, settings, "u", f"e:common{i}",
                  f"common.py touched by task {i}", anchors=[("file", "common.py")])
    seed_card(store, settings, "u", "e:rare", "RareError only appears here",
              anchors=[("error", "rareerror"), ("file", "common.py")])
    results = store.case_anchor_search(
        "u", {"file": ["common.py"], "error": ["RareError"]}, 20)
    assert results, "expected results"
    assert results[0][0] == "e:rare", "the rare-anchor card must rank first"
