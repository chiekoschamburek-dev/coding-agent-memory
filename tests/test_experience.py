"""Experience-layer tests (per-session card extraction).

Hermetic: a deterministic fake LLM stands in for gpt-4o-mini, so extraction,
reference validation, session finalisation, field recall and card-only return
are all verifiable without network access or credentials (the autouse
``_isolate_model_env`` fixture in conftest guarantees no real provider is
configured).

The invariants that matter:

* **Grounding is value-level**: condition values, files and symbols must occur in
  the supplied evidence; an unverifiable model ``basis`` is cleared. Structural
  fields prefer deterministic extraction over model narrative.
* **Cards are per session**: a session is finalised when a *different*
  session_id arrives, and Search finalises whatever is still pending. A
  re-opened session's finalisation replaces its previous cards.
* **Admission needs a repair**: problem and implement must both be non-empty
  and at least one cited chunk must exist.
* **Extraction is best-effort**: a failing or absent provider never fails Add.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from codemem.common.chunking import build_chunks  # noqa: E402
from codemem.common.config import Settings  # noqa: E402
from codemem.common.casebook import Condition, VerifyRecipe, build_packet  # noqa: E402
from codemem.add.experience import (  # noqa: E402
    ExperienceCard,
    build_cards,
    card_field_terms,
    card_field_texts,
    card_from_payload,
    card_to_payload,
    enrich_card,
    render_experience,
)
from codemem.common.guard import reset_guard_for_tests  # noqa: E402
from codemem.search.query import build_query_features  # noqa: E402
from codemem.search.retrieval import Candidate, SearchDiagnostics, SearchEngine  # noqa: E402
from codemem.service import MemoryService  # noqa: E402
from codemem.common.store import MessageInput, StoredChunk, Store, now_iso  # noqa: E402
from codemem.common.tokens import TokenCounter, reset_counter_for_tests  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        db_path=tmp_path / "exp.db",
        api_key=None,
        llm_base_url=None,
        llm_api_key=None,
        llm_model="gpt-4o-mini",
        embed_base_url=None,
        embed_api_key=None,
        embed_model="",
        add_sync_budget_s=40.0,
        search_hard_budget_s=20.0,
        token_budget_total=96_000,
        token_budget_target=64_000,
        experience_enabled=True,
    )
    base.update(overrides)
    return Settings(**base)


def make_stored_chunk(chunk_id: str, text: str, **overrides) -> StoredChunk:
    base = dict(
        chunk_id=chunk_id,
        user_id="u1",
        message_id="m1",
        ord=0,
        char_start=0,
        char_end=len(text),
        kind="prose",
        text=text,
        source_ts=None,
        ts_origin="received",
        session_id="s1",
        role="user",
    )
    base.update(overrides)
    return StoredChunk(**base)


def make_card(
    experience_id: str = "e1",
    kind: str = "debug",
    session_id: str = "s1",
    user_id: str = "u1",
) -> ExperienceCard:
    return ExperienceCard(
        experience_id=experience_id,
        user_id=user_id,
        session_id=session_id,
        kind=kind,
        extraction_version="card-v4",
        created_at="2026-01-01T00:00:00Z",
    )


def seed_ingestion(store: Store, settings: Settings, user_id: str, request_id: str,
                   text: str, session_id: str = "s1", mark_extracted: bool = False):
    """Commit one Add's chunks directly (no service, no LLM).

    ``mark_extracted=True`` simulates a session whose finalisation already
    happened, so a later Search does not try to extract it.
    """
    messages = [MessageInput("user", text)]
    chunks = build_chunks(user_id=user_id, request_id=request_id, messages=messages, settings=settings)
    state = store.begin_ingestion(
        user_id=user_id, request_id=request_id, session_id=session_id,
        digest=request_id, mode="full", lease_seconds=60,
    )
    store.commit_chunks(state=state, messages=messages, chunks=chunks, sync_ms=1)
    if mark_extracted:
        store.mark_session_extracted(
            user_id=user_id, session_id=session_id,
            extraction_version="t", card_count=0,
        )
    return state, chunks


_CHUNK_MARKER = re.compile(r"<<<CHUNK (\S+) \| session (\S+) \| role (\S+)>>>")


def parse_prompt_chunks(user: str) -> list[tuple[str, str]]:
    """Recover (chunk_id, text) pairs from the extraction prompt."""
    matches = list(_CHUNK_MARKER.finditer(user))
    out: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(user)
        seg = user[start:end]
        cut = seg.find("\n\n请抽取")
        if cut != -1:
            seg = seg[:cut]
        out.append((m.group(1), seg.strip("\n")))
    return out


def _quote_around(text: str, needle: str, pad: int = 16) -> str:
    i = text.find(needle)
    if i == -1:
        t = text.strip()
        return t[:40] if len(t) >= 6 else t
    start = max(0, i - pad)
    end = min(len(text), i + len(needle) + pad)
    quote = text[start:end].strip()
    return quote if len(quote) >= 6 else text.strip()[:40]


def configparse_builder(chunks: list[tuple[str, str]]) -> dict:
    target = next((c for c in chunks if "ConfigParseError" in c[1]), chunks[0])
    cid, text = target
    return {
        "cards": [
            {
                "kind": "debug",
                "problem": "ConfigParseError 配置解析失败",
                "locate": "src/loader.py aggregate",
                "implement": "在 aggregate 边界加保护",
                "files": ["src/loader.py"],
                "conditions": [
                    {
                        "type": "error",
                        "value": "ConfigParseError",
                        "basis": _quote_around(text, "ConfigParseError"),
                    }
                ],
                "source_chunk_ids": [cid],
            }
        ]
    }


def generic_builder(chunks: list[tuple[str, str]]) -> dict:
    cid, _text = chunks[0]
    return {
        "cards": [
            {
                "kind": "debug",
                "problem": "历史问题",
                "implement": "历史改法",
                "source_chunk_ids": [cid],
            }
        ]
    }


class FakeLLM:
    """Deterministic stand-in for the gpt-4o-mini calls."""

    configured = True

    def __init__(self, builder=None, fail: bool = False) -> None:
        self.builder = builder or generic_builder
        self.fail = fail
        self.calls = 0

    def complete_json(self, *, system, user, max_tokens=1024, timeout_s=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider boom")
        return self.builder(parse_prompt_chunks(user)), None


def _counter() -> TokenCounter:
    return TokenCounter(Settings(tokenizer_encoding="cl100k_base"))


# --------------------------------------------------------------------------
# A. experience.py pure functions
# --------------------------------------------------------------------------


def test_build_cards_accepts_payload_and_source_ids():
    chunks_by_id = {'c:1': make_stored_chunk('c:1', 'src/loader.py raised ConfigParseError')}
    cards = build_cards(
        raw={
            'cards': [{
                'problem': '配置加载报错',
                'locate': 'src/loader.py',
                'implement': '在 loader 中处理异常',
                'source_chunk_ids': ['c:1'],
            }]
        },
        chunks_by_id=chunks_by_id,
        user_id='u1',
        session_id='s1',
        id_for=lambda i: 'e1',
        extraction_version='card-v4',
        created_at='t',
        max_cards=4,
    )
    assert len(cards) == 1
    card = cards[0]
    assert card.problem == '配置加载报错'
    assert card.locate == 'src/loader.py'
    assert card.implement == '在 loader 中处理异常'
    assert card.source_chunk_ids == ('c:1',)
    payload = card_to_payload(card)
    assert payload['source_chunk_ids'] == ['c:1']
    # The removed duplicate/unused fields never come back on serialisation.
    for gone in ('symptom', 'intent', 'test', 'provenance', 'slots',
                 'action_stage', 'outcome'):
        assert gone not in payload


def test_new_card_render_uses_narrative_fields_and_source_chunk():
    chunk = make_stored_chunk('c:1', 'src/loader.py raised ConfigParseError')
    card = ExperienceCard(
        experience_id='e1',
        user_id='u1',
        session_id='s1',
        problem='配置加载报错',
        locate='src/loader.py',
        implement='在 loader 中处理异常',
        source_chunk_ids=('c:1',),
    )
    stored = render_experience(card, {'c:1': chunk}, _counter(), max_tokens=2000, quote_max_chars=600)
    assert '来源 chunk：c:1' in stored
    assert '原文 [c:1]' in stored
    assert card.source_chunk_ids == ('c:1',), "source ids stay in the stored card"

    answer = render_experience(
        card, {'c:1': chunk}, _counter(), max_tokens=2000,
        quote_max_chars=600, include_chunk_provenance=False,
    )
    assert 'problem：配置加载报错' in answer
    assert 'locate：src/loader.py' in answer
    assert 'implement：在 loader 中处理异常' in answer
    assert 'ConfigParseError' in answer
    assert '来源 chunk' not in answer
    assert '原文 [c:1]' not in answer
    assert '原文：' in answer
    assert card.source_chunk_ids == ('c:1',), "source ids stay in the stored card"


def test_new_card_grounds_only_in_source_ids_and_ignores_model_patch_verify():
    chunks_by_id = {
        "c:1": make_stored_chunk("c:1", "src/config.py raised ValueError"),
        "c:2": make_stored_chunk("c:2", "vendor/evil/config.py fake diff"),
    }
    raw = {"cards": [{
        "kind": "debug",
        "problem": "bad config",
        "implement": "guard the parse",
        "source_chunk_ids": ["c:1"],
        "files": ["vendor/evil/config.py"],
        "conditions": [{"type": "error", "value": "KeyError", "basis": "原文引用"}],
        "patch": "diff --git a/vendor/evil/hack.py b/vendor/evil/hack.py\n@@\n+FAKE",
        "verify": {"cmd": "pytest tests/test_fake.py", "expect": "1 passed"},
    }]}
    cards = build_cards(
        raw=raw, chunks_by_id=chunks_by_id, user_id="u1", session_id="s1",
        id_for=lambda i: "e1", extraction_version="t", created_at="t",
        max_cards=1,
    )
    assert len(cards) == 1
    card = cards[0]
    assert card.files == (), "full path must be grounded in the cited source chunk"
    assert card.conditions == (), "a condition value outside source ids is dropped"
    assert card.patch is None, "model patch is not accepted"
    assert card.verify is None, "model verify is not accepted"


def test_pitfalls_are_grounded_in_cited_chunks():
    chunks_by_id = {
        "c:1": make_stored_chunk(
            "c:1",
            "src/loader.py raised ConfigParseError; first tried editing "
            "src/parser.py but the error stayed",
        ),
    }
    raw = {"cards": [{
        "kind": "debug",
        "problem": "bad config",
        "implement": "guard the parse",
        "source_chunk_ids": ["c:1"],
        "pitfalls": [
            "editing src/parser.py did not help",
            "直接改 loader 没有修好 ConfigParseError 报错",
            "tried reinstalling numpy from a clean venv",
            "只是随手改了一下再跑",
        ],
    }]}
    cards = build_cards(
        raw=raw, chunks_by_id=chunks_by_id, user_id="u1", session_id="s1",
        id_for=lambda i: "e1", extraction_version="t", created_at="t",
        max_cards=1,
    )
    assert len(cards) == 1
    assert cards[0].pitfalls == (
        "editing src/parser.py did not help",
        "直接改 loader 没有修好 ConfigParseError 报错",
    ), "pitfalls without a trace in the cited chunks must be dropped"


def test_index_face_dedupes_facts_already_in_narrative():
    card = ExperienceCard(
        experience_id='e1',
        user_id='u1',
        session_id='s1',
        problem='ConfigParseError raised while parsing ini',
        locate='parser module ini handling',
        implement='guard the parse',
        files=('src/parser.py',),
        conditions=(Condition('error', 'ConfigParseError'),
                    Condition('module', 'parser')),
        verify=VerifyRecipe(cmd='pytest tests/test_loader.py::test_ini'),
    )
    texts = card_field_texts(card)
    assert texts['problem'].lower().count('configparseerror') == 1
    assert 'parser module ini handling' in texts['locate']
    assert not texts['locate'].endswith('parser'), \
        'a condition value already inside the narrative is not appended again'
    assert 'src/parser.py' in texts['locate'], 'new facts still widen the face'
    assert texts['test'].count('test_ini') == 1


def test_narrative_anchors_must_occur_in_cited_chunks():
    chunks_by_id = {
        'c:1': make_stored_chunk(
            'c:1', 'src/loader.py raised ConfigParseError in loader.parse'),
    }
    raw = {'cards': [{
        'kind': 'debug',
        'problem': 'ConfigParseError from loader.parse when reading ini',
        'locate': 'see vendor/evil/hack.py next',
        'implement': 'guard the parse',
        'source_chunk_ids': ['c:1'],
    }]}
    cards = build_cards(
        raw=raw, chunks_by_id=chunks_by_id, user_id='u1', session_id='s1',
        id_for=lambda i: 'e1', extraction_version='t', created_at='t',
        max_cards=1,
    )
    assert len(cards) == 1
    card = cards[0]
    assert 'vendor/evil/hack.py' not in (card.locate or ''), \
        'a path absent from the cited chunks is stripped from the narrative'
    assert card.locate == 'see next'
    assert card.problem and 'ConfigParseError' in card.problem
    assert 'loader.parse' in card.problem, 'grounded anchors survive'


def test_new_card_records_feature_kind_and_renders_it(tmp_path):
    chunk = make_stored_chunk(
        "c:1",
        "Support standard Axes in RGBAxes; add axes_class in "
        "lib/mpl_toolkits/axes_grid1/axes_rgb.py.",
    )
    raw = {"cards": [{
        "kind": "feature",
        "problem": "Support standard Axes in RGBAxes",
        "implement": "pass axes_class through",
        "files": ["lib/mpl_toolkits/axes_grid1/axes_rgb.py"],
        "source_chunk_ids": ["c:1"],
    }]}
    cards = build_cards(
        raw=raw, chunks_by_id={"c:1": chunk}, user_id="u1", session_id="s1",
        id_for=lambda i: "e1", extraction_version="t", created_at="2026-01-01T00:00:00Z",
        max_cards=4,
    )
    assert len(cards) == 1
    assert cards[0].kind == "feature"
    assert card_to_payload(cards[0])["kind"] == "feature"
    content = render_experience(cards[0], {"c:1": chunk}, _counter(), 2000, 600)
    assert "[历史修复案例]" in content
    assert "· feature" in content


def test_kind_is_inferred_when_extractor_omits_it():
    chunk = make_stored_chunk("c:1", "Support standard Axes in RGBAxes.")
    raw = {"cards": [{
        "problem": "Support standard Axes in RGBAxes",
        "implement": "add axes_class parameter",
        "source_chunk_ids": ["c:1"],
    }]}
    cards = build_cards(
        raw=raw, chunks_by_id={"c:1": chunk}, user_id="u1", session_id="s1",
        id_for=lambda i: "e1", extraction_version="t", created_at="2026-01-01T00:00:00Z",
        max_cards=4,
    )
    assert cards and cards[0].kind == "feature"


def test_feature_text_overrides_model_debug_kind():
    """gpt-4o-mini labels feature requests as debug; cues correct it."""
    chunk = make_stored_chunk("c:1", "Support standard Axes in RGBAxes.")
    raw = {"cards": [{
        "kind": "debug",
        "problem": "Support standard Axes in RGBAxes",
        "implement": "add axes_class parameter",
        "source_chunk_ids": ["c:1"],
    }]}
    cards = build_cards(
        raw=raw, chunks_by_id={"c:1": chunk}, user_id="u1", session_id="s1",
        id_for=lambda i: "e1", extraction_version="t", created_at="2026-01-01T00:00:00Z",
        max_cards=4,
    )
    assert cards and cards[0].kind == "feature"


def test_packet_structure_is_indexed_alongside_the_narrative_fields():
    """A packet card indexes its structure, not only its prose (stage 28).

    ``card_field_texts`` used to read the narrative fields *or* the packet
    fields.  Real packets fill both, so files, symbols, conditions, patch and
    verify never reached ``experience_terms`` -- and the structural values are
    the ones a query can repeat verbatim.
    """
    card = ExperienceCard(
        experience_id="e1", user_id="u1", session_id="s1",
        problem="check_array copied an int input that should stay float",
        locate="the dtype guard sits in the validator",
        implement="cast before the copy",
        files=("sklearn/utils/validation.py",),
        symbols=("check_array",),
        patch="+        X = X.astype(dtype, copy=False)",
        conditions=(
            Condition("error", "ValueError: Found input variables with inconsistent numbers"),
            Condition("test_name", "test_check_dtype"),
        ),
        verify=VerifyRecipe(cmd="pytest sklearn/tests/test_common.py", observed="1 failed"),
    )
    fields = card_field_texts(card_from_payload(card_to_payload(card)))
    assert "sklearn/utils/validation.py" in fields["locate"]
    assert "check_array" in fields["locate"]
    assert "inconsistent" in fields["problem"]
    assert "test_check_dtype" in fields["test"]
    assert "pytest" in fields["test"]
    assert "astype" in fields["implement"]

    terms = card_field_terms(card)
    assert terms and all("\x1f" in term for term in terms), "field-prefixed index"


# --------------------------------------------------------------------------
# B. store experience layer
# --------------------------------------------------------------------------


def _seed_multi_message(store, settings, texts):
    messages = [MessageInput("user", t) for t in texts]
    chunks = build_chunks(user_id="u1", request_id="r1", messages=messages, settings=settings)
    state = store.begin_ingestion(
        user_id="u1", request_id="r1", session_id="s1",
        digest="r1", mode="full", lease_seconds=60,
    )
    store.commit_chunks(state=state, messages=messages, chunks=chunks, sync_ms=1)
    return chunks


def test_fetch_session_chunks_preserves_source_order(store, settings):
    chunks = _seed_multi_message(store, settings, [
        "FIRST alpha one", "SECOND bravo two", "THIRD charlie three",
        "FOURTH delta four", "FIFTH echo five", "SIXTH foxtrot six",
    ])
    got = [c.chunk_id for c in store.fetch_session_chunks("u1", "s1", 100)]
    assert got == [c.chunk_id for c in chunks]


def test_fetch_session_chunks_orders_across_adds(store, settings):
    """The session timeline is (Add order, message index, chunk position)."""
    first = seed_ingestion(store, settings, "u1", "r1", "first add content", session_id="s1")
    second = seed_ingestion(store, settings, "u1", "r2", "second add content", session_id="s1")
    got = store.fetch_session_chunks("u1", "s1", -1)
    assert [c.chunk_id for c in first[1]] + [c.chunk_id for c in second[1]] == \
        [c.chunk_id for c in got]


def test_messages_are_not_persisted(store, settings):
    """The chunk is the minimal fact; raw message bodies never land in the DB."""
    seed_ingestion(store, settings, "u1", "r1", "SECRET message body content")
    tables = {
        row[0] for row in store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "messages" not in tables
    assert "postings" not in tables
    assert "chunk_vectors" not in tables
    assert "sessions" in tables
    rows = store.conn.execute(
        "SELECT chunk_count, status FROM sessions WHERE user_id='u1' AND session_id='s1'"
    ).fetchone()
    assert rows["chunk_count"] >= 1 and rows["status"] == "pending"


def test_fetch_evidence_chunks_preserves_source_order(store, settings):
    """The evidence window feeds patch/verify selection, so its order is load
    bearing: a scrambled window picks the wrong test command and truncates the
    wrong tail under LIMIT."""
    texts = [
        f"diff --git a/f{i}.py b/f{i}.py\n@@ -1 +1 @@\n-old{i}\n+new{i}"
        for i in range(6)
    ]
    chunks = _seed_multi_message(store, settings, texts)
    got = [c.chunk_id for c in store.fetch_evidence_chunks("u1", "s1", 100)]
    expected = [c.chunk_id for c in chunks]
    assert got == expected
    assert len(chunks) == len(expected), "every diff chunk should be evidence"


def test_long_card_fields_match_index_and_render_caps():
    chunk = make_stored_chunk("c:1", "src/loader.py raised ValueError")
    raw = {"cards": [{
        "kind": "debug",
        "problem": "P" * 400 + "RARE_PROBLEM",
        "implement": "I" * 400 + "RARE_IMPLEMENT",
        "source_chunk_ids": ["c:1"],
    }]}
    card = build_cards(
        raw=raw, chunks_by_id={"c:1": chunk}, user_id="u1", session_id="s1",
        id_for=lambda i: "e1", extraction_version="t", created_at="t",
        max_cards=1,
    )[0]
    assert len(card.problem) <= 320 and "RARE_PROBLEM" not in card.problem
    assert len(card.implement) <= 320 and "RARE_IMPLEMENT" not in card.implement
    fields = card_field_texts(card)
    assert "RARE_PROBLEM" not in fields["problem"]
    assert "RARE_IMPLEMENT" not in fields["implement"]
    content = render_experience(card, {"c:1": chunk}, _counter(), 2000, 600)
    assert "RARE_PROBLEM" not in content and "RARE_IMPLEMENT" not in content


def test_experience_bm25_dl_uses_indexed_terms_not_render_content(store, settings):
    state, _ = seed_ingestion(store, settings, "u1", "r1", "alpha")
    for eid, content in (("e1", "short"), ("e2", "long rendered packet " * 200)):
        store.insert_experience(
            user_id="u1", session_id="s1", experience_id=eid,
            ingestion_id=state.ingestion_id,
            kind="debug", payload_json="{}", full_content=content,
            extraction_version="t", created_at=now_iso(),
            terms=["problem\x1fa"], entities=[],
        )
    hits = dict(store.experience_field_search("u1", ["a"], 10))
    assert set(hits) == {"e1", "e2"}
    assert hits["e1"] == hits["e2"], "dl must come from experience_terms"


def test_evidence_fetch_includes_tail_verification(store, settings):
    texts = [f"$ pytest tests/test_{i}.py -q" for i in range(205)]
    texts.append("1 passed in 0.1s")
    chunks = _seed_multi_message(store, settings, texts)
    evidence = store.fetch_evidence_chunks("u1", "s1")
    assert evidence, "expected structural evidence"
    assert evidence[-1].chunk_id == chunks[-1].chunk_id, "tail result must not be cut at 200"
    from codemem.common.casebook import extract_verify
    assert extract_verify(evidence).expect.startswith("1 passed")


def test_experience_entity_search_is_scope_bound(store, settings):
    state, _ = seed_ingestion(store, settings, "u1", "r1", "text")
    store.insert_experience(
        user_id="u1", session_id="s1", experience_id="e1",
        ingestion_id=state.ingestion_id, kind="debug",
        payload_json="{}", full_content="x",
        extraction_version="card-v4", created_at=now_iso(), terms=["x"],
        entities=[("error", "configparseerror")],
    )
    assert store.experience_entity_search("u1", ["ConfigParseError"], 10)[0][0] == "e1"
    assert store.experience_entity_search("bob", ["ConfigParseError"], 10) == []


def test_purge_removes_experience(store, settings):
    state, _ = seed_ingestion(store, settings, "u1", "r1", "text")
    store.insert_experience(
        user_id="u1", session_id="s1", experience_id="e1",
        ingestion_id=state.ingestion_id, kind="debug",
        payload_json="{}", full_content="x",
        extraction_version="card-v4", created_at=now_iso(), terms=["x"], entities=[],
    )
    removed = store.purge_user("u1")
    assert removed.get("experiences", 0) >= 1
    assert removed.get("sessions", 0) == 1
    assert store.fetch_current_experiences("u1", ["e1"]) == {}


def test_experience_coverage_reports_cards(store, settings):
    seed_ingestion(store, settings, "u1", "r1", "text for coverage")
    cov = store.experience_coverage("u1")
    assert cov["chunks"] >= 1
    assert cov["experience_cards"] == 0


def test_delete_session_experiences_replaces_cards(store, settings):
    state, _ = seed_ingestion(store, settings, "u1", "r1", "text")
    store.insert_experience(
        user_id="u1", session_id="s1", experience_id="e:old",
        ingestion_id=state.ingestion_id, kind="debug",
        payload_json="{}", full_content="old",
        extraction_version="t", created_at=now_iso(), terms=["problem\x1fold"],
        entities=[], anchors=[("file", "a.py")],
    )
    removed = store.delete_session_experiences("u1", "s1")
    assert removed == 1
    assert store.fetch_current_experiences("u1", ["e:old"]) == {}
    assert store.case_anchor_search("u1", {"file": ["a.py"]}, 10) == []


# --------------------------------------------------------------------------
# C. service + retrieval end-to-end
# --------------------------------------------------------------------------


def _service(tmp_path, llm, **overrides):
    reset_guard_for_tests()
    reset_counter_for_tests()
    settings = make_settings(tmp_path, **overrides)
    store = Store(settings)
    return MemoryService(store, settings, llm_client=llm), store, settings


def test_session_finalizes_only_when_session_id_changes(tmp_path):
    def two_case_builder(chunks):
        ids = [cid for cid, _ in chunks[:2]]
        return {"cards": [
            {"kind": "debug", "problem": "first failure in a.py",
             "implement": "guard the loader", "files": ["a.py"],
             "source_chunk_ids": [ids[0]]},
            {"kind": "feature", "problem": "second change in b.py",
             "implement": "add the option", "files": ["b.py"],
             "source_chunk_ids": [ids[1]]},
        ]}

    svc, store, settings = _service(
        tmp_path, FakeLLM(builder=two_case_builder)
    )
    # First Add: s1 becomes pending, nothing finalises yet.
    outcome = svc.add(
        request_id="r1", user_id="u1", session_id="s1",
        raw_messages=[
            {"role": "user", "content": "first failure in a.py"},
            {"role": "user", "content": "second change in b.py"},
        ],
    )
    finalize = next(s for s in outcome.steps if s["name"] == "finalize")
    assert finalize["skipped"] is True, "the current session never finalises itself"
    assert store.experience_coverage("u1")["experience_cards"] == 0

    # A second Add for the SAME session: still pending, still no cards.
    svc.add(request_id="r2", user_id="u1", session_id="s1",
            raw_messages=[{"role": "user", "content": "more of the same session"}])
    assert store.experience_coverage("u1")["experience_cards"] == 0

    # A different session_id closes s1: its whole timeline is extracted now.
    outcome = svc.add(request_id="r3", user_id="u1", session_id="s2",
                      raw_messages=[{"role": "user", "content": "unrelated session"}])
    finalize = next(s for s in outcome.steps if s["name"] == "finalize")
    assert finalize.get("session") == "s1"
    assert finalize.get("cards") == 2
    assert store.experience_coverage("u1")["experience_cards"] == 2

    res = svc.search(user_id="u1", query="a.py b.py", top_k=100)
    ids = [item["id"] for item in res.data]
    assert len([value for value in ids if value.startswith("experience:")]) == 2


def test_search_backstop_finalizes_pending_tail_session(tmp_path):
    """The last session of an Add stream never sees a session_id change;
    Search must finalise it, or its memory is silently lost."""
    svc, store, settings = _service(tmp_path, FakeLLM(builder=configparse_builder))
    svc.add(
        request_id="r1", user_id="u1", session_id="s1",
        raw_messages=[{"role": "user", "content": "src/loader.py raised ConfigParseError near aggregate."}],
    )
    assert store.experience_coverage("u1")["experience_cards"] == 0

    res = svc.search(user_id="u1", query="ConfigParseError", top_k=100)
    assert res.diagnostics["finalized_sessions"] == 1
    assert store.experience_coverage("u1")["experience_cards"] >= 1
    assert res.data
    assert all(i.startswith("experience:") for i in (d["id"] for d in res.data))


def test_reopened_session_replaces_its_cards(tmp_path):
    """A session reopened by a later Add goes back to pending and its next
    finalisation replaces (not accumulates) the previous cards."""
    svc, store, settings = _service(tmp_path, FakeLLM(builder=configparse_builder))
    svc.add(request_id="r1", user_id="u1", session_id="s1",
            raw_messages=[{"role": "user", "content": "src/loader.py raised ConfigParseError near aggregate."}])
    svc.add(request_id="r2", user_id="u1", session_id="s2",
            raw_messages=[{"role": "user", "content": "another session"}])
    assert store.experience_coverage("u1")["experience_cards"] == 1

    # s1 reopens with more content; the next session change re-extracts it.
    svc.add(request_id="r3", user_id="u1", session_id="s1",
            raw_messages=[{"role": "user", "content": "src/loader.py raised ConfigParseError again in aggregate"}])
    svc.add(request_id="r4", user_id="u1", session_id="s3",
            raw_messages=[{"role": "user", "content": "closing both"}])
    cov = store.experience_coverage("u1")
    assert cov["experience_cards"] == 2, "s1 replaced its single card, s2 kept its own"
    rows = store.conn.execute(
        "SELECT session_id, COUNT(*) AS n FROM experiences WHERE user_id='u1' "
        "GROUP BY session_id ORDER BY session_id"
    ).fetchall()
    counts = {row["session_id"]: row["n"] for row in rows}
    assert counts.get("s1") == 1, "replacement must not accumulate stale cards"


def test_session_card_cap_is_three(tmp_path):
    def five_case_builder(chunks):
        return {"cards": [
            {"kind": "debug", "problem": f"problem {i}", "implement": f"fix {i}",
             "source_chunk_ids": [chunks[0][0]]}
            for i in range(5)
        ]}

    svc, store, settings = _service(tmp_path, FakeLLM(builder=five_case_builder))
    svc.add(request_id="r1", user_id="u1", session_id="s1",
            raw_messages=[{"role": "user", "content": "one problem with several angles"}])
    svc.add(request_id="r2", user_id="u1", session_id="s2",
            raw_messages=[{"role": "user", "content": "closing s1"}])
    assert store.experience_coverage("u1")["experience_cards"] == 3


def test_search_returns_case_packets_with_floor(tmp_path):
    svc, store, settings = _service(tmp_path, FakeLLM(builder=configparse_builder))
    svc.add(
        request_id="r1", user_id="u1", session_id="s1",
        raw_messages=[
            {"role": "user", "content": "While loading config, src/loader.py raised ConfigParseError near aggregate."},
            {"role": "user", "content": "unrelated_module_token checkpoint rotation happened later in the run."},
        ],
    )

    res = svc.search(user_id="u1", query="ConfigParseError unrelated_module_token", top_k=100)
    ids = [d["id"] for d in res.data]
    assert ids, ids
    assert all(i.startswith("experience:") for i in ids), ids
    # Single global ranking: scores never increase down the list.
    scores = [d["score"] for d in res.data]
    assert scores == sorted(scores, reverse=True)

    exp = next(d for d in res.data if d["id"].startswith("experience:"))
    # The card packet is self-contained: the reader must not need
    # to dereference a chunk id (the API only exposes content).
    assert "[历史修复案例]" in exp["content"]
    assert "问题：" in exp["content"]
    # The card's id is its stable identity: no version segment, because a card
    # is written once and a superseded one is deleted with its session.
    assert re.match(r"^experience:e:[0-9a-f]+$", exp["id"])


def test_cards_mode_never_returns_chunks(tmp_path):
    """Card-only recall: chunk mass cannot displace cards because chunks are
    never recalled as items at all."""
    svc, store, settings = _service(tmp_path, FakeLLM(builder=configparse_builder))
    svc.add(
        request_id="r1", user_id="u1", session_id="s1",
        raw_messages=[{"role": "user", "content": "src/loader.py raised ConfigParseError near aggregate."}],
    )
    # The search backstop finalises s1; from then on only cards exist for it.
    svc.search(user_id="u1", query="ConfigParseError", top_k=10)
    assert store.experience_coverage("u1")["experience_cards"] >= 1

    # Raw evidence that matches the query far more strongly than any card.
    # Sessions are pre-extracted so the search backstop does not extract them.
    for index in range(40):
        seed_ingestion(
            store, settings, "u1", f"noise-{index}",
            "ConfigParseError " * 30, session_id=f"n{index}", mark_extracted=True,
        )

    out = svc.search(user_id="u1", query="ConfigParseError", top_k=100)
    assert out.data
    assert all(item["id"].startswith("experience:") for item in out.data), out.data
    channels = out.diagnostics["channels"]
    assert "query_lexical" not in channels
    assert "entity" not in channels
    assert "experience_field" in channels


def test_search_without_cards_returns_empty(tmp_path):
    """No card in the scope: the response is empty, not a chunk dump."""
    svc, store, settings = _service(tmp_path, FakeLLM(fail=True))
    seed_ingestion(
        store, settings, "u1", "r1",
        "loader.py raised ConfigParseError while parsing aggregate options.",
    )
    assert store.experience_coverage("u1")["experience_cards"] == 0

    out = svc.search(user_id="u1", query="ConfigParseError", top_k=10)
    assert out.data == []


def test_extraction_runs_without_embeddings(tmp_path):
    """Extraction must not require the vector channel to be on."""
    svc, store, settings = _service(tmp_path, FakeLLM(builder=configparse_builder))
    assert settings.embeddings_enabled is False
    outcome = svc.add(
        request_id="r1", user_id="u1", session_id="s1",
        raw_messages=[{"role": "user", "content": "src/loader.py raised ConfigParseError near aggregate."}],
    )
    # No embed step exists anymore: only cards are ever embedded.
    assert all(s["name"] != "embed" for s in outcome.steps)
    svc.search(user_id="u1", query="ConfigParseError", top_k=10)
    assert store.experience_coverage("u1")["experience_cards"] >= 1


def test_extraction_failure_keeps_session_pending_and_add_alive(tmp_path):
    svc, store, settings = _service(tmp_path, FakeLLM(fail=True))
    outcome = svc.add(
        request_id="r1", user_id="u1", session_id="s1",
        raw_messages=[{"role": "user", "content": "src/a.py raised ConfigParseError in load_config"}],
    )
    assert outcome.success is True
    # The session stays pending; search retries and still fails cleanly.
    res = svc.search(user_id="u1", query="ConfigParseError", top_k=10)
    assert res.data == []
    assert store.experience_coverage("u1")["experience_cards"] == 0
    rows = store.conn.execute(
        "SELECT status FROM sessions WHERE user_id='u1' AND session_id='s1'"
    ).fetchone()
    assert rows["status"] == "pending"


def test_zero_card_extraction_marks_session_extracted(tmp_path):
    """A completed extraction that yields zero cards must not be retried on
    every Search."""
    def no_card_builder(chunks):
        return {"cards": []}

    svc, store, settings = _service(tmp_path, FakeLLM(builder=no_card_builder))
    svc.add(request_id="r1", user_id="u1", session_id="s1",
            raw_messages=[{"role": "user", "content": "reviewed the flaky dashboard rerender output"}])
    llm = svc._llm
    svc.search(user_id="u1", query="anything", top_k=10)
    assert llm.calls >= 1
    calls_after_first = llm.calls
    svc.search(user_id="u1", query="anything", top_k=10)
    assert llm.calls == calls_after_first, "an extracted session is not re-extracted"
    rows = store.conn.execute(
        "SELECT status, card_count FROM sessions WHERE user_id='u1' AND session_id='s1'"
    ).fetchone()
    assert rows["status"] == "extracted" and rows["card_count"] == 0


def test_extraction_skipped_when_llm_not_configured(tmp_path):
    reset_guard_for_tests()
    reset_counter_for_tests()
    settings = make_settings(tmp_path)
    store = Store(settings)
    svc = MemoryService(store, settings)  # no llm_client injected, none configured
    outcome = svc.add(
        request_id="r1", user_id="u1", session_id="s1",
        raw_messages=[{"role": "user", "content": "src/a.py raised ConfigParseError"}],
    )
    finalize = next(s for s in outcome.steps if s["name"] == "finalize")
    assert finalize["skipped"] is True


def test_select_dedups_identical_text(store, settings):
    """Exact/near-duplicate text still collapses by whitespace-normalised hash."""
    engine = SearchEngine(store, settings)
    a = Candidate(item_id="e:1", kind="experience", text="identical evidence text here",
                  channels={}, fused_score=2.0)
    b = Candidate(item_id="e:2", kind="experience", text="identical   evidence text here",
                  channels={}, fused_score=1.0)
    diag = SearchDiagnostics()
    selected = engine._select([a, b], diag, final_limit=10)
    assert [c.item_id for c in selected] == ["e:1"]
    assert diag.dropped_duplicate >= 1


def test_experience_per_item_cap_is_wider(settings):
    for rank in (1, 5, 20, 50, 90):
        assert settings.experience_per_item_token_cap(rank) >= settings.per_item_token_cap(rank)


def test_condition_type_outside_the_enum_is_dropped():
    chunks_by_id = {
        "c:1": make_stored_chunk("c:1", "src/config.py raised ValueError at line 42"),
    }
    raw = {"cards": [{
        "kind": "debug",
        "problem": "bad config",
        "implement": "validate the config",
        "source_chunk_ids": ["c:1"],
        "conditions": [
            {"type": "line", "value": "42", "basis": "42"},
            {"type": "error", "value": "ValueError", "basis": "ValueError"},
        ],
    }]}
    cards = build_cards(
        raw=raw, chunks_by_id=chunks_by_id, user_id="u1", session_id="s1",
        id_for=lambda _index: "e1", extraction_version="t", created_at="t",
        max_cards=1,
    )
    assert [condition.type for condition in cards[0].conditions] == ["error"]


def test_admission_requires_problem_and_implement():
    """A card needs both a query face (problem) and a repair (implement)."""
    chunks = {
        "c:1": make_stored_chunk(
            "c:1", "src/loader.py raised ConfigParseError while loading config"
        )
    }

    def build(raw_card):
        return build_cards(
            raw={"cards": [raw_card]},
            chunks_by_id=chunks,
            user_id="u1",
            session_id="s1",
            id_for=lambda index: f"e:{index}",
            extraction_version="card-v4",
            created_at="2026-01-01T00:00:00Z",
            max_cards=3,
        )

    # No problem: no query face.
    assert build({
        "kind": "debug",
        "implement": "changed load_config to validate the aggregate option",
        "source_chunk_ids": ["c:1"],
    }) == []
    # No implement: a phenomenon report, not a repair case.
    assert build({
        "kind": "debug",
        "problem": "ConfigParseError on src/loader.py",
        "source_chunk_ids": ["c:1"],
    }) == []

    kept = build({
        "kind": "debug",
        "problem": "ConfigParseError on src/loader.py",
        "implement": "changed load_config to validate the aggregate option",
        "source_chunk_ids": ["c:1"],
    })
    assert len(kept) == 1 and kept[0].problem


def test_rank_ignores_verification_status(store, settings):
    """Verification status is not a global ranking input: two cards differing
    only in their verify recipe rank by their channels."""
    engine = SearchEngine(store, settings)
    card_a = make_card(experience_id="a")
    card_b = ExperienceCard(
        experience_id="b", user_id="u1", session_id="s1",
        verify=VerifyRecipe(cmd="pytest x"),
    )
    pool = {
        "experience:a": Candidate(
            item_id="experience:a", kind="experience", text="t",
            channels={"experience_field": 1}, fused_score=1.0,
            payload={"card": card_a, "chunks_by_id": {}},
        ),
        "experience:b": Candidate(
            item_id="experience:b", kind="experience", text="t",
            channels={"experience_field": 1}, fused_score=1.0,
            payload={"card": card_b, "chunks_by_id": {}},
        ),
    }
    ranked = engine._rank(pool, build_query_features("query"))
    scores = {c.item_id: c.fused_score for c in ranked}
    assert scores["experience:a"] == pytest.approx(scores["experience:b"])


# --------------------------------------------------------------------------
# write-time vs query-time packet parity
# --------------------------------------------------------------------------

_DIFF_ONLY_CHUNK = """diff --git a/src/loader.py b/src/loader.py
--- a/src/loader.py
+++ b/src/loader.py
@@ -12,7 +12,7 @@ def load_config(path):
-        return {}
+        return defaults()
"""


def _uncited_pool():
    """A session whose diff and test command sit outside the cited chunks.

    The extractor cites what it read; the diff is normally in a later chunk it
    did not cite.  Search re-fetches only the cited ids, so this is the shape
    the query path actually sees.
    """
    cited = [
        make_stored_chunk('c:1', 'ConfigParseError raised while loading the config'),
        make_stored_chunk('c:2', 'the fix belongs in load_config, which returns an empty dict'),
    ]
    uncited = [
        make_stored_chunk('c:3', _DIFF_ONLY_CHUNK, kind='diff', message_id='m2', ord=0),
        make_stored_chunk('c:4', '$ pytest tests/test_loader.py -q\n1 passed in 0.3s',
                          kind='command', message_id='m3', ord=0),
    ]
    return cited, uncited


def test_query_render_keeps_structure_derived_from_uncited_chunks():
    """The served packet must equal the stored one on files/patch/verify.

    ``build_packet`` re-derives structure from whatever pool it is given.  The
    write path passes the whole session, Search passes only the cited chunks,
    so without reusing the persisted structure the answer loses the
    problem->file mapping, the historical change and the verification command
    -- exactly the three things the answering agent cannot reconstruct.
    """
    cited, uncited = _uncited_pool()
    pool = {c.chunk_id: c for c in (*cited, *uncited)}
    raw = ExperienceCard(
        experience_id='e1', user_id='u1', session_id='s1', kind='debug',
        problem='config load raises ConfigParseError',
        locate='load_config',
        implement='return the defaults instead of an empty dict',
        source_chunk_ids=tuple(c.chunk_id for c in cited),
        created_at='2026-01-01T00:00:00Z',
    )

    card = enrich_card(raw, list(pool.values()))
    assert card.patch and 'defaults()' in card.patch, 'write path derives the diff'
    assert card.verify and card.verify.cmd, 'write path derives the recipe'
    stored = render_experience(card, pool, _counter(), max_tokens=2000, quote_max_chars=600)

    reloaded = card_from_payload(card_to_payload(card))
    served = render_experience(
        reloaded, {c.chunk_id: c for c in cited}, _counter(), max_tokens=2000,
        quote_max_chars=600, include_chunk_provenance=False, persisted=True,
    )

    for marker in ('文件：src/loader.py', '历史改法', '修改前：', '修改后：',
                   'defaults()', 'pytest tests/test_loader.py'):
        assert marker in stored, f'write-time render lost {marker}'
        assert marker in served, f'query-time render lost {marker}'


def test_stored_patch_outranks_a_hunk_less_reextraction():
    """A cited chunk holding only the diff header must not mask the real patch.

    The chunker splits a patch across chunks, so the cited subset can contain a
    bare ``diff --git`` line.  ``extract_patch`` then returns a hunk-less
    section: non-empty, and -- if the fallback only fired on an empty result --
    it would hide the diff that was stored at write time.
    """
    header = make_stored_chunk('c:1', 'diff --git a/src/loader.py b/src/loader.py')
    card = ExperienceCard(
        experience_id='e1', user_id='u1', session_id='s1', kind='debug',
        problem='config load fails', locate='load_config', implement='return defaults',
        source_chunk_ids=('c:1',), patch=_DIFF_ONLY_CHUNK,
    )
    packet = build_packet(card, [header], persisted=True)
    assert 'defaults()' in packet.patch
    assert len(packet.patch.splitlines()) > 1, 'the hunk body survives'


def test_write_path_never_reuses_a_card_supplied_patch():
    """``persisted`` is the only door into the stored structure.

    A card straight out of the extractor carries no patch/verify even when the
    model wrote one; accepting it would bypass path grounding and could invent
    files.  The fallback is keyed on the read path, not on the value being set.
    """
    chunk = make_stored_chunk('c:1', 'load_config raised ConfigParseError')
    card = ExperienceCard(
        experience_id='e1', user_id='u1', session_id='s1', kind='debug',
        problem='config load fails', locate='load_config', implement='return defaults',
        source_chunk_ids=('c:1',),
        patch='diff --git a/invented/nowhere.py b/invented/nowhere.py',
        verify=VerifyRecipe(cmd='pytest invented_test.py'),
        files=('invented/nowhere.py',),
    )
    packet = build_packet(card, [chunk])
    assert packet.patch == ''
    assert packet.verify.is_empty()
    assert 'invented/nowhere.py' not in packet.files


# --- grounding/admission regressions --------------------------------------


def test_stripped_empty_face_rejects_card():
    """Admission must hold after the grounding clamp, not only before it.

    A card whose implement is entirely ungrounded symbols passes the classic
    gate, then gets clamped to None -- it must not be born as a silent
    phenomenon report."""
    from codemem.add.experience import _new_card_from_raw

    chunk = make_stored_chunk("c:1", "plain session text without anchors")
    card = _new_card_from_raw(
        {
            "kind": "debug",
            "problem": "breaks GhostThing.handler on save",
            "implement": "ghostsymlink.fakelib.FakeClass",
            "source_chunk_ids": ["c:1"],
        },
        chunks_by_id={"c:1": chunk},
        user_id="u1",
        session_id="s1",
        experience_id="e:x",
        extraction_version="card-v4",
        created_at="",
    )
    assert card is None


def test_path_regex_leaves_no_extension_residue():
    """.h is a C-header extension; it must not bite into ``.handler`` and
    leave a half-symbol fragment behind."""
    from codemem.add.experience import _strip_ungrounded_anchors

    out = _strip_ungrounded_anchors(
        "breaks GhostThing.handler on save", "no anchors here")
    assert "andler" not in (out or "")
    assert "GhostThing.handler" not in (out or "")


def test_strip_keeps_english_abbreviations():
    """Single-letter dotted segments are abbreviations (e.g. / i.e.), not
    qualified symbols; stripping them mangles the injected text."""
    from codemem.add.experience import _strip_ungrounded_anchors

    out = _strip_ungrounded_anchors(
        "Fails on Python 3.12, e.g. when subclasses override the parser",
        "no anchors here")
    assert "e.g." in out
    real = _strip_ungrounded_anchors(
        "handled by sympy_parser.EvaluateFalseTransformer", "no anchors here")
    assert "EvaluateFalseTransformer" not in real


def test_setup_filter_keeps_corpus_work_chunks():
    """300Lite work paths carry a ./swebench_<ts>/testbed/ prefix; the setup
    filter must match setup actions, never the corpus path prefix."""
    from codemem.add.experience import is_setup_chunk

    assert not is_setup_chunk(
        '[tool:Read] {"file_path": "./swebench_9_15/testbed/django/db/models/query.py"}')
    assert not is_setup_chunk(
        "cd ./swebench_9_15/testbed && python -m pytest tests/test_query.py -q")
    assert not is_setup_chunk(
        'find ./swebench_9_15/testbed -type f -name "*.py" | head')
    assert is_setup_chunk("git clone https://github.com/django/django.git testbed")
    assert is_setup_chunk("Collecting django==4.2 (from -r requirements.txt)")
    assert is_setup_chunk("pytest 7.0 or higher is required")


def test_card_from_payload_drops_enum_foreign_condition_types():
    """The read path applies the same closed enum as the write path, so a
    rogue type in an old payload cannot render into the injected text."""
    payload = {
        "experience_id": "e1", "session_id": "s1", "kind": "debug",
        "problem": "p", "implement": "i",
        "conditions": [
            {"type": "bogus", "value": "ZZZ"},
            {"type": "error", "value": "ValueError: bad"},
        ],
    }
    card = card_from_payload(payload)
    assert [c.type for c in card.conditions] == ["error"]
    assert all(
        kind != "bogus" for kind, _ in
        [(c.type, c.value) for c in card.conditions]
    )


# --- card-v4: three-part implement spec -----------------------------------


def _spec_raw_card():
    return {
        "kind": "debug",
        "problem": "mask propagation fails when one operand has no mask",
        "mechanism": "handle_mask receives None; confirmed by TypeError trace",
        "implement": "",
        "implement_shape": "None-guard sits before handle_mask dispatch in _arithmetic_mask",
        "implement_boundaries": "single-side None returns copy of other mask; both-None keeps NoneType branch",
        "implement_acceptance": "bitwise_or on (masked, None) must not raise; result mask is deepcopy",
        "source_chunk_ids": ["c:1"],
    }


def _spec_chunk():
    return make_stored_chunk(
        "c:1", "_arithmetic_mask handle_mask deepcopy operand.mask "
               "assert result.mask is not data.mask TypeError test_mask_none")


def test_shape_only_card_admitted_narrative_only_rejected():
    from codemem.add.experience import _new_card_from_raw
    chunk = _spec_chunk()
    kwargs = dict(chunks_by_id={"c:1": chunk}, user_id="u", session_id="s",
                  experience_id="e:x", extraction_version="card-v4", created_at="")
    card = _new_card_from_raw(_spec_raw_card(), **kwargs)
    assert card is not None and card.implement is None
    assert card.implement_shape.startswith("None-guard")
    assert card.implement_acceptance
    bare = {**_spec_raw_card(), "implement_shape": "",
                                 "implement_boundaries": "",
                                 "implement_acceptance": ""}
    assert _new_card_from_raw(bare, **kwargs) is None


def test_spec_payload_roundtrip():
    from codemem.add.experience import card_from_payload, card_to_payload
    card = card_from_payload({**_spec_raw_card(), "experience_id": "e1",
                              "session_id": "s1"})
    assert card.implement_boundaries
    payload = card_to_payload(card)
    assert payload["implement_acceptance"]


def test_digest_extracts_diff_and_assert_lines():
    from codemem.add.experience import _implement_evidence_digest
    chunks = [
        make_stored_chunk("c:1", "diff --git a/x.py b/x.py\n"
                                 "@@ -10,3 +10,5 @@ def foo:\n"
                                 "+        if mask is None:\n"
                                 "+            return deepcopy(self.mask)"),
        make_stored_chunk("c:2", "def test_mask_none():\n"
                                 "    assert result.mask is not data.mask\n"
                                 "    assert not np.shared_memory"),
    ]
    digest = _implement_evidence_digest(chunks)
    assert "diff 分支结构" in digest and "@@ -10,3 +10,5 @@ def foo" in digest
    assert "assert result.mask is not data.mask" in digest


def test_index_face_includes_acceptance():
    from codemem.add.experience import card_field_texts, card_from_payload
    card = card_from_payload({**_spec_raw_card(), "experience_id": "e1",
                              "session_id": "s1"})
    assert "deepcopy" in card_field_texts(card)["implement"]
