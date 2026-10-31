"""Hermetic card-v5-minimal tests: idea-level cards, no replayable action."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codemem.common.config import Settings
from codemem.add.experience import (
    build_cards,
    build_extraction_prompt,
    card_anchors,
    card_entities,
    card_field_terms,
    card_field_texts,
    card_from_payload,
    card_to_payload,
    extraction_system_prompt,
    render_experience,
)
from codemem.common.guard import reset_guard_for_tests
from codemem.add.minimal import (
    PAYLOAD_VERSION,
    VERSION,
    build_evidence,
    prepare_minimal,
    render_minimal,
)
from codemem.service import MemoryService
from codemem.common.store import Store, StoredChunk
from codemem.common.tokens import reset_counter_for_tests

ISSUE = (
    "Widgets become unresponsive to mouse input after pyplot.clf() is called "
    "inside a RangeSlider's on_changed callback. The widgets should still "
    "receive input after the figure is redrawn."
)
SUMMARY = """## Summary

The session reproduced a widget-input regression after a redraw.

### Root Cause

The mouse grab is not released when pyplot.clf() destroys the axes, so the
canvas keeps routing input to an axes that no longer exists. New widgets are
created but never receive mouse events because the canvas still believes the
destroyed axes owns the grab.

### The Fix

Add self.canvas.release_mouse(ax) in lib/matplotlib/figure.py at line 934.

### Testing

Run pytest lib/matplotlib/tests/test_backend_bases.py.
"""
SUMMARY_LONG = SUMMARY + ("The root cause did not involve the widget's own event handler. " * 5)
TEST_DIFF = """diff --git a/lib/matplotlib/tests/test_backend_bases.py b/lib/matplotlib/tests/test_backend_bases.py
+def test_grab_clear():
+    fig, ax = plt.subplots()
+    fig.canvas.grab_mouse(ax)
+    assert fig.canvas.mouse_grabber == ax
+    fig.clear()
+    assert fig.canvas.mouse_grabber is None
"""
TEST_OUTPUT = "[tool_result]\n1 passed in 0.12s"
CLEAN = {
    "problem": "Widgets become unresponsive to mouse input after pyplot.clf() "
               "inside a RangeSlider callback. The widgets should still receive input "
               "after the figure is redrawn.",
    "mechanism": "The mouse grab is not released when pyplot.clf() destroys the axes, "
                 "so the canvas keeps routing input to an axes that no longer exists.",
    "verify": {
        "invariants": [
            "After a redraw, widgets still receive mouse input.",
            "The canvas has no mouse grabber after the old axes is destroyed.",
        ],
        "boundaries": [
            {"case": "A redraw happens while a slider handle is dragged",
             "expected": "The new widgets still receive mouse input."},
        ],
    },
}


def chunk(cid: str, text: str, *, message_id: str = "m0", role: str = "user",
          kind: str = "prose", session_id: str = "s", user_id: str = "u",
          ord: int = 0) -> StoredChunk:
    return StoredChunk(
        chunk_id=cid, user_id=user_id, message_id=message_id, ord=ord,
        char_start=0, char_end=len(text), kind=kind, text=text,
        source_ts=None, ts_origin="received", session_id=session_id, role=role,
    )


def fixture_evidence():
    chunks = [
        chunk("issue", ISSUE, message_id="m0", role="user"),
        chunk("summary", SUMMARY_LONG, message_id="m9", role="assistant"),
        chunk("test_diff", TEST_DIFF, message_id="m8", role="user", kind="diff"),
        chunk("test_out", TEST_OUTPUT, message_id="m7", role="user"),
    ]
    return chunks


def settings_for(tmp_path, **kwargs):
    reset_counter_for_tests()
    reset_guard_for_tests()
    defaults = {
        "db_path": tmp_path / "minimal.db",
        "experience_extraction_version": VERSION,
        "api_key": None, "llm_base_url": None, "llm_api_key": None,
        "embed_base_url": None, "embed_api_key": None, "embed_model": "",
        "rerank_base_url": None, "applicability_enabled": False,
        "add_sync_budget_s": 40, "search_hard_budget_s": 20,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


def build(raw=None, chunks=None):
    raw = dict(CLEAN if raw is None else raw)
    chunks = fixture_evidence() if chunks is None else chunks
    return build_cards(
        raw={"cards": [raw]},
        chunks_by_id={c.chunk_id: c for c in chunks},
        grounding_chunks_by_id={c.chunk_id: c for c in chunks},
        user_id="u", session_id="s",
        id_for=lambda i: f"e{i}", extraction_version=VERSION,
        created_at="t", max_cards=1,
    )


# --------------------------------------------------------------------------
# evidence selection
# --------------------------------------------------------------------------


def test_evidence_prefers_final_summary_and_drops_fix_sections():
    chunks = fixture_evidence()
    evidence = build_evidence(chunks)
    assert evidence.is_usable()
    assert [c.message_id for c in evidence.summary_chunks] == ["m9"]
    assert "Root Cause" in evidence.summary_sections
    assert "release_mouse" not in evidence.summary_sections
    assert "The Fix" not in evidence.summary_sections
    assert "Testing" not in evidence.summary_sections
    assert "release_mouse" not in evidence.root_cause
    assert "mouse grab is not released" in evidence.root_cause
    assert "Run pytest" in evidence.verify_excerpt
    assert "test_grab_clear" in evidence.test_digest
    assert "assert fig.canvas.mouse_grabber is None" in evidence.test_digest


def test_evidence_ignores_a_longer_tool_write_summary():
    chunks = fixture_evidence()
    chunks.append(chunk("tool_summary", "[tool:Write] {\"file_path\": \"SOLUTION_SUMMARY.md\", "
                        "\"content\": \"## Root Cause\\n" + ("long " * 400) + "\"}",
                        message_id="m10", role="assistant"))
    evidence = build_evidence(chunks)
    assert [c.message_id for c in evidence.summary_chunks] == ["m9"]


# --------------------------------------------------------------------------
# admission
# --------------------------------------------------------------------------


def test_admission_accepts_clean_idea_card():
    card, problems = prepare_minimal(CLEAN, fixture_evidence(), user_id="u", session_id="s")
    assert problems == [] and card is not None
    assert card.problem.startswith("Widgets")
    assert card.verify.invariants and card.verify.boundaries
    assert card.source_chunk_ids == ("issue", "summary", "test_diff", "test_out")


@pytest.mark.parametrize("change,expected", [
    ("mechanism_fix", "历史修复动作"),
    ("mechanism_fix_call", "历史修复动作"),
    ("mechanism_path", "文件路径"),
    ("mechanism_line", "行号"),
    ("verify_command", "命令"),
    ("verify_path", "文件路径"),
    ("verify_fix", "历史修复动作"),
])
def test_admission_rejects_replayable_action(change, expected):
    raw = json.loads(json.dumps(CLEAN))
    if change == "mechanism_fix":
        raw["mechanism"] = "The fix adds a line in the delaxes method to release the grab."
    elif change == "mechanism_fix_call":
        raw["mechanism"] = "The fix calls self.canvas.release_mouse(ax) when the axes dies."
    elif change == "mechanism_path":
        raw["mechanism"] = "The mouse grab is not released in lib/matplotlib/figure.py."
    elif change == "mechanism_line":
        raw["mechanism"] = "The mouse grab is not released at line 934."
    elif change == "verify_command":
        raw["verify"]["invariants"] = ["Run pytest tests/test_backend_bases.py to verify the change."]
    elif change == "verify_path":
        raw["verify"]["invariants"] = ["The test in tests/test_backend_bases.py passes."]
    elif change == "verify_fix":
        raw["verify"]["invariants"] = ["The fix prevents the grab from surviving the redraw."]
    problems = prepare_minimal(raw, fixture_evidence(), user_id="u", session_id="s")[1]
    assert any(expected in problem for problem in problems), problems


def test_admission_allows_buggy_code_facts_in_mechanism():
    """A mechanism may name the buggy assignment/call; it is not a patch."""
    raw = json.loads(json.dumps(CLEAN))
    raw["mechanism"] = (
        "When the figure is cleared, the canvas keeps its mouse_grabber value; a later "
        "canvas.grab_mouse(ax) call is routed to the destroyed axes instead of the new widgets."
    )
    card, problems = prepare_minimal(raw, fixture_evidence(), user_id="u", session_id="s")
    assert problems == [] and card is not None


def test_admission_allows_issue_title_with_after_changing():
    raw = json.loads(json.dumps(CLEAN))
    raw["problem"] = ("Update widget mouse handling after changing the slider callback: "
                      "the widgets stop receiving input.")
    card, problems = prepare_minimal(raw, fixture_evidence(), user_id="u", session_id="s")
    assert problems == [] and card is not None


def test_admission_requires_verify_and_grounds_symbols():
    raw = json.loads(json.dumps(CLEAN))
    raw["verify"] = {"invariants": [], "boundaries": []}
    assert any("verify" in p for p in prepare_minimal(
        raw, fixture_evidence(), user_id="u", session_id="s")[1])

    raw = json.loads(json.dumps(CLEAN))
    raw["verify"] = {"invariants": [], "boundaries": [
        {"case": "A redraw", "expected": "Widgets remain responsive."}]}
    problems = prepare_minimal(raw, fixture_evidence(), user_id="u", session_id="s")[1]
    assert any("invariants" in problem for problem in problems), problems

    raw = json.loads(json.dumps(CLEAN))
    raw["mechanism"] += " The ghost_helper_xyz flag is involved."
    problems = prepare_minimal(raw, fixture_evidence(), user_id="u", session_id="s")[1]
    assert any("ghost_helper_xyz" in p for p in problems), problems


def test_string_boundary_is_split_into_case_and_expected():
    raw = json.loads(json.dumps(CLEAN))
    raw["verify"]["boundaries"] = [
        "Unpickling a figure → Expected: the DPI remains unchanged."
    ]
    card, problems = prepare_minimal(raw, fixture_evidence(), user_id="u", session_id="s")
    assert problems == [] and card is not None
    boundary = card.verify.boundaries[0]
    assert boundary.case == "Unpickling a figure"
    assert boundary.expected == "the DPI remains unchanged."


def test_english_abbreviation_is_not_a_code_anchor():
    raw = json.loads(json.dumps(CLEAN))
    raw["verify"]["invariants"].append(
        "Overflow produces the documented result (e.g., 257 -> 1 for uint8).")
    card, problems = prepare_minimal(raw, fixture_evidence(), user_id="u", session_id="s")
    assert problems == [] and card is not None


def test_boundary_without_expected_is_rejected():
    raw = json.loads(json.dumps(CLEAN))
    raw["verify"]["boundaries"] = [{"case": "A redraw happens"}]
    problems = prepare_minimal(raw, fixture_evidence(), user_id="u", session_id="s")[1]
    assert any("expected" in problem for problem in problems), problems


def test_admission_rejects_foreign_session_chunk():
    chunks = fixture_evidence()
    chunks[-1] = chunk("test_out", TEST_OUTPUT, session_id="other")
    card, problems = prepare_minimal(CLEAN, chunks, user_id="u", session_id="s")
    assert card is None
    assert any("其他用户/会话" in p for p in problems)


# --------------------------------------------------------------------------
# payload / index / render
# --------------------------------------------------------------------------


def test_payload_roundtrip_is_lossless_and_versioned():
    card = build()[0]
    assert card.is_minimal and card.has_narrative_fields
    payload = card_to_payload(card)
    assert payload["payload_version"] == PAYLOAD_VERSION
    assert payload["extraction_version"] == VERSION
    assert payload["verify"]["invariants"] == list(card.verify_semantics.invariants)
    assert card_from_payload(json.loads(json.dumps(payload))) == card


def test_index_face_is_problem_mechanism_verify_only():
    card = build()[0]
    fields = card_field_texts(card)
    assert set(fields) == {"problem", "mechanism", "locate", "implement", "test"}
    assert fields["locate"] == fields["implement"] == ""
    assert "mouse grab" in fields["mechanism"]
    assert "mouse input" in fields["test"]
    texts = json.dumps([fields, card_field_terms(card),
                        card_entities(card, {}), card_anchors(card)])
    assert "release_mouse" not in texts
    assert "lib/matplotlib/figure.py" not in texts
    assert "mouse input" in texts


def test_minimal_render_is_the_idea_card():
    card = build()[0]
    counter = SimpleNamespace(count=lambda text: len(text), truncate=lambda *_: "truncated")
    content = render_experience(card, {}, counter, 5000, 600)
    assert "问题：" in content and "机制（历史因果解释" in content
    assert "验收语义（历史验收" in content
    assert "release_mouse" not in content and "test_grab_clear" not in content
    assert "The Fix" not in content
    assert render_minimal(card, counter, 5000) == content


def test_render_omits_instead_of_amputating():
    card = build()[0]
    counter = SimpleNamespace(count=lambda text: len(text), truncate=lambda *_: "truncated")
    assert render_minimal(card, counter, 10) == ""


def test_packet_builder_refuses_minimal():
    from codemem.common.casebook import build_packet

    with pytest.raises(ValueError, match="card-v5-minimal"):
        build_packet(build()[0], [])


# --------------------------------------------------------------------------
# extraction dispatch
# --------------------------------------------------------------------------


def test_prompt_dispatch_uses_minimal_contract():
    chunks = fixture_evidence()
    prompt = build_extraction_prompt(chunks, 600, extraction_version=VERSION)
    assert "card-v5-minimal" in prompt
    assert "机制" in prompt and "验收语义" in prompt
    assert "implement_shape" not in prompt and "locate" not in prompt
    system = extraction_system_prompt(VERSION)
    assert "card-v5-minimal" in system
    assert "不输出历史动作" in system


class MinimalLLM:
    configured = True

    def __init__(self, cards=None):
        self.calls = []
        self.cards = [CLEAN] if cards is None else cards

    def complete_json(self, *, system, user, **kwargs):
        self.calls.append((system, user))
        return {"cards": self.cards}, None


def _messages():
    return [
        {"role": "user", "content": ISSUE},
        {"role": "assistant", "content": "I will inspect the widget event path. " * 8},
        {"role": "assistant", "content": SUMMARY_LONG},
        {"role": "user", "content": TEST_OUTPUT},
        {"role": "user", "content": TEST_DIFF},
    ]


def test_service_add_finalize_search_end_to_end(tmp_path):
    settings = settings_for(tmp_path)
    store = Store(settings)
    llm = MinimalLLM()
    service = MemoryService(store, settings, llm_client=llm)
    assert service.add(request_id="r", user_id="u", session_id="s",
                       raw_messages=_messages()).success
    step = service._finalize_session(user_id="u", session_id="s", trigger_ingestion_id=0,
                                     deadline=time.monotonic() + 20)
    assert step["cards"] == 1 and step.get("mode") == "minimal"
    row = store.conn.execute("SELECT payload, full_content FROM experiences").fetchone()
    payload = json.loads(row["payload"])
    assert payload["payload_version"] == PAYLOAD_VERSION
    assert payload["extraction_version"] == VERSION
    assert "release_mouse" not in row["full_content"]
    assert "The Fix" not in row["full_content"]
    outcome = service.search(user_id="u", query="widgets mouse input redraw", top_k=5)
    assert len(outcome.data) == 1
    assert "机制（历史因果解释" in outcome.data[0]["content"]
    assert len(llm.calls) == 1
    assert "card-v5-minimal" in llm.calls[0][0]


def test_admission_failure_gets_one_feedback_repair(tmp_path):
    settings = settings_for(tmp_path)
    service = MemoryService(Store(settings), settings)

    class RepairLLM:
        configured = True

        def __init__(self):
            self.calls = []

        def complete_json(self, *, system, user, **kwargs):
            self.calls.append(user)
            bad = json.loads(json.dumps(CLEAN))
            bad["mechanism"] = "The fix adds a line in lib/matplotlib/figure.py."
            return {"cards": [bad if len(self.calls) == 1 else CLEAN]}, None

    llm = RepairLLM()
    service._llm = llm
    assert service.add(request_id="r", user_id="u", session_id="s",
                       raw_messages=_messages()).success
    step = service._finalize_session(user_id="u", session_id="s", trigger_ingestion_id=0,
                                     deadline=time.monotonic() + 20)
    assert step["cards"] == 1 and step["repairs"] == 1
    assert "未通过程序化准入校验" in llm.calls[1]


def test_empty_verdict_marks_session_extracted(tmp_path):
    settings = settings_for(tmp_path)
    store = Store(settings)
    service = MemoryService(store, settings, llm_client=MinimalLLM(cards=[]))
    assert service.add(request_id="r", user_id="u", session_id="s",
                       raw_messages=_messages()).success
    step = service._finalize_session(user_id="u", session_id="s", trigger_ingestion_id=0,
                                     deadline=time.monotonic() + 20)
    assert step["cards"] == 0 and step["reason"] == "no_experience"
    version = store.conn.execute("SELECT extraction_version FROM sessions").fetchone()[0]
    assert version == VERSION


def test_version_switch_does_not_replace_existing_cards(tmp_path):
    settings = settings_for(tmp_path)
    store = Store(settings)
    service = MemoryService(store, settings, llm_client=MinimalLLM())
    assert service.add(request_id="r", user_id="u", session_id="s",
                       raw_messages=_messages()).success
    iid = store.last_ingestion_of_session("u", "s")
    store.insert_experience(
        user_id="u", session_id="s", experience_id="old", ingestion_id=iid,
        kind="debug", payload_json='{"problem":"legacy"}', full_content="legacy",
        extraction_version="card-v4", created_at="t", terms=[], entities=[])
    step = service._finalize_session(user_id="u", session_id="s",
                                     trigger_ingestion_id=iid,
                                     deadline=time.monotonic() + 20)
    assert step["reason"] == "extraction_version_conflict"
    assert store.conn.execute("SELECT full_content FROM experiences").fetchone()[0] == "legacy"
    assert not service._llm.calls


# --------------------------------------------------------------------------
# storage-pilot integration
# --------------------------------------------------------------------------


def test_no_summary_marks_session_extracted(tmp_path):
    settings = settings_for(tmp_path)
    store = Store(settings)
    service = MemoryService(store, settings, llm_client=MinimalLLM())
    messages = [{"role": "user", "content": ISSUE},
                {"role": "assistant", "content": "ok"}]
    assert service.add(request_id="r", user_id="u", session_id="s",
                       raw_messages=messages).success
    step = service._finalize_session(user_id="u", session_id="s", trigger_ingestion_id=0,
                                     deadline=time.monotonic() + 20)
    assert step["cards"] == 0 and step["reason"] == "no_issue_or_summary"
    assert not service._llm.calls
    version = store.conn.execute("SELECT extraction_version FROM sessions").fetchone()[0]
    assert version == VERSION


# --------------------------------------------------------------------------
# no-summary fallback (reconstruction window)
# --------------------------------------------------------------------------

FIX_DIFF = """diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py
--- a/lib/matplotlib/figure.py
+++ b/lib/matplotlib/figure.py
@@ -930,6 +930,8 @@ class FigureBase:
     def delaxes(self, ax):
+        # release the implicit mouse grab when the axes is destroyed
+        self.canvas.release_mouse(ax)
         self._remove_ax(ax)
"""


def _fallback_chunks():
    return [
        chunk("issue", ISSUE, message_id="m0", role="user"),
        chunk("tail", "Inspection shows the canvas keeps the grab after clf; "
                      "widgets created afterwards never receive events. " * 2,
              message_id="m1", role="assistant"),
        chunk("fixdiff", FIX_DIFF, message_id="m2", role="user", kind="diff"),
        chunk("test_out", TEST_OUTPUT, message_id="m3", role="user"),
    ]


def test_no_summary_fallback_reconstructs_window():
    evidence = build_evidence(_fallback_chunks())
    assert evidence.is_usable() and evidence.is_fallback
    assert [c.chunk_id for c in evidence.diff_chunks] == ["fixdiff"]
    assert [c.chunk_id for c in evidence.tail_analysis_chunks] == ["tail"]
    prompt = evidence.render()
    assert "没有最终总结" in prompt
    assert "历史最终修复补丁" in prompt and "会话尾部分析" in prompt
    assert "diff --git" in prompt
    assert "补丁只是解释因果的证据" in prompt
    # flag off keeps the legacy behaviour: the last substantial assistant
    # message is the mechanism source and no diff window is built
    legacy = build_evidence(_fallback_chunks(), summary_fallback=False)
    assert legacy.is_usable() and not legacy.is_fallback
    assert [c.chunk_id for c in legacy.summary_chunks] == ["tail"]
    # with no substantial assistant text either, the flag decides usability
    bare = [chunk("issue", ISSUE, message_id="m0", role="user"),
            chunk("fixdiff", FIX_DIFF, message_id="m2", role="user", kind="diff")]
    assert build_evidence(bare).is_usable()
    assert not build_evidence(bare, summary_fallback=False).is_usable()


def test_fallback_admission_rejects_patch_quotation():
    chunks = _fallback_chunks()
    raw = json.loads(json.dumps(CLEAN))
    raw["mechanism"] = ("clf() destroys the axes, so the code has to release the "
                        "implicit mouse grab when the axes is destroyed before "
                        "removing it.")
    card, problems = prepare_minimal(raw, chunks, user_id="u", session_id="s")
    assert card is None
    assert any("补丁新增行" in problem for problem in problems), problems


def test_fallback_provenance_flows_to_card_and_render():
    chunks_by_id = {c.chunk_id: c for c in build_evidence(_fallback_chunks()).chunks}

    def make(source):
        cards = build_cards(
            raw={"cards": [dict(CLEAN)]},
            chunks_by_id=chunks_by_id,
            grounding_chunks_by_id=chunks_by_id,
            user_id="u", session_id="s",
            id_for=lambda i: f"e{i}", extraction_version=VERSION,
            created_at="t", max_cards=1, mechanism_source=source,
        )
        return cards[0]

    card = make("summary")
    assert card.mechanism_source == "summary"
    assert card.source_chunk_ids == ("issue", "fixdiff", "tail")
    card_r = make("reconstructed")
    assert card_r.mechanism_source == "reconstructed"
    counter = SimpleNamespace(count=lambda text: len(text), truncate=lambda *_: "truncated")
    assert "机制（由历史修复补丁重构" in render_minimal(card_r, counter, 5000)
    assert "机制（历史因果解释" in render_minimal(card, counter, 5000)


def _fallback_messages():
    return [
        {"role": "user", "content": ISSUE},
        {"role": "assistant", "content": "I will inspect the widget event path. " * 8},
        {"role": "assistant", "content": FIX_DIFF},
        {"role": "user", "content": TEST_OUTPUT},
    ]


def test_no_summary_fallback_extracts_reconstructed_card(tmp_path):
    settings = settings_for(tmp_path)
    store = Store(settings)
    service = MemoryService(store, settings, llm_client=MinimalLLM())
    assert service.add(request_id="r", user_id="u", session_id="s",
                       raw_messages=_fallback_messages()).success
    step = service._finalize_session(user_id="u", session_id="s", trigger_ingestion_id=0,
                                     deadline=time.monotonic() + 20)
    assert step["cards"] == 1 and step.get("mode") == "minimal_fallback"
    row = store.conn.execute("SELECT payload FROM experiences").fetchone()
    card = card_from_payload(json.loads(row[0]))
    assert card.mechanism_source == "reconstructed"


def test_no_summary_flag_off_keeps_skip(tmp_path):
    # no substantial assistant text anywhere: the old hard-skip behaviour
    settings = settings_for(tmp_path, minimal_summary_fallback=False)
    store = Store(settings)
    service = MemoryService(store, settings, llm_client=MinimalLLM())
    messages = [{"role": "user", "content": ISSUE},
                {"role": "user", "content": FIX_DIFF},
                {"role": "user", "content": TEST_OUTPUT}]
    assert service.add(request_id="r", user_id="u", session_id="s",
                       raw_messages=messages).success
    step = service._finalize_session(user_id="u", session_id="s", trigger_ingestion_id=0,
                                     deadline=time.monotonic() + 20)
    assert step["cards"] == 0 and step["reason"] == "no_issue_or_summary"
    assert not service._llm.calls


# --------------------------------------------------------------------------
# deployment contract: persisted => immediately retrievable (api_guide.md)
# --------------------------------------------------------------------------


def test_background_finalize_extracts_without_explicit_call(tmp_path):
    settings = settings_for(tmp_path, finalize_background=True)
    store = Store(settings)
    service = MemoryService(store, settings, llm_client=MinimalLLM())
    assert service.add(request_id="r", user_id="u", session_id="s",
                       raw_messages=_messages()).success
    # no explicit _finalize_session: the Add worker must get there on its own
    deadline = time.monotonic() + 10
    cards = 0
    while time.monotonic() < deadline:
        cards = store.conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
        if cards:
            break
        time.sleep(0.05)
    assert cards == 1
    status = store.conn.execute("SELECT status FROM sessions").fetchone()[0]
    assert status == "extracted"


def test_pending_fallback_returns_added_content_immediately(tmp_path):
    # no LLM configured: extraction cannot run, the session stays pending, and
    # the provisional chunk item is the only honest answer
    settings = settings_for(tmp_path, search_pending_fallback=True)
    store = Store(settings)
    service = MemoryService(store, settings)
    assert service.add(request_id="r", user_id="u", session_id="s",
                       raw_messages=_messages()).success
    outcome = service.search(user_id="u", query="widgets mouse input", top_k=10)
    assert outcome.data, "Add must be immediately retrievable"
    item = outcome.data[0]
    assert item["id"].startswith("pending:")
    assert "mouse" in item["content"].lower()
    assert outcome.diagnostics["pending_fallback"] == 1
    # scope isolation: another tenant sees nothing
    other = service.search(user_id="tenant_y", query="widgets mouse input", top_k=10)
    assert other.data == [] and other.diagnostics["pending_fallback"] == 0


def test_pending_fallback_off_returns_nothing_without_cards(tmp_path):
    settings = settings_for(tmp_path)
    store = Store(settings)
    service = MemoryService(store, settings)
    assert service.add(request_id="r", user_id="u", session_id="s",
                       raw_messages=_messages()).success
    outcome = service.search(user_id="u", query="widgets mouse input", top_k=10)
    assert outcome.data == []
    assert outcome.diagnostics["pending_fallback"] == 0


def test_contract_add_then_immediate_search(tmp_path):
    # the competition contract end to end: success=true, then a Search issued
    # right away must return the content (provisional until the card exists)
    settings = settings_for(tmp_path, finalize_background=True,
                            search_pending_fallback=True)
    store = Store(settings)
    service = MemoryService(store, settings, llm_client=MinimalLLM())
    outcome = service.add(request_id="r", user_id="u", session_id="s",
                          raw_messages=_messages())
    assert outcome.success
    assert any(step.get("name") == "finalize_background" and not step.get("skipped")
               for step in outcome.steps)
    # immediate search: either the card (worker already done) or the
    # provisional chunk item (worker still running) must be returned
    first = service.search(user_id="u", query="widgets mouse input", top_k=10)
    assert first.data and "mouse" in first.data[0]["content"].lower()
    # the background worker eventually upgrades the provisional item to a card
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if store.conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]:
            break
        time.sleep(0.05)
    later = service.search(user_id="u", query="widgets mouse input", top_k=10)
    assert later.data
    assert not later.data[0]["id"].startswith("pending:")
    assert "机制（历史因果解释" in later.data[0]["content"]


def test_search_waits_for_background_card_replacement(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    settings = settings_for(tmp_path, search_pending_fallback=True)
    store = Store(settings)
    service = MemoryService(store, settings, llm_client=MinimalLLM())
    service.add(request_id="r", user_id="u", session_id="s", raw_messages=_messages())
    entered, release, searching = threading.Event(), threading.Event(), threading.Event()
    original = service._store_extracted_cards

    def paused_store(*args, **kwargs):
        entered.set()  # the previous cards have already been deleted
        assert release.wait(5)
        return original(*args, **kwargs)

    def search():
        searching.set()
        return service.search(user_id="u", query="widgets mouse input", top_k=10)

    monkeypatch.setattr(service, "_store_extracted_cards", paused_store)
    with ThreadPoolExecutor(max_workers=2) as pool:
        finalizer = pool.submit(service._finalize_session, user_id="u", session_id="s",
                                trigger_ingestion_id=0, deadline=time.monotonic() + 10)
        try:
            assert entered.wait(5)
            query = pool.submit(search)
            assert searching.wait(5)
            assert not query.done()
            # A different scope is unaffected by the replacement.
            assert service.search(user_id="other", query="mouse", top_k=10).data == []
        finally:
            release.set()
        finalizer.result(timeout=5)
        assert query.result(timeout=5).data


def test_add_after_extraction_snapshot_remains_pending(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    settings = settings_for(tmp_path)
    store = Store(settings)
    service = MemoryService(store, settings, llm_client=MinimalLLM())
    service.add(request_id="old", user_id="u", session_id="s", raw_messages=_messages())
    entered, release, adding = threading.Event(), threading.Event(), threading.Event()

    def paused_extract(**kwargs):
        entered.set()
        assert release.wait(5)
        return [], True, {"reason": "offline controlled result"}

    def add():
        adding.set()
        return service.add(request_id="new", user_id="u", session_id="s",
                           raw_messages=[{"role": "user", "content": "new evidence"}])

    monkeypatch.setattr(service, "_extract_session_cards", paused_extract)
    with ThreadPoolExecutor(max_workers=2) as pool:
        finalizer = pool.submit(service._finalize_session, user_id="u", session_id="s",
                                trigger_ingestion_id=0, deadline=time.monotonic() + 10)
        try:
            assert entered.wait(5)
            incoming = pool.submit(add)
            assert adding.wait(5)
            assert not incoming.done()
        finally:
            release.set()
        finalizer.result(timeout=5)
        assert incoming.result(timeout=5).success
    assert store.conn.execute("SELECT status FROM sessions WHERE user_id='u'").fetchone()[0] == "pending"
