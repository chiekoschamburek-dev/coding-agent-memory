"""Case-packet and listwise-applicability tests.

The refactor's contract:

* a packet is **self-contained** -- rendering needs no chunk lookup, and no
  chunk id carries meaning;
* conditions are **facts**, never an ``applicable`` verdict;
* the listwise judge is a **soft** signal with a deterministic fallback;
* Search returns a small, high-precision set, not 100 diluted items.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from codemem.search.applicability import (  # noqa: E402
    ListwiseApplicabilityJudge,
    NullApplicabilityJudge,
    Verdict,
    build_judge_prompt,
    fuse_with_applicability,
    parse_verdicts,
)
from codemem.common.casebook import (  # noqa: E402
    CasePacket,
    Condition,
    VerifyRecipe,
    build_packet,
    derive_conditions,
    extract_files,
    extract_patch,
    format_change_for_agent,
    extract_symbols,
    extract_verify,
    render_packet,
)
from codemem.search.query import build_query_features, denoise_query, extract_anchors  # noqa: E402
from codemem.common.store import StoredChunk  # noqa: E402


def chunk(
    text: str,
    kind: str = "prose",
    cid: str = "c:1",
) -> StoredChunk:
    return StoredChunk(
        chunk_id=cid,
        user_id="u1",
        message_id="m1",
        ord=0,
        char_start=0,
        char_end=len(text),
        kind=kind,
        text=text,
        source_ts=None,
        ts_origin="received",
        session_id="s1",
        role="user",
    )


class Counter:
    mode = "estimate"

    def count(self, text: str) -> int:
        return max(1, len(text) // 4)

    def truncate(self, text: str, max_tokens: int) -> str:
        return text[: max_tokens * 4]


DIFF = """diff --git a/src/auth/token.py b/src/auth/token.py
--- a/src/auth/token.py
+++ b/src/auth/token.py
@@ -40,7 +40,7 @@
-def refresh_token(user_id):
+def refresh_token(user_id, now):
     return jwt.encode({'sub': user_id}, SECRET, algorithm='HS256')
"""
COMMAND = "$ pytest tests/test_token.py -q\n3 passed, 1 failed in 0.42s\n"
TEST_RESULT = "4 passed in 0.51s\n"


# --------------------------------------------------------------------------
# deterministic extraction
# --------------------------------------------------------------------------


def test_diff_extraction_files_symbols_patch_and_verify():
    chunks = [
        chunk(DIFF, "diff", "c:diff"),
        chunk(COMMAND, "command", "c:cmd"),
        chunk(TEST_RESULT, "test_result", "c:test"),
    ]
    assert extract_files(chunks) == ("src/auth/token.py",)
    assert "refresh_token" in extract_symbols(chunks)
    patch = extract_patch(chunks)
    assert patch.startswith("diff --git") and "refresh_token(user_id, now)" in patch
    verify = extract_verify(chunks)
    assert verify.cmd.startswith("pytest tests/test_token.py")
    assert verify.expect == "4 passed in 0.51s"


def test_ls_listing_is_not_a_patch_and_scratch_paths_are_dropped():
    listing = (
        "-rw-r--r--  1 user staff 8196 Dec 17 42 .DS_Store\n"
        "-rw-------  1 user staff 15838 Nov 23 calculate_session_usage.py\n"
        "drwxr-xr-x 3 user staff 96 Nov 23 testbed\n"
    )
    chunks = [chunk(listing, "diff", "c:ls")]
    assert extract_patch(chunks) == ""
    assert extract_files(chunks) == ()
    # Tool-call paths to harness scratch files are not repository evidence.
    tool = chunk('[Read] {"file_path": "swebench_9_15/testbed/test_issue.py"}', "prose", "c:t")
    assert extract_files([tool]) == ()
    repo = chunk('[Read] {"file_path": "lib/mpl_toolkits/axes_grid1/axes_grid.py"}', "prose", "c:r")
    assert extract_files([repo]) == ("lib/mpl_toolkits/axes_grid1/axes_grid.py",)


def test_verify_ignores_prose_result_lines():
    prose = chunk("the helper printed 1 passed but no test was run here", "prose", "c:p")
    assert extract_verify([prose]).expect == ""
    cmd = chunk("$ pytest tests/test_a.py -q", "command", "c:c")
    real = chunk("3 passed, 1 failed in 0.42s\n", "test_result", "c:t")
    recipe = extract_verify([cmd, real])
    assert recipe.cmd.startswith("pytest tests/test_a.py")
    assert recipe.expect.startswith("3 passed")


def test_harness_prefix_is_stripped_to_repo_relative_path():
    tool = chunk('[Read] {"file_path": "swebench_9_15/testbed/lib/a.py"}', "prose", "c:p")
    assert extract_files([tool]) == ("lib/a.py",)


def test_judge_prompt_contains_whole_cases_only():
    packets = [
        CasePacket(
            experience_id=f"e{i}",
            problem="symptom " * 40,
            files=("lib/a.py",),
            implement="fix " * 30,
        )
        for i in range(8)
    ]
    prompt = build_judge_prompt("issue " * 500, packets, max_chars=3000)
    assert len(prompt) <= 3000
    assert prompt.count("<case ") == prompt.count("</case>")
    assert prompt.count("<case ") >= 1
    assert prompt.rstrip().endswith("</case>"), "never end inside a case"


def test_declared_scratch_paths_are_dropped_from_packet():
    card = type("Card", (), {
        "experience_id": "e1", "user_id": "u1", "session_id": "s1",
        "kind": "debug", "problem": "bug", "implement": "fix",
        "source_chunk_ids": ("c:1",),
        "files": ("test_issue.py", "swebench_9_15/testbed/test_issue.py", "lib/a.py"),
        "created_at": "",
    })()
    packet = build_packet(
        card,
        [chunk("diff --git a/lib/a.py b/lib/a.py\n@@ -1 +1 @@\n-old\n+new")],
    )
    assert packet.files == ("lib/a.py",)


def test_tool_call_paths_are_used_when_no_diff_exists():
    chunks = [chunk('[工具调用] Read {"file_path": "./src/a.py"}', "prose")]
    assert extract_files(chunks) == ("src/a.py",)


def test_edit_tool_call_synthesizes_a_change_snippet():
    text = (
        '[工具调用] Edit {"file_path": "./testbed/lib/mpl_toolkits/axes_grid1/axes_grid.py", '
        '"old_string": "def _tick_only(ax):\\n    ax.axis[\\"bottom\\"].toggle()", '
        '"new_string": "def _tick_only(ax):\\n    try:\\n        ax.axis[\\"bottom\\"].toggle()\\n'
        '    except TypeError:\\n        pass"}'
    )
    chunks = [chunk(text, "prose", "c:e")]
    patch = extract_patch(chunks)
    assert "lib/mpl_toolkits/axes_grid1/axes_grid.py" in patch
    assert "-def _tick_only(ax):" in patch
    assert "+    try:" in patch
    assert extract_files(chunks) == ("lib/mpl_toolkits/axes_grid1/axes_grid.py",)


def test_bash_tool_call_supplies_verification_command():
    cmd = chunk(
        '[工具调用] Bash {"command": "cd /testbed && pytest lib/mpl_toolkits/axes_grid1/tests/test_axes_grid1.py -q"}',
        "prose", "c:c",
    )
    result = chunk("3 passed, 1 failed in 0.42s\n", "prose", "c:r")
    recipe = extract_verify([cmd, result])
    assert recipe.cmd.startswith("cd /testbed && pytest")
    assert recipe.expect.startswith("3 passed")


def test_patch_files_lead_packet_files():
    card = type("Card", (), {
        "experience_id": "e1", "user_id": "u1", "session_id": "s1",
        "kind": "debug", "problem": "bug", "implement": "fix",
        "source_chunk_ids": ("c:1",),
        "files": ("test_simple.py", "sklearn/utils/validation.py", "vendor/evil/config.py"),
        "patch": "diff --git a/vendor/evil/hack.py b/vendor/evil/hack.py\n"
                 "--- a/vendor/evil/hack.py\n+++ b/vendor/evil/hack.py\n@@\n-x = 1\n+x = 2\n",
        "created_at": "",
    })()
    diff = (
        "diff --git a/sklearn/utils/validation.py b/sklearn/utils/validation.py\n"
        "--- a/sklearn/utils/validation.py\n"
        "+++ b/sklearn/utils/validation.py\n@@\n-x = 1\n+x = 2\n"
    )
    packet = build_packet(card, [chunk(diff, "diff", "c:1")])
    assert packet.files[0] == "sklearn/utils/validation.py"
    assert "test_simple.py" not in packet.files
    assert "vendor/evil/hack.py" not in packet.files
    assert "vendor/evil/config.py" not in packet.files


def test_extract_verify_prefers_last_success():
    chunks = [
        chunk("$ pytest tests/test_alpha.py -q", "command", "c:c"),
        chunk("1 failed in 0.2s", "test_result", "c:r1"),
        chunk("1 passed in 0.3s", "test_result", "c:r2"),
    ]
    recipe = extract_verify(chunks)
    assert recipe.cmd == "pytest tests/test_alpha.py -q"
    assert recipe.expect == "1 passed in 0.3s"


def test_scratch_test_command_is_not_a_verification_recipe():
    cmd = chunk('[工具调用] Bash {"command": "cd testbed && python test_issue.py"}', "prose", "c:c")
    result = chunk("SUCCESS: Test 2 passed: Default AxesGrid created successfully!", "prose", "c:r")
    recipe = extract_verify([cmd, result])
    assert recipe.is_empty(), "agent scratch test is not a repo verification recipe"


def test_test_centric_conditions_and_verify_fallback():
    text = (
        "FAILED tests/test_validation.py::test_check_array_dtype - AssertionError\n"
        'File "sklearn/utils/validation.py", line 464, in check_array\n'
        "1 failed, 20 passed in 0.42s\n"
    )
    chunks = [chunk(text, "test_result", "c:t")]
    conditions = derive_conditions(chunks, ("sklearn/utils/validation.py",))
    values = {(c.type, c.value) for c in conditions}
    assert ("test_name", "tests/test_validation.py::test_check_array_dtype") in values
    assert ("frame", "sklearn/utils/validation.py:464") in values
    recipe = extract_verify(chunks)
    assert recipe.cmd.startswith("pytest tests/test_validation.py::test_check_array_dtype")


def test_add_experience_candidates_embeds_cited_chunks(store, settings):
    import json as _json

    from codemem.add.experience import ExperienceCard, card_to_payload
    from codemem.search.retrieval import SearchEngine

    card = ExperienceCard(
        experience_id="e1", user_id="u1", session_id="s1",
        problem="p", implement="r", files=("src/a.py",), source_chunk_ids=("c:1",),
    )
    engine = SearchEngine(store, settings)
    engine.store.fetch_current_experiences = lambda user_id, ids: {  # type: ignore[assignment]
        "e1": {"payload": _json.dumps(card_to_payload(card)), "created_at": ""}
    }
    engine.store.fetch_chunks = lambda user_id, ids: {"c:1": chunk("evidence", "code", "c:1")}  # type: ignore[assignment]
    pool = {}
    engine._add_experience_candidates("u1", ["e1"], {"e1": {"experience_field": 1}}, {"e1": 0.5}, pool)
    candidate = pool[card.stable_id]
    assert candidate is not None
    assert "c:1" in candidate.payload["chunks_by_id"], "cited evidence rides along"
    assert candidate.payload["files"] == ["src/a.py"]


def test_patch_summary_has_no_copyable_code():
    packet = CasePacket(
        experience_id="e1", problem="check_array does not warn on DataFrame",
        files=("sklearn/utils/validation.py",),
        symbols=("check_array",),
        implement="read dtype from array.values",
        patch="--- a/sklearn/utils/validation.py\n"
              "+++ b/sklearn/utils/validation.py\n"
              "@@ -464,3 +464,4 @@ def check_array(...)\n"
              "-        dtype_orig = None\n"
              "+        if hasattr(array, 'values'):\n"
              "+            dtype_orig = array.values.dtype\n",
    )
    content = render_packet(packet, Counter(), max_tokens=400, patch_render="summary")
    assert "历史改法（仅作参考，不提供可复制代码" in content
    assert "check_array" in content
    assert "修改前：" not in content and "修改后：" not in content
    assert "dtype_orig = array.values.dtype" not in content


def test_packet_conditions_are_facts_not_verdicts():
    chunks = [chunk("TokenExpiredError raised; matplotlib version 3.8\n", "prose")]
    card = type(
        "Card",
        (),
        {
            "experience_id": "e1",
            "user_id": "u1",
            "session_id": "s1",
            "kind": "debug",
            "problem": "TokenExpiredError when version 3.8 refreshes",
            "implement": "use monotonic clock",
            "source_chunk_ids": ("c:1",),
            "conditions": (Condition("module", "src/auth", "issue"),),
            "files": ("src/auth/token.py",),
            "created_at": "2026-01-01T00:00:00Z",
        },
    )()
    packet = build_packet(card, chunks)
    types = {c.type for c in packet.conditions}
    assert "module" in types and "error" in types
    payload = packet.to_payload()
    assert "applicable" not in payload
    assert "conditions" in payload


def test_packet_renders_self_contained_under_budget():
    packet = CasePacket(
        experience_id="e1",
        kind="debug",
        problem="TokenExpiredError after refresh",
        conditions=(Condition("version", "3.8", "issue"),),
        files=("src/auth/token.py",),
        symbols=("refresh_token",),
        implement="compare against monotonic clock",
        patch=DIFF,
        verify=VerifyRecipe(cmd="pytest tests/test_token.py -q", expect="4 passed"),
        pitfalls=("wall clock made tests flaky",),
        provenance={"commit": "abc123", "chunks": ["c:1"], "chunk_count": 1},
    )
    content = render_packet(packet, Counter(), max_tokens=400)
    assert "[历史修复案例]" in content
    assert "文件：src/auth/token.py" in content
    assert "历史改法" in content
    assert "pytest tests/test_token.py" in content
    assert "来源：abc123 · chunks c:1（1）" in content
    assert packet.provenance["chunks"] == ["c:1"], \
        "chunk ids stay in the stored provenance for audit"
    assert Counter().count(content) <= 400
    # No chunk-excerpt block: the packet is the evidence, not a pointer.
    assert "原文 [" not in content

    answer = render_packet(
        packet, Counter(), max_tokens=400, include_chunk_provenance=False)
    assert "来源：chunks" not in answer
    assert "c:1" not in answer
    assert "来源：abc123" in answer
    assert packet.provenance["chunks"] == ["c:1"]


def test_gate_downgrades_historical_file_hint():
    packet = CasePacket(
        experience_id="e1", problem="AxesGrid fails with custom axes",
        files=("lib/mpl_toolkits/axes_grid1/axes_grid.py",),
        implement="guard axis access",
        patch="--- a/axes_grid.py\n+++ b/axes_grid.py\n@@\n-old\n+new\n",
        verify=VerifyRecipe(cmd="pytest -q", expect="1 passed"),
        conditions=(Condition("test_name", "tests/test_axes_grid1.py::test_axes_class_tuple"),),
    )
    gated = render_packet(packet, Counter(), max_tokens=400, allow_patch=False)
    assert "文件：" not in gated
    assert "历史路径" not in gated
    assert "历史改法（参考" not in gated
    assert "历史改法思路（仅供参考）：" in gated
    assert "问题（历史线索，需在当前仓库确认）：" in gated
    assert "历史测试（需在当前仓库确认）：" in gated
    assert "确认方式：在当前仓库检索" in gated
    normal = render_packet(packet, Counter(), max_tokens=400, allow_patch=True)
    assert "文件：" in normal
    assert "历史改法" in normal


def test_packet_render_drops_optional_sections_before_evidence():
    packet = CasePacket(
        experience_id="e1",
        problem="x" * 200,
        files=("a/b.py",),
        implement="intent",
        patch="+ line 0\n+ line 1",
        verify=VerifyRecipe(cmd="pytest -q", expect="1 passed"),
        pitfalls=("p1", "p2", "p3"),
        provenance={"commit": "deadbeef"},
    )
    content = render_packet(packet, Counter(), max_tokens=100)
    assert "历史改法" in content  # change survives
    assert "修改后：" in content
    assert Counter().count(content) <= 100


def test_change_snippet_keeps_hunk_line_and_anchor():
    """A tool-using agent reads big files truncated; the packet must say where."""
    text = format_change_for_agent(
        "diff --git a/sklearn/utils/validation.py b/sklearn/utils/validation.py\n"
        "--- a/sklearn/utils/validation.py\n"
        "+++ b/sklearn/utils/validation.py\n"
        "@@ -464,7 +464,11 @@ def check_array(array, accept_sparse=False,\n"
        "     dtype_orig = getattr(array, \"dtype\", None)\n"
        "-        dtype_orig = None\n"
        "+        if hasattr(array, 'values'):\n"
        "+            dtype_orig = array.values.dtype\n"
    )
    assert "第 464 行" in text
    assert "def check_array" in text
    assert "定位锚点" in text
    assert "dtype_orig = None" in text


def test_change_snippet_without_hunk_numbers_gets_grep_anchor():
    """Edit-tool trajectories have no line numbers; the removed line is the anchor."""
    text = format_change_for_agent(
        "--- a/sklearn/utils/validation.py\n"
        "+++ b/sklearn/utils/validation.py\n"
        "@@\n"
        "-    dtype_orig = getattr(array, \"dtype\", None)\n"
        "-    if not hasattr(dtype_orig, 'kind'):\n"
        "-        dtype_orig = None\n"
        "+    dtype_orig = getattr(array, \"dtype\", None)\n"
        "+    if not hasattr(dtype_orig, 'kind'):\n"
        "+        dtype_orig = array.values.dtype\n"
    )
    assert "定位锚点（在目标文件中检索这一行）：dtype_orig = getattr(array, \"dtype\", None)" in text
    assert "位置：@@" not in text
    assert "修改前：" in text and "修改后：" in text


# --------------------------------------------------------------------------
# query denoising and anchors
# --------------------------------------------------------------------------


def test_denoise_removes_pr_boilerplate_urls_and_checkboxes():
    raw = (
        "<!-- template comment -->\n"
        "- [ ] new code is tested (https://docs.example.com/testing.html)\n"
        "Support RGBAxes in AxesGrid (#26031)\n"
    )
    clean = denoise_query(raw)
    assert "<!--" not in clean and "https://" not in clean
    assert "- [ ]" not in clean
    assert "RGBAxes" in clean


def test_anchors_filter_url_and_checkbox_noise():
    anchors = extract_anchors(
        "See https://docs.example.com/a.html and [N/A]; fix RGBAxes.py in AxesGrid (#26031)"
    )
    values = [v.lower() for bucket in anchors.values() for v in bucket]
    assert "n/a" not in values
    assert not any("docs.example.com" in v for v in values)
    assert any("rgba" in v for v in values)
    assert "#26031" in values


def test_query_features_expose_anchors_without_changing_tokens():
    features = build_query_features("TokenExpiredError in src/auth/token.py", denoise=True)
    assert "tokenexpirederror" in features.tokens
    assert features.anchors.get("path")
    assert any("src/auth/token.py" in value for value in features.anchors["path"])


# --------------------------------------------------------------------------
# listwise applicability
# --------------------------------------------------------------------------


class FakeLLM:
    configured = True

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def complete_json(self, *, system, user, max_tokens=1024, timeout_s=None):
        self.calls += 1
        return self.payload, None


class Deadline:
    def __init__(self, left: float) -> None:
        self.time_left = left

    @property
    def expired(self) -> bool:
        return self.time_left <= 0


def packet(eid: str, problem: str = "same bug") -> CasePacket:
    return CasePacket(experience_id=eid, problem=problem, files=("a.py",))


def test_parse_verdicts_ignores_unknown_ids_and_bad_labels():
    candidates = [packet("e1"), packet("e2")]
    raw = {
        "items": [
            {"id": "experience:e1", "app": "yes", "confidence": 0.9, "why": "same"},
            {"id": "experience:nope", "app": "yes", "confidence": 0.9},
            {"id": "experience:e2", "app": "maybe", "confidence": 0.5},
        ]
    }
    verdicts = parse_verdicts(raw, candidates)
    assert list(verdicts) == ["experience:e1"]
    assert verdicts["experience:e1"].app == "yes"


def test_listwise_judge_caches_and_reports_tri_state():
    settings = type("S", (), {
        "applicability_enabled": True,
        "applicability_max_candidates": 20,
        "applicability_max_tokens": 200,
        "applicability_timeout_s": 2.0,
    })()
    llm = FakeLLM({"items": [
        {"id": "experience:e1", "app": "yes", "confidence": 0.8, "why": "same file"},
        {"id": "experience:e2", "app": "unknown", "confidence": 0.2, "why": "version unknown"},
    ]})
    judge = ListwiseApplicabilityJudge(llm, settings)
    packets = [packet("e1"), packet("e2")]
    first = judge.rerank(query="fix a.py", packets=packets)
    assert first["experience:e1"].app == "yes"
    assert first["experience:e2"].app == "unknown"
    second = judge.rerank(query="fix a.py", packets=packets)
    assert second == first
    assert llm.calls == 1  # cached


def test_judge_skips_when_deadline_spent():
    settings = type("S", (), {
        "applicability_enabled": True,
        "applicability_max_candidates": 20,
        "applicability_max_tokens": 200,
        "applicability_timeout_s": 2.0,
    })()
    judge = ListwiseApplicabilityJudge(FakeLLM({"items": []}), settings)
    assert judge.rerank(query="q", packets=[packet("e1"), packet("e2")], deadline=Deadline(0.1)) is None


def test_null_judge_keeps_retrieval_order():
    class C:
        def __init__(self, item_id, score):
            self.item_id = item_id
            self.fused_score = score
            self.payload = {}

    ranked = [C("a", 2.0), C("b", 1.0)]
    judge = NullApplicabilityJudge()
    assert judge.rerank(query="q", packets=[]) is None
    assert fuse_with_applicability(ranked, {}, 0.01) == ranked


def test_fusion_is_soft_and_sorted():
    class C:
        def __init__(self, item_id, score):
            self.item_id = item_id
            self.fused_score = score
            self.payload = {}

    ranked = [C("experience:e1", 2.0), C("experience:e2", 1.999)]
    verdicts = {
        "experience:e1": Verdict("no", 0.9),
        "experience:e2": Verdict("yes", 0.9),
    }
    fused = fuse_with_applicability(ranked, verdicts, 0.01)
    assert [c.item_id for c in fused] == ["experience:e2", "experience:e1"]
    assert len(fused) == 2  # never dropped
    assert fused[0].payload["applicability"]["app"] == "yes"


# --------------------------------------------------------------------------
# Search shape
# --------------------------------------------------------------------------


def test_listwise_reorders_without_dropping(store, settings):
    from codemem.add.experience import ExperienceCard
    from codemem.search.query import build_query_features
    from codemem.search.retrieval import Candidate, SearchDiagnostics, SearchEngine

    class Judge:
        available = True

        def rerank(self, *, query, packets, deadline=None):
            by_id = {p.stable_id: p for p in packets}
            first, second = list(by_id)
            return {
                first: Verdict("no", 0.9, "different root cause"),
                second: Verdict("yes", 0.9, "same file"),
            }

    card_a = ExperienceCard(experience_id="e1", user_id="u1",
                            session_id="s1", problem="alpha bug")
    card_b = ExperienceCard(experience_id="e2", user_id="u1",
                            session_id="s2", problem="beta bug")
    a = Candidate(item_id=card_a.stable_id, kind="experience", text="alpha",
                  fused_score=2.0, payload={"card": card_a, "chunks_by_id": {}})
    b = Candidate(item_id=card_b.stable_id, kind="experience", text="beta",
                  fused_score=1.999, payload={"card": card_b, "chunks_by_id": {}})
    engine = SearchEngine(store, settings, judge=Judge())
    diag = SearchDiagnostics()
    out = engine._apply_applicability(
        [a, b], build_query_features("fix beta"), diag, None
    )
    assert [c.item_id for c in out] == [card_b.stable_id, card_a.stable_id]
    assert len(out) == 2  # soft signal, never dropped
    assert diag.applicability["applied"] is True
    assert out[0].payload["applicability"]["app"] == "yes"


def test_card_identity_is_the_ranking_unit_not_session_lineage(store, settings):
    """Chunks are gone from the pool, so session smoothing has no object: two
    cards from one session and one from another rank purely by their own
    channel scores."""
    from codemem.add.experience import ExperienceCard
    from codemem.search.query import build_query_features
    from codemem.search.retrieval import Candidate, SearchEngine

    def card_cand(eid, session, score):
        card = ExperienceCard(
            experience_id=eid, user_id="u1", session_id=session,
            problem="alpha bug", implement="a repair",
        )
        return Candidate(
            item_id=card.stable_id, kind="experience", text="alpha evidence",
            fused_score=score,
            payload={"card": card, "chunks_by_id": {}, "created_at": None},
        )

    def channel_cand(eid, session, score):
        cand = card_cand(eid, session, score)
        cand.channels["experience_field"] = 1
        return cand

    a = channel_cand("e1", "s1", 2.0)
    b = channel_cand("e2", "s1", 1.0)
    c = channel_cand("e3", "s2", 1.5)
    engine = SearchEngine(store, settings)
    ranked = engine._rank(
        {x.item_id: x for x in (a, b, c)}, build_query_features("alpha")
    )
    scores = {x.item_id: x.fused_score for x in ranked}
    # The shared packet/channel boosts shift every card equally; the fused
    # gaps (and therefore the order) are the ranking signal.
    assert scores[a.item_id] - scores[b.item_id] == pytest.approx(1.0)
    assert scores[a.item_id] - scores[c.item_id] == pytest.approx(0.5)
    assert [x.item_id for x in ranked] == [a.item_id, c.item_id, b.item_id]


def test_judge_sees_l2_candidates_before_final_truncation(store, settings):
    """L3 must see ~20 candidates; only L4 may truncate to 8."""
    from codemem.add.experience import ExperienceCard
    from codemem.search.retrieval import Candidate, SearchEngine

    class RecordingJudge:
        available = True
        seen = 0

        def rerank(self, *, query, packets, deadline=None):
            self.seen = max(self.seen, len(packets))
            return {}

    judge = RecordingJudge()
    engine = SearchEngine(store, settings, judge=judge)
    candidates = []
    for i in range(12):
        card = ExperienceCard(
            experience_id=f"e{i}", user_id="u1",
            session_id="s1", problem=f"bug number {i} in module {i}",
        )
        candidates.append(Candidate(
            item_id=card.stable_id, kind="experience", text=f"bug {i}",
            fused_score=10.0 - i * 0.1,
            payload={"card": card, "chunks_by_id": {}},
        ))
    engine._recall_cards = lambda *a, **k: {c.item_id: c for c in candidates}  # type: ignore[assignment]
    engine._rank = lambda pool, features: sorted(  # type: ignore[assignment]
        pool.values(), key=lambda c: -c.fused_score)
    data, diag = engine.search(user_id="u1", query="bug", top_k=100)
    assert judge.seen >= 10, "judge only saw the truncated final list"
    assert len(data) <= settings.search_max_items
    assert diag.selected <= settings.search_max_items


def test_applicability_hint_is_reserved_in_budget(store, settings):
    """A warning must never push its own item over the hard budget."""
    from codemem.add.experience import ExperienceCard
    from codemem.search.retrieval import Candidate, SearchEngine

    settings.token_budget_total = 200
    settings.token_budget_target = 200
    card = ExperienceCard(
        experience_id="e1", user_id="u1", session_id="s1",
        problem="TokenExpiredError after refresh",
        files=("src/auth/token.py",),
        implement="use monotonic clock",
        patch="+ x = 1",
        source_chunk_ids=("c:1",),
    )
    candidate = Candidate(
        item_id=card.stable_id, kind="experience", text="card",
        fused_score=2.0,
        payload={
            "card": card,
            "chunks_by_id": {},
            "applicability": {"app": "no", "confidence": 0.9,
                              "reason": "different root cause"},
        },
    )
    engine = SearchEngine(store, settings)
    engine._recall_cards = lambda *a, **k: {candidate.item_id: candidate}  # type: ignore[assignment]
    engine._rank = lambda pool, features: [candidate]  # type: ignore[assignment]
    data, diag = engine.search(user_id="u1", query="refresh", top_k=100)
    assert len(data) == 1, "a reserved warning must not drop the item"
    assert "适用性提示" in data[0]["content"]
    assert diag.tokens_used <= 200


def test_patch_admission_rejects_class_only_overlap():
    from codemem.add.experience import ExperienceCard
    from codemem.search.query import build_query_features
    from codemem.search.retrieval import Candidate, _patch_allowed

    card = ExperienceCard(
        experience_id="e1", user_id="u1", session_id="s1",
        problem="TypeError in _tick_only for GeoAxes",
        files=("lib/mpl_toolkits/axes_grid1/axes_grid.py",),
        patch="--- a/axes_grid.py\n+++ b/axes_grid.py\n@@\n-old\n+new\n",
    )
    candidate = Candidate(item_id=card.stable_id, kind="experience", text="card",
                          payload={"card": card})
    features = build_query_features("Support standard Axes in RGBAxes, like AxesGrid (#26020)")
    assert _patch_allowed(candidate, features) is False


def test_patch_admission_allows_function_match():
    from codemem.add.experience import ExperienceCard
    from codemem.search.query import build_query_features
    from codemem.search.retrieval import Candidate, _patch_allowed

    card = ExperienceCard(
        experience_id="e1", user_id="u1", session_id="s1",
        problem="check_array does not warn on pandas DataFrame",
        files=("sklearn/utils/validation.py",),
        patch="--- a/validation.py\n+++ b/validation.py\n@@\n-old\n+new\n",
    )
    candidate = Candidate(item_id=card.stable_id, kind="experience", text="card",
                          payload={"card": card})
    features = build_query_features("check_array warn_on_dtype fails for DataFrame")
    assert _patch_allowed(candidate, features) is True


def test_card_items_survive_a_large_card_pool(store, settings):
    """An 8-item response must not silently drop the whole experience layer."""
    from codemem.add.experience import ExperienceCard
    from codemem.search.retrieval import Candidate, SearchEngine

    candidate_map = {}
    ordered = []
    # Ten low-signal cards outrank two high-signal ones on raw score; the
    # response must still keep the experience layer visible and bounded.
    for i in range(10):
        card = ExperienceCard(
            experience_id=f"n{i}", user_id="u1",
            session_id=f"noise{i}", problem=f"noise card {i}",
            implement="a repair",
        )
        cand = Candidate(item_id=card.stable_id, kind="experience",
                         text=f"noise card {i}", fused_score=10.0 - i * 0.1,
                         payload={"card": card, "chunks_by_id": {}, "created_at": None})
        candidate_map[cand.item_id] = cand
        ordered.append(cand)
    for i in range(2):
        card = ExperienceCard(
            experience_id=f"e{i}", user_id="u1",
            session_id="s1", problem=f"card {i}", implement="a repair",
            files=(f"src/m{i}.py",),
        )
        cand = Candidate(item_id=card.stable_id, kind="experience",
                         text=f"card {i}", fused_score=0.1 + i * 0.01,
                         payload={"card": card, "chunks_by_id": {}, "created_at": None})
        candidate_map[cand.item_id] = cand
        ordered.append(cand)
    engine = SearchEngine(store, settings)
    engine._recall_cards = lambda *a, **k: dict(candidate_map)  # type: ignore[assignment]
    engine._rank = lambda pool, features: sorted(  # type: ignore[assignment]
        pool.values(), key=lambda c: -c.fused_score)
    data, diag = engine.search(user_id="u1", query="anything", top_k=100)
    ids = [item["id"] for item in data]
    assert sum(1 for value in ids if value.startswith("experience:")) >= 2
    assert len(data) <= settings.search_max_items


def test_search_returns_a_small_precise_set(store, settings, service):
    import json

    from codemem.add.experience import ExperienceCard, card_to_payload
    from codemem.common.store import now_iso

    state = store.begin_ingestion(
        user_id="u1", request_id="r0", session_id="s1", digest="d",
        mode="lex_only", lease_seconds=60,
    )
    for i in range(12):
        card = ExperienceCard(
            experience_id=f"e:{i}", user_id="u1", session_id="s1",
            problem=f"module_{i} ConfigParseError failure {i}",
            implement="a repair",
        )
        store.insert_experience(
            user_id="u1", session_id="s1", experience_id=card.experience_id,
            ingestion_id=state.ingestion_id, kind="debug",
            payload_json=json.dumps(card_to_payload(card), ensure_ascii=False),
            full_content=f"module_{i} ConfigParseError failure {i}",
            extraction_version="t", created_at=now_iso(),
            terms=["problemconfigparseerror", f"problemmodule{i}"],
            entities=[],
        )
    store.mark_session_extracted(
        user_id="u1", session_id="s1", extraction_version="t", card_count=12,
    )
    outcome = service.search(user_id="u1", query="ConfigParseError", top_k=100)
    assert 0 < len(outcome.data) <= settings.search_max_items

