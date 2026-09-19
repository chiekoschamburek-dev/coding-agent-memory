"""Listwise (LLM) reranking.

Two properties matter most:

* robustness — a missing endpoint, a malformed response, or a wrong-length
  answer must leave the existing ranking untouched rather than corrupting it;
* the generation boundary — only numbers may flow back from the model, never text.

All tests use a stub, so the suite stays fast, hermetic, and independent of any
API key or network.
"""

from __future__ import annotations

import pytest

from codemem.add.pipeline import AddPipeline
from codemem.core.config import Settings
from codemem.index.store import Store
from codemem.listwise import ListwiseReranker, normalise, _parse_scores
from codemem.search.service import SearchPipeline


class _StubCompletions:
    def __init__(self, reply: str | Exception):
        self.reply = reply
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply

        class _Msg:
            content = self.reply

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _StubClient:
    def __init__(self, reply):
        self.chat = type("_Chat", (), {"completions": _StubCompletions(reply)})()


def _reranker(reply, **kwargs) -> ListwiseReranker:
    r = ListwiseReranker(model="stub-model", base_url="http://x", api_key="k", **kwargs)
    r._client = _StubClient(reply)
    return r


# ------------------------------------------------------------- parsing ------


def test_parses_plain_json_array():
    assert _parse_scores("[9,0,0,0,1]", 5) == [9.0, 0.0, 0.0, 0.0, 1.0]


def test_tolerates_code_fences_and_preamble():
    assert _parse_scores("```json\n[9,0,1]\n```", 3) == [9.0, 0.0, 1.0]
    assert _parse_scores("Here you go: [9,0,1]", 3) == [9.0, 0.0, 1.0]


def test_rejects_wrong_length():
    """A partial answer must be discarded, not padded: mixing judged and
    invented scores would corrupt the ranking invisibly."""
    assert _parse_scores("[9,0,0]", 5) is None
    assert _parse_scores("[9,0,0,0,0,0]", 5) is None


def test_rejects_non_numeric_entries():
    assert _parse_scores('[9,"x",0,0,1]', 5) is None
    assert _parse_scores("[9,null,0,0,1]", 5) is None
    assert _parse_scores("[true,0,0,0,1]", 5) is None


def test_rejects_unparseable():
    assert _parse_scores("not json at all", 5) is None
    assert _parse_scores("", 5) is None


def test_clamps_out_of_range_values():
    assert _parse_scores("[12,-3,5,5,5]", 5) == [10.0, 0.0, 5.0, 5.0, 5.0]


def test_normalise_maps_0_10_to_0_1():
    assert normalise([10.0, 5.0, 0.0]) == [1.0, 0.5, 0.0]
    assert normalise([None, 10.0]) == [None, 1.0]


# ---------------------------------------------------------- degradation -----


def test_unavailable_without_credentials(settings):
    r = ListwiseReranker(model="m", base_url=None, api_key=None)
    assert not r.available
    assert r.score("q", ["doc"]) is None


def test_request_failure_returns_none():
    r = _reranker(RuntimeError("network down"))
    assert r.score("q", ["a", "b"]) is None


def test_malformed_reply_returns_none():
    r = _reranker("I think the first one is best.")
    assert r.score("q", ["a", "b"]) is None


def test_unjudged_tail_is_not_penalised():
    """Only the head is judged; candidates beyond the cap keep a neutral score
    rather than being treated as irrelevant."""
    r = _reranker("[10,0,0]", max_candidates=3)
    scores = r.score("q", ["a", "b", "c", "d", "e"])
    assert scores is not None
    assert len(scores) == 5
    assert scores[:3] == [10.0, 0.0, 0.0]
    assert scores[3:] == [None, None]


def test_search_works_when_listwise_unavailable(settings):
    store = Store(settings)
    try:
        class M:
            role = "user"
            content = "Fixed IndexError in src/parser/tokenizer.py by guarding the buffer."
            timestamp = None

        AddPipeline(settings, store).handle(
            request_id="a", user_id="u1", session_id="s1", messages=[M()]
        )
        broken = ListwiseReranker(model="m", base_url=None, api_key=None)
        items = SearchPipeline(settings, store, listwise=broken).handle(
            user_id="u1", query="IndexError tokenizer buffer", options=None, top_k=10
        )
        assert items, "the fused ranking must still answer"
    finally:
        store.close()


# --------------------------------------------------------- integration ------


def _corpus(settings: Settings) -> Store:
    store = Store(settings)
    add = AddPipeline(settings, store)

    class M:
        def __init__(self, content, ts):
            self.role = "user"
            self.content = content
            self.timestamp = ts

    rows = [
        ("relevant", "Fixed the tokenizer crash: IndexError from Tokenizer.read_token "
                     "because the buffer was drained on BOM input. Guarded the buffer."),
        ("linter", "Reformatted src/parser/tokenizer.py to satisfy the linter."),
        ("types", "Added type annotations to Tokenizer.read_token for the public API."),
        ("docs", "Updated the docstring of src/parser/tokenizer.py."),
    ]
    for i, (key, text) in enumerate(rows):
        add.handle(
            request_id=f"r{i}",
            user_id="u1",
            session_id=key,
            messages=[M(text, 1704067200000 + i * 1000)],
        )
    return store


def test_listwise_promotes_the_diagnosis_over_same_file_noise(settings):
    """The decisive case: three distractors touch the same file and symbol, so
    only joint reading of the whole list separates them from the diagnosis."""
    settings.listwise_enabled = True
    settings.listwise_max_candidates = 10
    store = _corpus(settings)
    try:
        stub = _reranker("[10,0,0,0]")  # judge says only #0 is relevant
        items = SearchPipeline(settings, store, listwise=stub).handle(
            user_id="u1",
            query="Why did Tokenizer.read_token raise IndexError on BOM input?",
            options=None,
            top_k=10,
        )
        assert items
        assert "IndexError" in items[0].content
        # The judged-irrelevant distractors are demoted or gated out entirely.
        assert all("linter" not in i.content for i in items), (
            "the linter-only memory should not survive a 0 relevance judgement"
        )
    finally:
        store.close()


def test_only_scores_reach_the_pipeline_not_model_text(settings):
    """The model's reply must be consumed as numbers only. Even if it returns
    prose, none of it may appear in returned content."""
    settings.listwise_enabled = True
    store = _corpus(settings)
    try:
        prose = (
            "The answer is: the buffer was drained on BOM input. "
            "[10,0,0,0]"
        )
        stub = _reranker(prose)
        items = SearchPipeline(settings, store, listwise=stub).handle(
            user_id="u1", query="IndexError on BOM input", options=None, top_k=10
        )
        for item in items:
            assert "The answer is" not in item.content, (
                "model prose leaked into returned content"
            )
            assert "[10,0,0,0]" not in item.content
    finally:
        store.close()


def test_prompt_sends_query_and_memory_text(settings):
    """The judge must see the real question and the real memory; otherwise it is
    scoring something other than the retrieval task."""
    settings.listwise_enabled = True
    store = _corpus(settings)
    try:
        stub = _reranker("[10,0,0,0]")
        SearchPipeline(settings, store, listwise=stub).handle(
            user_id="u1", query="Why did BOM input break the tokenizer?", options=None,
            top_k=10,
        )
        call = stub._client.chat.completions.calls[0]
        user_msg = call["messages"][1]["content"]
        assert "Why did BOM input break the tokenizer?" in user_msg
        assert "IndexError" in user_msg
        # And it must be a scoring request, not a generation request.
        assert "JSON array" in user_msg
        assert call["temperature"] == 0
    finally:
        store.close()
