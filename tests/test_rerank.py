"""Rerank placement and fallback tests."""

from __future__ import annotations

from pathlib import Path

from codemem.common.config import Settings
from codemem.search.query import build_query_features
from codemem.search.rerank import HttpReranker
from codemem.search.retrieval import Candidate, SearchDiagnostics, SearchEngine
from codemem.common.store import Store, StoredChunk


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        db_path=tmp_path / "rerank.db",
        api_key=None,
        llm_base_url=None,
        llm_api_key=None,
        embed_base_url=None,
        embed_api_key=None,
        embed_model="",
        search_max_items=5,
        rerank_candidates=8,
        rerank_fusion_alpha=1.0,
    )
    base.update(overrides)
    return Settings(**base)


def _chunk(chunk_id: str, text: str) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        user_id="u1",
        message_id=f"m:{chunk_id}",
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


class FakeReranker:
    available = True

    def __init__(self, scores_fn) -> None:
        self.scores_fn = scores_fn
        self.documents: list[str] = []

    def rerank(self, query, documents, deadline=None):
        self.documents = list(documents)
        return self.scores_fn(len(self.documents))


def test_rerank_happens_before_final_cut(tmp_path):
    settings = _settings(tmp_path)
    engine = SearchEngine(Store(settings), settings, reranker=FakeReranker(
        lambda n: [0.0] * (n - 1) + [1.0]
    ))
    features = build_query_features("alpha")
    candidates = [
        Candidate("c:a", "chunk", "alpha", "alpha"),
        Candidate("c:b", "chunk", "alpha beta", "alpha beta"),
        Candidate("c:c", "chunk", "alpha gamma", "alpha gamma"),
    ]
    diag = SearchDiagnostics()
    ranked = engine._apply_rerank(candidates, features, diag, None)
    assert ranked[0].item_id == "c:c"
    assert ranked[0].fused_score == 1.0
    final = engine._finalize(ranked, diag, 1)
    assert [candidate.item_id for candidate in final] == ["c:c"]
    assert diag.rerank["applied"] is True


def test_rerank_sends_only_configured_candidates(tmp_path):
    settings = _settings(tmp_path, rerank_candidates=3)
    reranker = FakeReranker(lambda n: [1.0] * n)
    engine = SearchEngine(Store(settings), settings, reranker=reranker)
    features = build_query_features("alpha")
    candidates = [
        Candidate(f"c:{i}", "chunk", f"alpha {i}", f"alpha {i}")
        for i in range(8)
    ]
    engine._apply_rerank(candidates, features, SearchDiagnostics(), None)
    assert len(reranker.documents) == 3


def test_rerank_failure_keeps_original_order(tmp_path):
    settings = _settings(tmp_path)
    engine = SearchEngine(Store(settings), settings, reranker=FakeReranker(lambda n: None))
    features = build_query_features("alpha")
    candidates = [
        Candidate("c:a", "chunk", "alpha", "alpha"),
        Candidate("c:b", "chunk", "alpha beta", "alpha beta"),
    ]
    ranked = engine._apply_rerank(
        candidates, features, SearchDiagnostics(), None
    )
    assert [candidate.item_id for candidate in ranked] == ["c:a", "c:b"]


def test_http_reranker_rejects_non_list_results(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"results": null}'

    import codemem.search.rerank as rerank_module

    monkeypatch.setattr(
        rerank_module.urllib.request, "urlopen", lambda *args, **kwargs: Response()
    )
    assert HttpReranker("http://local").rerank("q", ["a", "b"]) is None


def test_apply_rerank_swallows_provider_exception(tmp_path):
    class RaisingReranker:
        available = True

        def rerank(self, query, documents, deadline=None):
            raise TypeError("'NoneType' object is not iterable")

    settings = _settings(tmp_path)
    engine = SearchEngine(Store(settings), settings, reranker=RaisingReranker())
    features = build_query_features("alpha")
    candidates = [
        Candidate("c:a", "chunk", "alpha", "alpha"),
        Candidate("c:b", "chunk", "alpha beta", "alpha beta"),
    ]
    diag = SearchDiagnostics()
    ranked = engine._apply_rerank(candidates, features, diag, None)
    assert [candidate.item_id for candidate in ranked] == ["c:a", "c:b"]
    assert diag.rerank == {"applied": False, "reason": "error"}
    assert "rerank_failed" in diag.degraded
