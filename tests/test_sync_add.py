"""Synchronous session acknowledgement, atomic publication and concurrency."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
from fastapi.testclient import TestClient

from codemem.api import create_app
from codemem.common.embeddings import reset_model_limiter_for_tests
from codemem.common.llm import LLMUnavailable
from codemem.service import MemoryService
from codemem.common.store import Store
from test_minimal import MinimalLLM, _messages, settings_for


def setup(tmp_path, llm=None, **kwargs):
    reset_model_limiter_for_tests()
    cfg = settings_for(tmp_path, add_finalize_sync=True, **kwargs)
    store = Store(cfg)
    llm = llm if llm is not None else MinimalLLM()
    return MemoryService(store, cfg, llm_client=llm), store, llm


def add(service, request="r", session="s"):
    return service.add(request_id=request, user_id="u", session_id=session,
                       raw_messages=_messages())


def test_success_means_cards_exist_and_replay_needs_no_extraction(tmp_path, monkeypatch):
    service, store, llm = setup(tmp_path, finalize_background=True,
                              search_pending_fallback=True)
    monkeypatch.setattr(service, "_spawn_background_finalize",
                        lambda *a: pytest.fail("sync Add must not start background work"))
    assert add(service).success
    assert store.session_is_extracted("u", "s")
    assert len(store.session_cards("u", "s")) == 1
    assert add(service).deduplicated
    assert len(llm.calls) == 1
    monkeypatch.setattr(service, "_finalize_closed_sessions",
                        lambda **kw: pytest.fail("Search must not extract"))
    result = service.search(user_id="u", query="widgets mouse input", top_k=100)
    assert result.data and result.diagnostics["pending_fallback"] == 0
    assert result.diagnostics["finalized_sessions"] == 0


def test_failed_extraction_retries_committed_chunks_without_duplication(tmp_path):
    class Failing(MinimalLLM):
        def complete_json(self, **kwargs):
            raise RuntimeError("temporary backend failure")

    service, store, _ = setup(tmp_path, Failing(), search_pending_fallback=True)
    with pytest.raises(LLMUnavailable):
        add(service)
    assert not store.session_is_extracted("u", "s")
    count = store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert service.search(user_id="u", query="widgets mouse input", top_k=100).data == []
    llm = MinimalLLM()
    service._llm = llm
    assert add(service).success
    assert store.session_is_extracted("u", "s")
    assert store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == count
    assert store.conn.execute("SELECT COUNT(*) FROM ingestions").fetchone()[0] == 1
    assert len(llm.calls) == 1


def test_http_returns_retryable_503_until_extraction_completes(tmp_path):
    cfg = settings_for(tmp_path, add_finalize_sync=True)
    with TestClient(create_app(cfg)) as client:
        service = client.app.state.service
        # The unconfigured backend cannot produce a successful Add.
        payload = dict(request_id="r", user_id="u", session_id="s", messages=_messages())
        first = client.post("/add", json=payload)
        assert first.status_code == 503 and first.json()["retryable"] is True
        service._llm = MinimalLLM()
        second = client.post("/add", json=payload)
        assert second.status_code == 200 and second.json()["success"] is True
        assert client.app.state.store.session_cards("u", "s")


def test_same_user_sixteen_sessions_use_four_model_slots_without_dropping(tmp_path):
    class FourAtATime(MinimalLLM):
        barrier = threading.Barrier(4)

        def complete_json(self, **kwargs):
            self.barrier.wait(timeout=10)
            return super().complete_json(**kwargs)

    service, store, llm = setup(tmp_path, FourAtATime(), model_concurrency=4)
    with ThreadPoolExecutor(max_workers=16) as pool:
        jobs = [pool.submit(add, service, f"r{i}", f"s{i}") for i in range(16)]
        assert all(job.result(timeout=20).success for job in jobs)
    assert len(llm.calls) == 16
    assert store.conn.execute("SELECT COUNT(*) FROM sessions WHERE status='extracted'").fetchone()[0] == 16
    assert store.conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0] == 16


def test_same_session_duplicate_waits_but_other_session_and_search_proceed(tmp_path):
    entered, release, duplicate_started = threading.Event(), threading.Event(), threading.Event()

    class Paused(MinimalLLM):
        def complete_json(self, **kwargs):
            if threading.current_thread().name.endswith("_0"):
                entered.set()
                assert release.wait(10)
            return super().complete_json(**kwargs)

    service, store, llm = setup(tmp_path, Paused())

    def duplicate():
        duplicate_started.set()
        return add(service)

    with ThreadPoolExecutor(max_workers=4) as pool:
        first = pool.submit(add, service)
        try:
            assert entered.wait(5)
            second = pool.submit(duplicate)
            assert duplicate_started.wait(5)
            other = pool.submit(add, service, "other-r", "other-s")
            assert other.result(timeout=5).success
            search = pool.submit(service.search, user_id="u", query="widgets mouse input", top_k=100)
            assert search.result(timeout=5).data
            assert not second.done()
        finally:
            release.set()
        assert first.result(timeout=5).success
        assert second.result(timeout=5).deduplicated
    assert len(llm.calls) == 2
    assert len(store.session_cards("u", "s")) == 1


def test_search_sees_old_complete_card_during_atomic_replacement(tmp_path, monkeypatch):
    service, store, _ = setup(tmp_path)
    add(service)
    old = store.session_cards("u", "s")
    entered, release = threading.Event(), threading.Event()
    original = service._store_extracted_cards

    def paused(*args, **kwargs):
        entered.set()  # old cards have been deleted inside the uncommitted tx
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_store_extracted_cards", paused)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writing = pool.submit(add, service, "new-r", "s")
        try:
            assert entered.wait(5)
            searching = pool.submit(service.search, user_id="u", query="widgets mouse input", top_k=10)
            data = searching.result(timeout=5).data
            assert [item["id"] for item in data] == ["experience:" + eid for eid in old]
        finally:
            release.set()
        assert writing.result(timeout=5).success


def test_failed_publication_rolls_back_old_cards_and_can_retry(tmp_path, monkeypatch):
    service, store, _ = setup(tmp_path)
    add(service)
    old = store.session_cards("u", "s")
    original = store._local.conn  # assert publication leaves no open tx below
    with monkeypatch.context() as patch:
        patch.setattr(store, "insert_experience", lambda **kw: (_ for _ in ()).throw(RuntimeError("disk write failed")))
        with pytest.raises(RuntimeError, match="disk write failed"):
            add(service, "new-r", "s")
    assert store.session_cards("u", "s") == old
    assert not original.in_transaction
    assert not store.session_is_extracted("u", "s")
    assert add(service, "new-r", "s").success


def test_valid_zero_card_outcome_completes_add(tmp_path):
    service, store, _ = setup(tmp_path, MinimalLLM(cards=[]))
    assert add(service).success
    assert store.session_is_extracted("u", "s")
    assert store.session_cards("u", "s") == []


def test_ten_returned_after_twenty_candidate_batch(tmp_path):
    from test_transferability import engine

    e, _, llm = engine(tmp_path, search_max_items=10)
    data, diag = e.search(user_id="u", query="Synthetic parser loses empty fields in nested records.", top_k=100)
    assert len(data) == 10
    assert diag.applicability["candidates"] == 20
    assert len(llm.calls) == 2
