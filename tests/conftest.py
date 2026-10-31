import os
from pathlib import Path

import pytest

os.environ.setdefault("CODEMEM_API_KEY", "")


@pytest.fixture(autouse=True)
def _isolate_model_env():
    """Keep tests hermetic: never inherit real model credentials from ``.env``.

    ``codemem.common.config`` loads ``.env`` into ``os.environ`` at import time, and
    ``Settings`` field defaults read those vars. Without this, any test that
    builds ``Settings`` without explicitly nulling the LLM/embedding creds would
    let the best-effort enrichment path (experience extraction, embeddings) make
    a real network call. Tests that want a provider inject a fake or pass
    ``embed_*``/``llm_*`` kwargs explicitly, which override the (now empty) env.
    """
    keys = [
        "CODEMEM_LLM_BASE_URL",
        "CODEMEM_LLM_API_KEY",
        "CODEMEM_LLM_MODEL",
        "CODEMEM_EXPERIENCE_EXTRACTION_VERSION",
        "CODEMEM_MINIMAL_SUMMARY_FALLBACK",
        "CODEMEM_FINALIZE_BACKGROUND",
        "CODEMEM_ADD_FINALIZE_SYNC",
        "CODEMEM_SEARCH_PENDING_FALLBACK",
        "CODEMEM_TRANSFERABILITY_ENABLED",
        "CODEMEM_PENDING_FALLBACK_MAX_SESSIONS",
        "CODEMEM_PENDING_FALLBACK_MAX_CHARS",
        "CODEMEM_EMBED_BASE_URL",
        "CODEMEM_EMBED_API_KEY",
        "CODEMEM_EMBED_MODEL",
        "CODEMEM_RERANK_BASE_URL",
        "CODEMEM_RERANK_API_KEY",
        "RERANK_BASE_URL",
        "RERANK_API_KEY",
        "RERANK_MODEL",
        "EVAL_LLM_BASE_URL",
        "EVAL_LLM_API_KEY",
        "EVAL_LLM_MODEL",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture()
def settings(tmp_path: Path):
    from codemem.common.config import Settings, reset_settings_for_tests

    reset_settings_for_tests()
    cfg = Settings(
        db_path=tmp_path / "test.db",
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
    )
    yield cfg
    reset_settings_for_tests()


@pytest.fixture()
def store(settings):
    from codemem.common.store import Store

    return Store(settings)


@pytest.fixture()
def service(store, settings):
    from codemem.common.guard import reset_guard_for_tests
    from codemem.service import MemoryService
    from codemem.common.tokens import reset_counter_for_tests

    reset_guard_for_tests()
    reset_counter_for_tests()
    return MemoryService(store, settings)


@pytest.fixture()
def client(settings):
    from fastapi.testclient import TestClient

    from codemem.api import create_app
    from codemem.common.guard import reset_guard_for_tests
    from codemem.common.tokens import reset_counter_for_tests

    reset_guard_for_tests()
    reset_counter_for_tests()
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


class _FakeCardLLM:
    """Deterministic stand-in so the client fixture can produce cards.

    The card's problem/implement echo the first cited chunk, which keeps the
    safety tests' "returned memory is traceable to the input" property.
    """

    configured = True

    def complete_json(self, *, system, user, max_tokens=1024, timeout_s=None):
        import re

        marker = re.compile(r"<<<CHUNK (\S+) \| session (\S+) \| role (\S+)>>>\n")
        matches = list(marker.finditer(user))
        if not matches:
            return {"meta": None}, None
        first = matches[0]
        end = matches[1].start() if len(matches) > 1 else len(user)
        text = user[first.end():end].strip("\n")
        excerpt = " ".join(text.split())[:300]
        return {"cards": [{
            "kind": "debug",
            "problem": excerpt or "historical problem statement module error",
            "locate": "src/deep/nested/module.py",
            "implement": "guard the token refresh path",
            "source_chunk_ids": [first.group(1)],
        }]}, None


@pytest.fixture()
def card_client(settings):
    """API client whose extraction LLM is a deterministic fake, so Search has
    cards to rank (chunks alone are not retrieval items)."""
    from fastapi.testclient import TestClient

    from codemem.api import create_app
    from codemem.common.guard import reset_guard_for_tests
    from codemem.service import MemoryService
    from codemem.common.tokens import reset_counter_for_tests

    reset_guard_for_tests()
    reset_counter_for_tests()
    app = create_app(settings)
    with TestClient(app) as test_client:
        # The lifespan handler builds the real service on startup; swap it for
        # one with the fake LLM once it exists.
        app.state.service = MemoryService(
            app.state.service.store, settings, llm_client=_FakeCardLLM()
        )
        yield test_client
