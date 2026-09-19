"""Shared fixtures."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from codemem.api.app import create_app
from codemem.core.config import Settings


@pytest.fixture(autouse=True)
def _quiet_logs() -> None:
    logging.disable(logging.WARNING)


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Settings for the hermetic test suite.

    The optional channels are disabled here even though ``rerank_enabled``
    defaults to True in production: loading a real model per test would make the
    suite slow and dependent on the local cache. The default values themselves
    are asserted separately in ``test_dense.py`` and ``test_rerank.py``, and the
    rerank path is exercised with a stub model there.
    """
    return Settings(
        data_dir=tmp_path / "data",
        dense_enabled=False,
        rerank_enabled=False,
    )


@pytest.fixture()
def client(settings: Settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def auth_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        api_key="sekret-key",
        dense_enabled=False,
        rerank_enabled=False,
    )


@pytest.fixture()
def auth_client(auth_settings: Settings):
    app = create_app(auth_settings)
    with TestClient(app) as c:
        yield c
