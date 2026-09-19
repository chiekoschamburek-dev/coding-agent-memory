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
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture()
def client(settings: Settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def auth_settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", api_key="sekret-key")


@pytest.fixture()
def auth_client(auth_settings: Settings):
    app = create_app(auth_settings)
    with TestClient(app) as c:
        yield c
