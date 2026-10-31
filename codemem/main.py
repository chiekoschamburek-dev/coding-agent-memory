"""ASGI entry point.

    uvicorn codemem.main:app --host 0.0.0.0 --port 8080

Also runnable directly (``python -m codemem.main``) for smoke testing.
"""

from __future__ import annotations

import logging
import os

from .api import create_app
from .common.config import get_settings

_settings = get_settings()

logging.basicConfig(
    level=os.environ.get("CODEMEM_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = create_app(_settings)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "codemem.main:app",
        host=_settings.host,
        port=_settings.port,
        log_level=os.environ.get("CODEMEM_LOG_LEVEL", "info").lower(),
        # Add/Search may legitimately run for a while under a degraded provider;
        # keep-alive must not kill a slow request before its own budget does.
        timeout_keep_alive=65,
    )


if __name__ == "__main__":
    main()