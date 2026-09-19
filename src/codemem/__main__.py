"""Entry point: ``python -m codemem`` starts the Add/Search service."""

from __future__ import annotations

import uvicorn

from .core.config import Settings


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        "codemem.api.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        # Add is synchronous and may legitimately take minutes on large
        # payloads (up to 30 min per the contract), so the server must not cut
        # the connection early.
        timeout_keep_alive=120,
    )


if __name__ == "__main__":
    main()
