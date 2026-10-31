"""Code-track engineering-experience memory system.

Session-scoped card library: chunks are the persisted minimal facts, cards are
extracted per session after it closes, and Search returns cards. Public surface
is intentionally small: the Add / Search contract from api_guide.md plus a
health probe.
"""

from .common.store import SCHEMA_VERSION

__version__ = "0.3.0"

__all__ = ["SCHEMA_VERSION", "__version__"]
