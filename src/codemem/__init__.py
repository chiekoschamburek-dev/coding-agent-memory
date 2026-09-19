"""codemem — code-memory Add/Search service.

Design invariants (see docs/DESIGN.md):
1. All content generation happens during Add. Search only scores, filters,
   orders and formats existing memory. Search never generates new text.
2. ``user_id`` is the only hard isolation boundary on every read/write path.
3. Add never fails because of enrichment failure: raw text is persisted first,
   so a degraded response is still ``success: true``.
"""

__version__ = "0.1.0"
