"""Configuration.

Every tunable from plan.md lives here and is read from the environment, so no
call site hard-codes a budget, a quota, or a model name. Defaults are the
values plan.md documents; the capacity assumptions it flags as unverified
(``C_PER_TASK``, ``N_SEARCH``) are *not* used at runtime -- they only informed
the choice of the sync budgets below.

This module never logs or echoes credential values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader.

    Deliberately dependency-free and non-overriding: a real environment
    variable always wins, so container config is not silently shadowed by a
    stray file. Values are never logged.
    """
    if not path.is_file():
        return
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key.startswith("export "):
            key = key.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(REPO_ROOT / ".env")


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class BudgetTier:
    """One row of the graded token budget from plan.md section 2.4."""

    rank_start: int  # 1-based inclusive
    rank_end: int  # 1-based inclusive
    per_item_tokens: int
    tier_total_tokens: int

    def covers(self, rank: int) -> bool:
        return self.rank_start <= rank <= self.rank_end


@dataclass
class Settings:
    # --- storage -----------------------------------------------------------
    db_path: Path = field(default_factory=lambda: Path(_env_str("CODEMEM_DB_PATH", str(REPO_ROOT / "data" / "codemem.db"))))

    # --- transport / auth --------------------------------------------------
    api_key: str | None = field(default_factory=lambda: os.environ.get("CODEMEM_API_KEY") or None)
    host: str = field(default_factory=lambda: _env_str("CODEMEM_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("CODEMEM_PORT", 8080))

    # --- model adapter (never called on the sync critical path) ------------
    llm_base_url: str | None = field(default_factory=lambda: os.environ.get("CODEMEM_LLM_BASE_URL") or None)
    llm_api_key: str | None = field(default_factory=lambda: os.environ.get("CODEMEM_LLM_API_KEY") or None)
    # plan.md section 2.2: the rules mandate this exact model for Add/Search.
    llm_model: str = field(default_factory=lambda: _env_str("CODEMEM_LLM_MODEL", "gpt-4o-mini"))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("CODEMEM_LLM_TIMEOUT_S", 30.0))
    llm_max_retries: int = field(default_factory=lambda: _env_int("CODEMEM_LLM_MAX_RETRIES", 2))

    # Embeddings are a separate, optional capability: plan.md section 2.2
    # flags whether the gpt-4o-mini rule also binds embeddings as *unverified*,
    # and section 9 stage 1 requires vector failure to leave lexical retrieval
    # intact. So this is independently configurable and independently failable.
    embed_base_url: str | None = field(default_factory=lambda: os.environ.get("CODEMEM_EMBED_BASE_URL") or None)
    embed_api_key: str | None = field(default_factory=lambda: os.environ.get("CODEMEM_EMBED_API_KEY") or None)
    embed_model: str = field(default_factory=lambda: _env_str("CODEMEM_EMBED_MODEL", ""))
    embed_dim: int = field(default_factory=lambda: _env_int("CODEMEM_EMBED_DIM", 1536))
    embed_timeout_s: float = field(default_factory=lambda: _env_float("CODEMEM_EMBED_TIMEOUT_S", 20.0))
    embed_max_retries: int = field(default_factory=lambda: _env_int("CODEMEM_EMBED_MAX_RETRIES", 1))
    # Best-effort embedding window on Add: embeddings are the optional
    # enrichment from plan.md section 5, so they get their own slice of the
    # request budget and are abandoned rather than allowed to overrun.
    embed_budget_s: float = field(default_factory=lambda: _env_float("CODEMEM_EMBED_BUDGET_S", 8.0))
    embed_batch_size: int = field(default_factory=lambda: _env_int("CODEMEM_EMBED_BATCH_SIZE", 10))
    embed_cache_path: Path = field(
        default_factory=lambda: Path(_env_str("CODEMEM_EMBED_CACHE_PATH", str(REPO_ROOT / "datasets" / "swecontextbench" / "cache" / "embeddings.jsonl")))
    )
    # Concurrency bound on model calls (FIXES.md P1.2). The service runs in a
    # threadpool, so unbounded model work would occupy every worker and starve
    # health checks. Capping concurrent model calls keeps the bulk of the pool
    # free; a request that cannot acquire a slot within the timeout degrades to
    # lex-only instead of queueing.
    model_concurrency: int = field(default_factory=lambda: _env_int("CODEMEM_MODEL_CONCURRENCY", 4))
    model_acquire_timeout_s: float = field(
        default_factory=lambda: _env_float("CODEMEM_MODEL_ACQUIRE_TIMEOUT_S", 0.5)
    )

    # --- request budgets (plan.md section 2.3) -----------------------------
    # Wall-clock guard for the *sync* path that must succeed. The 30-minute
    # server limit is a fault-tolerance ceiling, not a working budget.
    add_sync_budget_s: float = field(default_factory=lambda: _env_float("CODEMEM_ADD_SYNC_BUDGET_S", 40.0))
    search_budget_p95_s: float = field(default_factory=lambda: _env_float("CODEMEM_SEARCH_BUDGET_P95_S", 5.0))
    search_hard_budget_s: float = field(default_factory=lambda: _env_float("CODEMEM_SEARCH_HARD_BUDGET_S", 20.0))
    # Best-effort model work on Add. It is deliberately independent of the
    # sync-path budget: a session-sized extraction may take a minute, while the
    # commit itself must never wait for a model.
    add_enrich_budget_s: float = field(default_factory=lambda: _env_float("CODEMEM_ADD_ENRICH_BUDGET_S", 180.0))

    # --- adaptive degradation (plan.md section 5) --------------------------
    guard_window: int = field(default_factory=lambda: _env_int("CODEMEM_GUARD_WINDOW", 50))
    guard_slow_ratio_enter: float = field(default_factory=lambda: _env_float("CODEMEM_GUARD_SLOW_RATIO_ENTER", 0.2))
    guard_slow_ratio_exit: float = field(default_factory=lambda: _env_float("CODEMEM_GUARD_SLOW_RATIO_EXIT", 0.1))
    guard_token_ratio_skip: float = field(default_factory=lambda: _env_float("CODEMEM_GUARD_TOKEN_RATIO_SKIP", 0.8))
    guard_token_ratio_healthy: float = field(default_factory=lambda: _env_float("CODEMEM_GUARD_TOKEN_RATIO_HEALTHY", 0.6))
    guard_recover_streak: int = field(default_factory=lambda: _env_int("CODEMEM_GUARD_RECOVER_STREAK", 10))

    # --- input limits (A1: publish them, never silently truncate) ----------
    max_messages_per_request: int = field(default_factory=lambda: _env_int("CODEMEM_MAX_MESSAGES", 2000))
    max_message_chars: int = field(default_factory=lambda: _env_int("CODEMEM_MAX_MESSAGE_CHARS", 1_000_000))
    max_request_chars: int = field(default_factory=lambda: _env_int("CODEMEM_MAX_REQUEST_CHARS", 4_000_000))

    # --- chunking ----------------------------------------------------------
    chunk_target_chars: int = field(default_factory=lambda: _env_int("CODEMEM_CHUNK_TARGET_CHARS", 1100))
    chunk_max_chars: int = field(default_factory=lambda: _env_int("CODEMEM_CHUNK_MAX_CHARS", 2000))
    chunk_min_chars: int = field(default_factory=lambda: _env_int("CODEMEM_CHUNK_MIN_CHARS", 60))

    # --- retrieval (plan.md section 6.2) -----------------------------------
    top_k_max: int = field(default_factory=lambda: _env_int("CODEMEM_TOP_K_MAX", 100))
    # The answer model is a tool-using agent: it has ~5 reads and 6 test runs.
    # Five self-contained items is the response budget; top_k remains the
    # contract upper bound, but precision beats volume.
    search_max_items: int = field(default_factory=lambda: _env_int("CODEMEM_SEARCH_MAX_ITEMS", 5))
    # Candidates handed to the cross-encoder before the final cut: recall 20,
    # rerank all 20, then the floor + final response cut.
    rerank_candidates: int = field(default_factory=lambda: _env_int("CODEMEM_RERANK_CANDIDATES", 20))
    rerank_fusion_alpha: float = field(default_factory=lambda: _env_float("CODEMEM_RERANK_FUSION_ALPHA", 0.3))
    rerank_timeout_s: float = field(default_factory=lambda: _env_float("CODEMEM_RERANK_TIMEOUT_S", 15.0))
    # External rerank API. Primary names are RERANK_* (as configured in .env);
    # CODEMEM_RERANK_* is accepted as an alias for container deployments.
    rerank_base_url: str | None = field(default_factory=lambda: (
        os.environ.get("RERANK_BASE_URL") or os.environ.get("CODEMEM_RERANK_BASE_URL") or None))
    rerank_api_key: str | None = field(default_factory=lambda: (
        os.environ.get("RERANK_API_KEY") or os.environ.get("CODEMEM_RERANK_API_KEY") or None))
    rerank_model: str = field(default_factory=lambda: _env_str("RERANK_MODEL", "qwen3-rerank"))
    # Cards are the only reuse unit; these slots guarantee a
    # card-heavy ranking cannot be crowded out of the final response.
    experience_min_items: int = field(default_factory=lambda: _env_int("CODEMEM_EXPERIENCE_MIN_ITEMS", 2))
    card_recall_limit: int = field(default_factory=lambda: _env_int("CODEMEM_CARD_RECALL_LIMIT", 200))
    card_evidence_items: int = field(default_factory=lambda: _env_int("CODEMEM_CARD_EVIDENCE_ITEMS", 0))
    # How the historical change is exposed to the answer agent.
    # before_after = 修改前/修改后 blocks (default): the diff is the one thing the
    #                current repo cannot reconstruct, and the reader is a strong
    #                coding agent that can check it against the current code
    # summary      = structure/semantics only, no copyable code (ablation; two
    #                measured runs never reached an edit, so it is no longer default)
    # off          = never expose the change
    experience_patch_render: str = field(default_factory=lambda: _env_str("CODEMEM_PATCH_RENDER", "before_after"))
    pool_per_channel: int = field(default_factory=lambda: _env_int("CODEMEM_POOL_PER_CHANNEL", 400))
    pool_total: int = field(default_factory=lambda: _env_int("CODEMEM_POOL_TOTAL", 1000))
    # Rank fusion spans ~0.002-0.016 per channel.  Heuristic boosts must stay in
    # that band or they silently become the ranking function: a +0.14 "diff"
    # bonus used to outrank a 400-place lexical gap.  This scale multiplies the
    # normalised factors below into RRF-sized units.
    rank_boost_scale: float = field(default_factory=lambda: _env_float("CODEMEM_RANK_BOOST_SCALE", 0.004))
    fusion_weight_case_anchor: float = field(default_factory=lambda: _env_float("CODEMEM_W_CASE_ANCHOR", 1.2))
    fusion_weight_options: float = field(default_factory=lambda: _env_float("CODEMEM_W_OPTIONS", 0.35))
    fusion_weight_experience_field: float = field(default_factory=lambda: _env_float("CODEMEM_W_EXPERIENCE_FIELD", 1.0))
    fusion_weight_experience_entity: float = field(default_factory=lambda: _env_float("CODEMEM_W_EXPERIENCE_ENTITY", 0.8))
    # The card's own embedding: the card is the retrieval unit, so it carries a
    # vector of its index face.  Chunks are never embedded.
    fusion_weight_card_vector: float = field(default_factory=lambda: _env_float("CODEMEM_W_CARD_VECTOR", 1.0))
    # Field-aware card retrieval: a query error should weigh the card's
    # problem field more than its test field, a path should weigh locate, etc.
    card_field_weight_problem: float = field(default_factory=lambda: _env_float("CODEMEM_CARD_W_PROBLEM", 1.0))
    card_field_weight_locate: float = field(default_factory=lambda: _env_float("CODEMEM_CARD_W_LOCATE", 1.0))
    card_field_weight_implement: float = field(default_factory=lambda: _env_float("CODEMEM_CARD_W_IMPLEMENT", 0.5))
    card_field_weight_test: float = field(default_factory=lambda: _env_float("CODEMEM_CARD_W_TEST", 0.8))
    card_anchor_field_boost: float = field(default_factory=lambda: _env_float("CODEMEM_CARD_ANCHOR_BOOST", 1.6))
    card_bm25_k1: float = field(default_factory=lambda: _env_float("CODEMEM_CARD_BM25_K1", 1.2))
    card_bm25_b: float = field(default_factory=lambda: _env_float("CODEMEM_CARD_BM25_B", 0.75))
    rrf_k: int = field(default_factory=lambda: _env_int("CODEMEM_RRF_K", 60))

    # --- query understanding ----------------------------------------------
    # Task prompts carry PR-template boilerplate (HTML comments, checkboxes,
    # documentation URLs).  Those tokens and entities pollute BM25 and the
    # exact-entity channel, so they are removed before feature extraction.
    query_denoise: bool = field(default_factory=lambda: _env_bool("CODEMEM_QUERY_DENOISE", True))

    # --- listwise applicability rerank ------------------------------------
    # Applicability is a relation between a historical case and the current
    # issue; it is computed per query, never stored on the card.  The judge is
    # a soft signal: it may reorder and demote, but it never drops a candidate.
    applicability_enabled: bool = field(default_factory=lambda: _env_bool("CODEMEM_APPLICABILITY_ENABLED", True))
    applicability_weight: float = field(default_factory=lambda: _env_float("CODEMEM_APPLICABILITY_WEIGHT", 0.010))
    applicability_max_candidates: int = field(default_factory=lambda: _env_int("CODEMEM_APPLICABILITY_MAX_CANDIDATES", 20))
    applicability_max_tokens: int = field(default_factory=lambda: _env_int("CODEMEM_APPLICABILITY_MAX_TOKENS", 900))
    applicability_timeout_s: float = field(default_factory=lambda: _env_float("CODEMEM_APPLICABILITY_TIMEOUT_S", 3.0))

    # --- experience cards (per-session extraction) --------------------------
    # Cards are extracted once per session, after the session is fully added
    # (a session_id change closes it; Search finalises the tail). Extraction
    # is best-effort enrichment: it never runs on the sync critical path and
    # is gated at runtime on the LLM being configured.
    experience_enabled: bool = field(default_factory=lambda: _env_bool("CODEMEM_EXPERIENCE_ENABLED", True))
    experience_budget_s: float = field(default_factory=lambda: _env_float("CODEMEM_EXPERIENCE_BUDGET_S", 120.0))
    # Hard cap on cards per session: fewer, higher-quality cards beat a pile
    # of near-duplicates. One card is a perfectly good outcome.
    experience_max_cards_per_session: int = field(default_factory=lambda: _env_int("CODEMEM_EXPERIENCE_MAX_CARDS", 3))
    # 600 chunks covers the largest measured history session (~160k chars);
    # larger sessions are extracted in batches, bounded by the per-finalisation
    # model-call budget.
    experience_session_max_chunks: int = field(default_factory=lambda: _env_int("CODEMEM_EXPERIENCE_SESSION_MAX_CHUNKS", 600))
    # v5 attempt cards are opt-in; use a separate DB/library for A/B evaluation.
    experience_extraction_version: str = field(
        default_factory=lambda: _env_str("CODEMEM_EXPERIENCE_EXTRACTION_VERSION", "card-v4")
    )
    # Per-quote ceiling for evidence embedded inside a card, so one long quote
    # cannot consume the whole item budget.
    experience_quote_max_chars: int = field(default_factory=lambda: _env_int("CODEMEM_EXPERIENCE_QUOTE_MAX_CHARS", 600))
    # Ceiling on the LLM completion for one extraction call.
    experience_max_output_tokens: int = field(default_factory=lambda: _env_int("CODEMEM_EXPERIENCE_MAX_OUTPUT_TOKENS", 2600))
    # Hard cap on LLM calls per session finalisation (extraction batches +
    # retries).
    extract_calls_per_finalization: int = field(default_factory=lambda: _env_int("CODEMEM_EXTRACT_CALLS_PER_FINALIZATION", 5))
    # v5 only: independent extraction samples per batch whose admitted cards
    # are merged (deduplicated) into the session's card set. Sampling variance
    # is the dominant single-sample yield loss; the union recovers recall
    # without touching the admission gate.
    experience_v5_sample_calls: int = field(default_factory=lambda: _env_int("CODEMEM_V5_SAMPLE_CALLS", 2))

    # Self-contained packets are budgeted per item; these are hard ceilings that
    # the renderer enforces (see casebook.py).
    case_patch_max_lines: int = field(default_factory=lambda: _env_int("CODEMEM_CASE_PATCH_MAX_LINES", 60))
    case_packet_max_tokens: int = field(default_factory=lambda: _env_int("CODEMEM_CASE_PACKET_MAX_TOKENS", 900))

    # --- token budget (plan.md section 2.4) --------------------------------
    token_budget_total: int = field(default_factory=lambda: _env_int("CODEMEM_TOKEN_BUDGET_TOTAL", 96_000))
    token_budget_target: int = field(default_factory=lambda: _env_int("CODEMEM_TOKEN_BUDGET_TARGET", 64_000))
    # Minimum evidence body (excluding the provenance header) for an item to be
    # worth returning. Kept very low by design: this only exists to reject an
    # entry whose budget was consumed entirely by its header, not to demand
    # long evidence. A higher default would silently drop short-but-valid
    # chunks (FIXES.md P2.5 follow-up).
    min_output_body_tokens: int = field(
        default_factory=lambda: _env_int("CODEMEM_MIN_OUTPUT_BODY_TOKENS", 1)
    )
    # Output admission. The response is not padded to ``search_max_items``: a
    # candidate has to earn its slot. The floor is relative to the best score in
    # the same response (RRF magnitudes are not comparable across queries) and
    # ``min_output_items`` keeps a weak query from returning nothing at all.
    min_score_ratio: float = field(
        default_factory=lambda: _env_float("CODEMEM_MIN_SCORE_RATIO", 0.15)
    )
    min_score_abs: float = field(
        default_factory=lambda: _env_float("CODEMEM_MIN_SCORE_ABS", 0.0)
    )
    min_output_items: int = field(
        default_factory=lambda: _env_int("CODEMEM_MIN_OUTPUT_ITEMS", 2)
    )
    tokenizer_encoding: str = field(default_factory=lambda: _env_str("CODEMEM_TOKENIZER", "cl100k_base"))
    # Conservative fallback when tiktoken is unavailable: plan.md section 2.4
    # forbids claiming exact control, so we over-count rather than under-count.
    fallback_chars_per_token: float = field(default_factory=lambda: _env_float("CODEMEM_FALLBACK_CPT", 3.2))

    tiers: tuple[BudgetTier, ...] = field(
        default_factory=lambda: (
            BudgetTier(1, 10, 1200, 12_000),
            BudgetTier(11, 30, 700, 14_000),
            BudgetTier(31, 60, 350, 10_500),
            BudgetTier(61, 100, 200, 8_000),
        )
    )

    # Experience items carry a three-section, self-contained structure
    # (memory.md §7.5.3): 历史经验 + 实施证据引文 + 验证证据引文, all of which
    # count toward the item's content tokens (§7.5.4). The chunk tiers above are
    # too tight for that, so experience items get their own, wider per-item caps.
    # The summed hard cap (token_budget_total) is unchanged and still enforced by
    # the output stage; only the per-item allowance differs.
    experience_tiers: tuple[BudgetTier, ...] = field(
        default_factory=lambda: (
            BudgetTier(1, 10, 1600, 16_000),
            BudgetTier(11, 30, 1000, 20_000),
            BudgetTier(31, 60, 700, 21_000),
            BudgetTier(61, 100, 450, 18_000),
        )
    )

    @property
    def embeddings_enabled(self) -> bool:
        return bool(self.embed_base_url and self.embed_api_key and self.embed_model)

    def per_item_token_cap(self, rank: int) -> int:
        for tier in self.tiers:
            if tier.covers(rank):
                return tier.per_item_tokens
        return self.tiers[-1].per_item_tokens

    def experience_per_item_token_cap(self, rank: int) -> int:
        for tier in self.experience_tiers:
            if tier.covers(rank):
                return tier.per_item_tokens
        return self.experience_tiers[-1].per_item_tokens

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_for_tests() -> None:
    global _settings
    _settings = None