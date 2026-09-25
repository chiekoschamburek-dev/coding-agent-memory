"""Runtime configuration.

Every tunable is settable through an environment variable so the Docker image
stays host-agnostic: the same image runs locally on CPU, on the GPU box, or
behind any public ingress.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


def _env_str(name: str, default: str | None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    # ---- storage -------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    db_filename: str = "codemem.sqlite3"

    # ---- transport -----------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8080
    # When unset, authentication is disabled (allowed for public smoke only).
    api_key: str | None = None
    log_level: str = "INFO"
    # Do not log raw memory text; see docs/DESIGN.md data-handling section.
    log_request_bodies: bool = False

    # ---- contract limits ----------------------------------------------
    # Platform sends top_k=100. We accept more but always clamp the response.
    max_top_k: int = 1000
    max_messages_per_add: int = 5000
    max_content_chars: int = 4_000_000
    add_deadline_seconds: float = 1500.0  # 25 min internal guard (< 30 min)

    # ---- chunking ------------------------------------------------------
    target_chunk_tokens: int = 320
    max_chunk_tokens: int = 1400
    # Code fences are never cut mid-line; this is the hard cap that forces a
    # split at line boundaries for pathological blobs.
    hard_chunk_chars: int = 24_000
    chunk_overlap_tokens: int = 32

    # ---- evidence assembly --------------------------------------------
    evidence_full_count: int = 8  # items rendered in full form
    # Cap for a full-form item. Raised from 380 after measuring evidence
    # sufficiency (`eval/run_evidence.py`, n=30): the decisive line survived
    # into the returned window 0.700 of the time at 380 and 0.800 at 600 and
    # above, and 800 is the peak for `decidable` (0.433 → 0.567 at a 4k-token
    # prefix) with no question losing evidence (paired: +3 decisive / −0,
    # +2 decidable / −0) for +9.8 % payload tokens. 1200 is no better and
    # raises ambiguity. The ranking metrics are bit-identical across this
    # range — only the text each item carries changes.
    evidence_item_tokens: int = 800  # cap for a full-form item
    evidence_ptr_tokens: int = 110  # cap for a pointer-form item
    evidence_excerpt_chars: int = 900
    # Cap on items returned from one session. A session yields many chunks, and
    # without a cap they monopolise the ranked list: measured on the proxy
    # benchmark, ~100 returned chunks collapsed to ~23 distinct sessions, so
    # other relevant prior work never got a slot. Coverage depends on session
    # diversity, because a task is answered from a session, not from one chunk.
    # Raised from 3 to 5 after the evidence-sufficiency audit: the decisive
    # chunk often sits deep inside a read-heavy trajectory and the previous
    # cap of 3 frequently truncated before reaching it.
    max_evidence_per_session: int = 5
    # Cap on distinct sessions in one payload. 0 means unlimited, which was the
    # earlier default. Tightening it to 2 is the companion move to the cap
    # increase above: with fewer sessions competing for the token prefix,
    # distractor operative chunks are kept out of the window while the
    # answer session is given more slots to expose its decisive evidence.
    evidence_max_sessions: int = 2
    # Promote a session's operative chunk into its first slot, but only for the
    # top this-many sessions (-1 = every session, 0 = never). Promotion is a
    # double-edged lever: it recovers an edit that its own read-heavy siblings
    # outscored, but it also surfaces *other* sessions' edits, which is what
    # makes a distractor look modified too. With evidence_max_sessions now
    # capped at 2, promoting the top 2 sessions recovers decisive evidence
    # in both candidate sessions without opening the ambiguity floodgate that
    # appeared when every session was promoted (0.567 ambiguity at -1).
    evidence_operative_promotion: int = 2
    # Weight given to "operative" lines (what a session DID) relative to query-term
    # matches when choosing a verbatim window from a long memory. An engineering
    # trajectory records its actions as tool calls and diffs, and those lines are
    # JSON or patch syntax, so they overlap the issue wording poorly and are
    # systematically skipped by term-density selection alone. Measured: the
    # decisive line survived into the returned window only 30% of the time while
    # the session itself was retrieved 100% of the time.
    evidence_operative_weight: float = 1.0
    # Total budget for the whole `data` payload (platform input window minus
    # answer/safety reservation; kept well under 117_760 on purpose).
    evidence_budget_tokens: int = 60_000
    # Noise gate: drop candidates below this normalised score. Prevents
    # filling the token prefix with same-repo distractors. Calibrated with the
    # local benchmark (see eval/), not by intuition.
    min_evidence_score: float = 0.15
    # Always return at least this many if any candidate clears the gate.
    min_evidence_count: int = 1

    # ---- retrieval -----------------------------------------------------
    rrf_k: int = 60
    recall_per_channel: int = 120
    candidate_pool: int = 300

    # ---- listwise reranking (LLM) ----------------------------------------
    # Final ranking stage: judges candidates *comparatively*, which a
    # cross-encoder cannot do because it scores each pair independently. Costs
    # one request per search, so it runs only when explicitly enabled; when the
    # endpoint is missing or the response is malformed, the fused ranking stands.
    # OFF by default, on measurement. On the proxy benchmark every setting
    # scored below the configuration without it (MRR 0.780 -> 0.766, recall@10
    # 0.719 -> 0.675) for ~5x the search latency. Attenuating the weight moves
    # the numbers monotonically back toward the baseline, which is the signature
    # of a stage adding noise rather than signal -- if it carried signal there
    # would be a weight at which it beat the baseline, and there is none.
    #
    # Kept implemented and switchable because the proxy measures the wrong thing
    # here: its ground truth is file overlap, while a judge scores usefulness.
    # A case inspected by hand showed the judge calling boilerplate useless
    # (correctly) while our recall supplied no relevant memory at all, so the
    # proxy penalises the judge for disagreeing with it. "Unproven here", not
    # "proven useless".
    listwise_enabled: bool = False
    listwise_weight: float = 0.5
    listwise_max_candidates: int = 40
    # 1800 measured better than 700 (recall@10 0.692 vs 0.675): our retrieval
    # unit is a whole chunk, and truncating it removes the deciding text.
    listwise_excerpt_chars: int = 1800

    # ---- llm (enrichment / query understanding; Add+Search share it) ----
    llm_enabled: bool = False
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 60.0
    enrich_ratio: float = 0.25  # fraction of chunks eligible for enrichment

    # ---- dense retrieval ------------------------------------------------
    # On by default. The cost/benefit was measured on both devices and it flips
    # with hardware, so the earlier CPU-only "not worth it" conclusion does not
    # hold on a GPU host:
    #
    #   GPU (RTX 5060): dense+rerank beats rerank-only on nDCG@10 (0.6586 vs
    #     0.6474) and recall@10 (0.7133 vs 0.6914); Add 155 s for 300 requests.
    #   CPU: the same configuration added ~no metric gain for ~5x the Add cost.
    #
    # Because device defaults to "auto", a GPU host gets the better ranking and
    # a CPU-only host still fits comfortably inside the 30-minute Add budget.
    # See eval/README.md: the proxy's file-overlap ground truth cannot credit a
    # session that helps semantically without sharing files, so dense's real
    # advantage is understated here.
    dense_enabled: bool = True
    embed_backend: str = "local"  # local | openai | none
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    # "auto" picks CUDA when available and falls back to CPU, so the same image
    # is fast on a GPU host and correct on a CPU one. Measured on a Blackwell
    # laptop GPU: embedding 14 -> 112 docs/s and reranking 15.9 -> 2.0 ms/doc,
    # which is the difference between dense Add taking ~850 s and ~110 s.
    embed_device: str = "auto"
    # Base URL for the `openai` embed backend only. Previously read directly
    # from the environment inside the encoder, which kept it out of Settings
    # and therefore out of .env.example.
    embed_base_url: str | None = None
    embed_batch_size: int = 16
    # Load the encoder from the local cache only. Default true because a
    # deployment host may have no route to huggingface.co, where model loading
    # otherwise stalls for minutes on retries. Set false only when the model
    # still needs downloading, then pre-bake it (see Dockerfile).
    embed_offline: bool = True
    # Similarity floor. In a same-repository corpus every neighbour is somewhat
    # similar, so without a floor the dense channel injects uninformative
    # candidates into every query.
    dense_min_similarity: float = 0.30
    # Cap on how many memories get embedded per Add call, to bound Add latency
    # on very large trajectories.
    dense_max_per_add: int = 400

    # ---- rerank ---------------------------------------------------------
    # On by default: it is the single largest measured gain (MRR +0.093,
    # nDCG@10 +0.067) and costs Add nothing, since it runs only at search time.
    # When the model is unavailable the stage is skipped and the fused recall
    # ranking is used unchanged.
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_device: str = "auto"
    # How many fused candidates to send through the cross-encoder. It is
    # O(pool) forward passes, so this is the latency knob.
    rerank_top_n: int = 120
    # Blend: final = (1-w) * fused + w * rerank. Reranking is accurate but its
    # scores are on a different scale than rank fusion, so it is blended rather
    # than allowed to fully replace the recall ordering.
    # Swept on the proxy benchmark: 0.65 peaks (MRR 0.818, nDCG@10 0.681,
    # recall@10 0.705); 0.85 degrades, 0.45 is measurably worse.
    rerank_weight: float = 0.65
    # Temperature for converting cross-encoder logits into a 0..1 score. Fixed
    # rather than derived from the reranked set, so the same logit always maps to
    # the same contribution and results do not depend on the pool size.
    rerank_temperature: float = 2.0
    # Some cross-encoders (the bge-reranker family) emit a 0..1 relevance
    # score instead of an unbounded logit. Feeding those through the sigmoid
    # above flattens every candidate towards 0.5 and destroys the ordering, so
    # with this set the score is used as it comes.
    rerank_probability_scores: bool = False
    # Cross-encoder context cap. Memory chunks can be long; truncating keeps
    # per-pair cost bounded.
    rerank_max_chars: int = 2000
    # What the cross-encoder reads for one memory. 0 keeps the prefix above;
    # a positive value selects the window by query-term density (with
    # operative lines weighted) exactly as the returned item's content is
    # selected, instead of keeping the first N characters — which on a long
    # trajectory means keeping the opening rather than the matching part.
    #
    # Measured and left off: MRR −0.007 (p=0.63), item recall@10 +0.005
    # (p=0.58), and only 7–10 of 89 queries move at all. A memory entry is a
    # median of 151 characters here, so rerank_max_chars already covers most of
    # them whole — there is no wrong window to fix, and the selection only
    # shortens the long entries while costing latency. See eval/README.md.
    rerank_span_tokens: int = 0
    # ---- session-major assembly -----------------------------------------
    # One session averages ~96 memory entries on the proxy corpus (28 675
    # entries over 300 sessions), so every entry-counted quota above is much
    # smaller than it reads: candidate_pool 300 spans ~3 sessions and
    # rerank_top_n 120 only ~1.3. These two make the pool session-major —
    # recall deeper per channel, then keep at most N entries per session before
    # truncating to candidate_pool. 0 disables both and restores the legacy
    # entry-major pool.
    #
    # Measured on the proxy benchmark and left off: MRR +0.019 (p=0.25) against
    # item recall@10 −0.019 (p=0.25) — noise in both directions. It does widen
    # the payload, but candidate generation was already shown not to be the
    # constraint (pool 300 → 800 moved nothing), and a session that now holds
    # three pool slots costs entry-window coverage. See eval/README.md.
    candidate_per_session: int = 0
    recall_channel_depth: int = 0  # 0 = use recall_per_channel
    # Score one representative document per session instead of one per entry,
    # so the cross-encoder budget distinguishes sessions — the unit the
    # platform consumes — rather than re-ordering chunks inside the one or two
    # sessions an entry-major pool happens to contain.
    #
    # Measured and left off: every metric sat below the entry-level stage
    # (MRR −0.059, p=0.006) and lowering the blend weight drifted monotonically
    # back toward the baseline — the signature of a stage adding noise. A single
    # chunk does not stand for a 96-entry session; the cross-encoder's value
    # here is picking the best chunk within a session, which the mode above
    # already does. See eval/README.md.
    rerank_session_level: bool = False

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Settings":
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}

        def put(name: str, value: Any) -> None:
            if name in known:
                kwargs[name] = value

        put("data_dir", Path(_env_str("CODEMEM_DATA_DIR", "./data") or "./data"))
        put("db_filename", _env_str("CODEMEM_DB_FILENAME", "codemem.sqlite3"))
        put("host", _env_str("CODEMEM_HOST", "0.0.0.0"))
        put("port", _env_int("CODEMEM_PORT", 8080))
        put("api_key", _env_str("CODEMEM_API_KEY", None))
        put("log_level", (_env_str("CODEMEM_LOG_LEVEL", "INFO") or "INFO").upper())
        put("log_request_bodies", _env_bool("CODEMEM_LOG_REQUEST_BODIES", False))
        put("max_top_k", _env_int("CODEMEM_MAX_TOP_K", 1000))
        put("add_deadline_seconds", _env_float("CODEMEM_ADD_DEADLINE_SECONDS", 1500.0))
        put("target_chunk_tokens", _env_int("CODEMEM_TARGET_CHUNK_TOKENS", 320))
        put("max_chunk_tokens", _env_int("CODEMEM_MAX_CHUNK_TOKENS", 1400))
        put("hard_chunk_chars", _env_int("CODEMEM_HARD_CHUNK_CHARS", 24_000))
        put("evidence_full_count", _env_int("CODEMEM_EVIDENCE_FULL_COUNT", 8))
        put(
            "evidence_max_sessions",
            _env_int("CODEMEM_EVIDENCE_MAX_SESSIONS", 2),
        )
        put(
            "evidence_operative_promotion",
            _env_int("CODEMEM_EVIDENCE_OPERATIVE_PROMOTION", 2),
        )
        put(
            "evidence_operative_weight",
            _env_float("CODEMEM_EVIDENCE_OPERATIVE_WEIGHT", 1.0),
        )
        put(
            "max_evidence_per_session",
            _env_int("CODEMEM_MAX_EVIDENCE_PER_SESSION", 5),
        )
        put("evidence_item_tokens", _env_int("CODEMEM_EVIDENCE_ITEM_TOKENS", 800))
        put("evidence_ptr_tokens", _env_int("CODEMEM_EVIDENCE_PTR_TOKENS", 110))
        put("evidence_budget_tokens", _env_int("CODEMEM_EVIDENCE_BUDGET_TOKENS", 60_000))
        put("min_evidence_score", _env_float("CODEMEM_MIN_EVIDENCE_SCORE", 0.15))
        put("min_evidence_count", _env_int("CODEMEM_MIN_EVIDENCE_COUNT", 1))
        put("rrf_k", _env_int("CODEMEM_RRF_K", 60))
        put("recall_per_channel", _env_int("CODEMEM_RECALL_PER_CHANNEL", 120))
        put("candidate_pool", _env_int("CODEMEM_CANDIDATE_POOL", 300))
        put("listwise_enabled", _env_bool("CODEMEM_LISTWISE_ENABLED", False))
        put("listwise_weight", _env_float("CODEMEM_LISTWISE_WEIGHT", 0.5))
        put(
            "listwise_max_candidates",
            _env_int("CODEMEM_LISTWISE_MAX_CANDIDATES", 40),
        )
        put(
            "listwise_excerpt_chars",
            _env_int("CODEMEM_LISTWISE_EXCERPT_CHARS", 1800),
        )
        put("llm_enabled", _env_bool("CODEMEM_LLM_ENABLED", False))
        put("llm_base_url", _env_str("CODEMEM_LLM_BASE_URL", None))
        put("llm_api_key", _env_str("CODEMEM_LLM_API_KEY", None))
        put("llm_model", _env_str("CODEMEM_LLM_MODEL", "gpt-4o-mini"))
        put("enrich_ratio", _env_float("CODEMEM_ENRICH_RATIO", 0.25))
        put("dense_enabled", _env_bool("CODEMEM_DENSE_ENABLED", True))
        put("embed_backend", _env_str("CODEMEM_EMBED_BACKEND", "local"))
        put("embed_model", _env_str("CODEMEM_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
        put("embed_device", _env_str("CODEMEM_EMBED_DEVICE", "auto"))
        put("embed_base_url", _env_str("CODEMEM_EMBED_BASE_URL", None))
        put("embed_offline", _env_bool("CODEMEM_EMBED_OFFLINE", True))
        put(
            "dense_min_similarity",
            _env_float("CODEMEM_DENSE_MIN_SIMILARITY", 0.30),
        )
        put("dense_max_per_add", _env_int("CODEMEM_DENSE_MAX_PER_ADD", 400))
        put("rerank_enabled", _env_bool("CODEMEM_RERANK_ENABLED", True))
        put(
            "rerank_model",
            _env_str("CODEMEM_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        )
        put("rerank_device", _env_str("CODEMEM_RERANK_DEVICE", "auto"))
        put("rerank_top_n", _env_int("CODEMEM_RERANK_TOP_N", 120))
        put("rerank_span_tokens", _env_int("CODEMEM_RERANK_SPAN_TOKENS", 0))
        put(
            "candidate_per_session",
            _env_int("CODEMEM_CANDIDATE_PER_SESSION", 0),
        )
        put("recall_channel_depth", _env_int("CODEMEM_RECALL_CHANNEL_DEPTH", 0))
        put(
            "rerank_session_level",
            _env_bool("CODEMEM_RERANK_SESSION_LEVEL", False),
        )
        put("rerank_weight", _env_float("CODEMEM_RERANK_WEIGHT", 0.65))
        put("rerank_temperature", _env_float("CODEMEM_RERANK_TEMPERATURE", 2.0))
        return cls(**kwargs)
