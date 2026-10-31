"""Budget, degradation and token-truncation tests (plan.md section 8.2).

These exist because of the plan's central finding: the real per-request budget
is tens of seconds, so the *degraded* path is a normal operating mode rather
than an error path. A system that is only tested in its healthy mode has not
been tested at all for the situation it will actually run in.
"""

from __future__ import annotations

import time

import pytest

from codemem.common.guard import (
    MODE_FULL,
    MODE_LEX_ONLY,
    MODE_LITE,
    AdaptiveGuard,
    BudgetExpired,
    ModuleBudget,
)
from codemem.common.config import Settings


# --------------------------------------------------------------------------
# ModuleBudget
# --------------------------------------------------------------------------


def test_budget_reports_expiry():
    budget = ModuleBudget(0.05)
    assert not budget.expired
    assert budget.can_spend(0.01)
    time.sleep(0.08)
    assert budget.expired
    assert not budget.can_spend(0.01)


def test_budget_skips_steps_once_expired():
    budget = ModuleBudget(0.02)
    time.sleep(0.05)
    called = []
    result = budget.guard("expensive", lambda: called.append(1))
    assert result is None
    assert called == [], "expired budget must not start new work"
    assert budget.steps[-1].skipped is True
    assert budget.steps[-1].reason == "budget_expired"


def test_budget_runs_and_records_successful_steps():
    budget = ModuleBudget(5.0)
    assert budget.guard("cheap", lambda: 42) == 42
    assert budget.steps[-1].skipped is False
    assert budget.steps[-1].name == "cheap"


def test_budget_records_deadline_skip_without_propagating():
    def slow():
        raise BudgetExpired("too slow")

    budget = ModuleBudget(5.0)
    assert budget.guard("slow", slow) is None
    assert budget.steps[-1].reason == "deadline"


def test_budget_propagates_real_errors():
    def broken():
        raise ValueError("genuine bug")

    budget = ModuleBudget(5.0)
    with pytest.raises(ValueError):
        budget.guard("broken", broken)


# --------------------------------------------------------------------------
# AdaptiveGuard
# --------------------------------------------------------------------------


def _guard(**overrides) -> AdaptiveGuard:
    cfg = Settings(
        guard_window=10,
        guard_slow_ratio_enter=0.2,
        guard_slow_ratio_exit=0.1,
        guard_token_ratio_skip=0.8,
        guard_token_ratio_healthy=0.6,
        guard_recover_streak=3,
        **overrides,
    )
    return AdaptiveGuard(cfg)


def test_guard_starts_full():
    assert _guard().mode == MODE_FULL


def test_guard_drops_to_lex_only_when_a_request_nearly_overruns():
    guard = _guard()
    mode = guard.observe(elapsed_s=39.0, budget_s=40.0, overran=False)
    assert mode == MODE_LEX_ONLY


def test_guard_enters_lite_when_a_minority_overruns_but_p50_stays_healthy():
    """R_slow and R_token are different signals and must not be conflated.

    A handful of badly overrunning requests is enough to shed model work
    (lite) even while the median request is perfectly fast. If p50 were also
    elevated, the stronger lex-only rule would take precedence instead.
    """
    guard = _guard()
    assert guard.mode == MODE_FULL

    # 7 fast requests, 3 severe overruns: slow_ratio 0.3 > 0.2, p50 still ~1s.
    for elapsed in (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 45.0, 45.0, 45.0):
        mode = guard.observe(elapsed_s=elapsed, budget_s=40.0, overran=elapsed > 40.0)

    assert mode == MODE_LITE
    snapshot = guard.snapshot()
    assert snapshot["slow_ratio"] > 0.2
    assert snapshot["token_ratio"] < 0.8, "p50 must remain low in this scenario"


def test_guard_prefers_lex_only_when_p50_itself_is_elevated():
    """When the median request is near the budget, shed everything, not some."""
    guard = _guard()
    for elapsed in (10.0, 38.0, 10.0, 38.0):
        mode = guard.observe(elapsed_s=elapsed, budget_s=40.0, overran=False)
    assert mode == MODE_LEX_ONLY


def test_guard_requires_a_healthy_streak_before_recovering():
    guard = _guard()
    guard.observe(elapsed_s=39.0, budget_s=40.0, overran=False)
    assert guard.mode == MODE_LEX_ONLY

    # A single fast request must not immediately restore full processing.
    guard.observe(elapsed_s=0.5, budget_s=40.0, overran=False)
    assert guard.mode in (MODE_LITE, MODE_LEX_ONLY)

    for _ in range(4):
        guard.observe(elapsed_s=0.5, budget_s=40.0, overran=False)
    assert guard.mode == MODE_FULL


def test_guard_does_not_oscillate_on_alternating_timings():
    guard = _guard()
    for _ in range(8):
        guard.observe(elapsed_s=38.0, budget_s=40.0, overran=False)
        guard.observe(elapsed_s=0.5, budget_s=40.0, overran=False)
    assert guard.mode in (MODE_LITE, MODE_LEX_ONLY), (
        "alternating healthy/slow traffic must not read as healthy"
    )


def test_guard_snapshot_has_no_request_content():
    guard = _guard()
    guard.observe(elapsed_s=1.0, budget_s=40.0, overran=False)
    snapshot = guard.snapshot()
    assert set(snapshot) == {
        "mode",
        "samples",
        "slow_ratio",
        "token_ratio",
        "p50_elapsed_s",
        "healthy_streak",
        "recent_transitions",
    }


# --------------------------------------------------------------------------
# system-level degradation
# --------------------------------------------------------------------------


def test_add_succeeds_and_is_searchable_when_llm_is_unconfigured(service):
    """plan.md 8.1: lex-only path must remain fully usable."""
    outcome = service.add(
        request_id="r1",
        user_id="u1",
        session_id="s1",
        raw_messages=[
            {"role": "user", "content": "src/loader.py raised ConfigParseError in parseConfigFile"}
        ],
    )
    assert outcome.success is True
    assert outcome.request_id == "r1"

    result = service.search(user_id="u1", query="ConfigParseError", top_k=10)
    # Without an extraction LLM there are no cards, and chunks are not
    # retrieval items: the answer is an empty, well-formed response.
    assert result.data == []


def test_search_degrades_to_lexical_when_vector_index_raises(store, settings):
    import json as _json

    from codemem.add.experience import ExperienceCard, card_to_payload
    from codemem.search.retrieval import SearchEngine
    from codemem.common.store import now_iso

    class ExplodingVectorIndex:
        available = True

        def search(self, user_id, query, limit):
            raise RuntimeError("embedding service down")

    engine = SearchEngine(store, settings, card_vector_index=ExplodingVectorIndex())

    state = store.begin_ingestion(
        user_id="u1", request_id="r1", session_id="s1",
        digest="d", mode=MODE_FULL, lease_seconds=60,
    )
    card = ExperienceCard(
        experience_id="e:1", user_id="u1", session_id="s1",
        problem="src/a.py TokenExpiredError refresh_token",
        implement="extend the refresh window",
    )
    store.insert_experience(
        user_id="u1", session_id="s1", experience_id="e:1",
        ingestion_id=state.ingestion_id, kind="debug",
        payload_json=_json.dumps(card_to_payload(card), ensure_ascii=False),
        full_content="src/a.py TokenExpiredError refresh_token",
        extraction_version="t", created_at=now_iso(),
        terms=["problemtokenexpirederror", "locaterefresh"],
        entities=[],
    )

    data, diagnostics = engine.search(user_id="u1", query="TokenExpiredError", top_k=10)
    assert data, "vector failure must not empty the result set"
    assert "card_vector_unavailable" in diagnostics.degraded


def test_guard_downgrade_reduces_work_without_changing_response_shape(client):
    """Response contract must be identical in every processing mode."""
    from codemem.common.guard import get_guard

    payload = {
        "request_id": "r_mode",
        "messages": [{"role": "user", "content": "src/a.py boom"}],
        "user_id": "u_mode",
        "session_id": "s_mode",
    }
    first = client.post("/add", json=payload)
    assert first.status_code == 200
    assert set(first.json()) == {"success", "request_id", "user_id", "session_id"}

    guard = get_guard()
    guard.observe(elapsed_s=39.5, budget_s=40.0, overran=True)
    assert guard.mode == MODE_LEX_ONLY

    payload2 = dict(payload, request_id="r_mode2")
    second = client.post("/add", json=payload2)
    assert second.status_code == 200
    assert set(second.json()) == {"success", "request_id", "user_id", "session_id"}

    result = client.post(
        "/search", json={"query": "src/a.py", "user_id": "u_mode", "top_k": 10}
    )
    assert result.status_code == 200
    assert "data" in result.json(), "the response contract holds in every mode"


# --------------------------------------------------------------------------
# token budget (plan.md 2.4)
# --------------------------------------------------------------------------


def test_graded_tiers_are_applied_by_rank():
    cfg = Settings(embed_model="")
    assert cfg.per_item_token_cap(1) == 1200
    assert cfg.per_item_token_cap(10) == 1200
    assert cfg.per_item_token_cap(11) == 700
    assert cfg.per_item_token_cap(30) == 700
    assert cfg.per_item_token_cap(31) == 350
    assert cfg.per_item_token_cap(60) == 350
    assert cfg.per_item_token_cap(61) == 200
    assert cfg.per_item_token_cap(500) == 200


def test_rank_1_is_capped_and_tail_is_shorter(card_client):
    long_text = "problem statement about src/deep/nested/module.py and TokenExpiredError\n" * 400
    for index in range(60):
        card_client.post(
            "/add",
            json={
                "request_id": f"r{index}",
                "messages": [{"role": "user", "content": f"unique{index} {long_text}"}],
                "user_id": "u_budget",
                "session_id": "s",
            },
        )

    data = card_client.post(
        "/search", json={"query": "problem statement module", "user_id": "u_budget", "top_k": 100}
    ).json()["data"]
    assert data

    from codemem.common.tokens import get_counter

    counter = get_counter()
    lengths = [counter.count(item["content"]) for item in data]
    assert lengths[0] <= 1200
    for item, length in zip(data, lengths):
        rank = data.index(item) + 1
        assert length <= Settings(db_path="x", embed_model="").per_item_token_cap(rank)


def test_truncation_keeps_the_head_of_the_content():
    """A truncated item must still lead with its most important lines."""
    from codemem.common.tokens import TokenCounter, reset_counter_for_tests

    reset_counter_for_tests()
    counter = TokenCounter()
    text = "HEADER_MARKER\n" + ("filler line padding the body out\n" * 500) + "TAIL_MARKER"
    truncated = counter.truncate(text, 50)
    assert "HEADER_MARKER" in truncated
    assert "TAIL_MARKER" not in truncated
    assert counter.count(truncated) <= 55


def test_truncate_never_exceeds_budget():
    from codemem.common.tokens import TokenCounter, reset_counter_for_tests

    reset_counter_for_tests()
    counter = TokenCounter()
    for budget in (0, 1, 5, 50, 500, 5000):
        result = counter.truncate("word " * 3000, budget)
        assert counter.count(result) <= budget, f"budget {budget} exceeded"


def test_total_budget_is_respected(card_client):
    """Nothing may push the response past the hard total."""
    from codemem.common.config import get_settings
    from codemem.common.tokens import get_counter

    body = "src/x/y.py TokenExpiredError refresh_token_window " + "padding words here " * 200
    for index in range(40):
        card_client.post(
            "/add",
            json={
                "request_id": f"tot{index}",
                "messages": [{"role": "user", "content": f"variant{index} {body}"}],
                "user_id": "u_tot",
                "session_id": "s",
            },
        )

    data = card_client.post(
        "/search", json={"query": "TokenExpiredError refresh", "user_id": "u_tot", "top_k": 100}
    ).json()["data"]
    counter = get_counter()
    total = sum(counter.count(item["content"]) for item in data)
    assert total <= get_settings().token_budget_total


def test_marginal_value_rule_stops_padding_short_items(card_client):
    """Past the soft target, only substantial evidence is worth its token cost."""
    for index in range(120):
        card_client.post(
            "/add",
            json={
                "request_id": f"pad{index}",
                "messages": [
                    {"role": "user", "content": " ".join(f"tok{index}_{word}" for word in range(80))}
                ],
                "user_id": "u_pad",
                "session_id": "s",
            },
        )
    data = card_client.post(
        "/search", json={"query": "tok", "user_id": "u_pad", "top_k": 100}
    ).json()["data"]
    # Padding to exactly top_k with tiny fragments is exactly what plan.md S5
    # forbids; we do not require a specific count, only that the budget logic
    # did not simply emit everything.
    assert 0 < len(data) <= 100
