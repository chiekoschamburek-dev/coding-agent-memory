"""Time-budget guard and adaptive degradation (plan.md section 5).

plan.md section 2.3 is the reason this module exists: with ~1800 Add requests
inside a 0.5-2 day run, the *real* per-request budget is tens of seconds, not
the 30-minute server ceiling. So model work is optional, budgeted, and
abandoned the moment it overruns.

Three mechanisms:

* ``ModuleBudget`` -- a deadline handle. Every step checks ``expired`` and
  ``time_left``; nothing waits past its deadline.
* ``Deadline`` -- a hard cut-off applied to any callable, so a hung provider
  cannot stretch a request toward the ceiling.
* ``AdaptiveGuard`` -- the sliding-window controller from plan.md section 5.
  Pure in-memory; it must not add an I/O dependency to the hot path.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from .config import Settings, get_settings

T = TypeVar("T")

MODE_FULL = "full"
MODE_LITE = "extract_lite"
MODE_LEX_ONLY = "lex_only"


class BudgetExpired(Exception):
    """Raised when a step exceeds its allotted time."""


@dataclass
class StepRecord:
    name: str
    duration_ms: int
    skipped: bool
    reason: str = ""


@dataclass
class ModuleBudget:
    """Deadline for one request, shared across its steps."""

    total_seconds: float
    started_at: float = field(default_factory=time.monotonic)
    steps: list[StepRecord] = field(default_factory=list)
    # Safety margin so a step is not started right at the edge and then
    # overrun. Scaled to the budget: a fixed 0.25s would consume the entire
    # allowance of a short budget and make it unusable.
    margin_seconds: float | None = None

    @property
    def margin(self) -> float:
        if self.margin_seconds is not None:
            return self.margin_seconds
        return min(0.25, self.total_seconds * 0.05)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def time_left(self) -> float:
        return self.total_seconds - self.elapsed

    @property
    def expired(self) -> bool:
        return self.time_left <= 0

    def can_spend(self, seconds: float) -> bool:
        """True if ``seconds`` fits inside what is left, plus the margin."""
        return self.time_left > seconds + self.margin

    def guard(self, name: str, fn: Callable[[], T], *, allow_skip: bool = True) -> T | None:
        """Run ``fn`` only if it can be started and finished in budget."""
        if allow_skip and self.expired:
            self.steps.append(StepRecord(name, 0, True, "budget_expired"))
            return None
        started = time.monotonic()
        try:
            result = fn()
        except BudgetExpired:
            self.steps.append(StepRecord(name, int((time.monotonic() - started) * 1000), True, "deadline"))
            return None
        except Exception:
            self.steps.append(StepRecord(name, int((time.monotonic() - started) * 1000), True, "error"))
            raise
        self.steps.append(StepRecord(name, int((time.monotonic() - started) * 1000), False))
        return result



class AdaptiveGuard:
    """Sliding-window health controller (plan.md section 5 code block).

    The plan distinguishes two window-level signals, and the distinction
    matters:

    * ``R_slow``  -- the fraction of requests that *overran* their budget.
      Sustained overruns mean something is structurally too expensive, so the
      whole system steps down to ``lite``.
    * ``R_token`` -- the window's **p50** elapsed time over the budget. A p50
      close to the budget means requests are routinely near the edge even
      without crossing it, which individual overrun counting would miss
      entirely. When p50 sits above the skip threshold, model work is dropped
      for the current request.

    Recovery requires a streak of healthy requests so a single fast request
    cannot flip the system back to full processing.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._lock = threading.Lock()
        self._durations: deque[float] = deque(maxlen=self.settings.guard_window)
        self._slow_flags: deque[bool] = deque(maxlen=self.settings.guard_window)
        self._mode = MODE_FULL
        self._healthy_streak = 0
        self._transitions: deque[tuple[float, str, str]] = deque(maxlen=100)

    def _p50(self) -> float:
        if not self._durations:
            return 0.0
        ordered = sorted(self._durations)
        return ordered[len(ordered) // 2]

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def _ratios(self, budget_s: float) -> tuple[float, float]:
        slow_ratio = (
            sum(1 for flag in self._slow_flags if flag) / len(self._slow_flags)
            if self._slow_flags
            else 0.0
        )
        token_ratio = self._p50() / budget_s if budget_s > 0 else 1.0
        return slow_ratio, token_ratio

    def observe(self, *, elapsed_s: float, budget_s: float, overran: bool) -> str:
        """Record one request outcome and return the mode for the *next* one."""
        with self._lock:
            self._slow_flags.append(overran or elapsed_s > budget_s)
            self._durations.append(elapsed_s)
            slow_ratio, token_ratio = self._ratios(budget_s)

            previous = self._mode
            if (
                token_ratio > self.settings.guard_token_ratio_skip
                or slow_ratio > self.settings.guard_slow_ratio_enter
            ):
                # p50 near the ceiling -> skip model work (lex-only).
                # Too many real overruns -> shed model work (lite).
                next_mode = (
                    MODE_LEX_ONLY
                    if token_ratio > self.settings.guard_token_ratio_skip
                    else MODE_LITE
                )
                self._healthy_streak = 0
            elif (
                slow_ratio < self.settings.guard_slow_ratio_exit
                and token_ratio < self.settings.guard_token_ratio_healthy
            ):
                self._healthy_streak += 1
                if self._healthy_streak >= self.settings.guard_recover_streak:
                    next_mode = MODE_FULL
                else:
                    # Climb one rung at a time, never skip straight to full.
                    next_mode = MODE_LITE if previous == MODE_LEX_ONLY else previous
            else:
                next_mode = previous

            self._mode = next_mode
            if next_mode != previous:
                self._transitions.append((time.time(), previous, next_mode))
            return next_mode

    def plan_for_request(self, budget_s: float) -> str:
        """Mode for one request about to start.

        A request may be downgraded below the steady-state mode, but never
        upgraded: dropping model work for a single request is always safe,
        while resuming it early is what provokes the next overrun.
        """
        with self._lock:
            _, token_ratio = self._ratios(budget_s)
            if token_ratio > self.settings.guard_token_ratio_skip:
                return MODE_LEX_ONLY
            return self._mode

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            slow_ratio, token_ratio = self._ratios(self.settings.add_sync_budget_s)
            return {
                "mode": self._mode,
                "samples": len(self._durations),
                "slow_ratio": round(slow_ratio, 4),
                "token_ratio": round(token_ratio, 4),
                "p50_elapsed_s": round(self._p50(), 3),
                "healthy_streak": self._healthy_streak,
                "recent_transitions": [
                    {"at": at, "from": src, "to": dst} for at, src, dst in list(self._transitions)[-5:]
                ],
            }

    def reset(self) -> None:
        with self._lock:
            self._durations.clear()
            self._slow_flags.clear()
            self._mode = MODE_FULL
            self._healthy_streak = 0
            self._transitions.clear()


def _mode_order(mode: str) -> int:
    return {MODE_LEX_ONLY: 0, MODE_LITE: 1, MODE_FULL: 2}.get(mode, 0)


_guard: AdaptiveGuard | None = None


def get_guard(settings: Settings | None = None) -> AdaptiveGuard:
    global _guard
    if _guard is None:
        _guard = AdaptiveGuard(settings)
    return _guard


def reset_guard_for_tests() -> None:
    global _guard
    _guard = None