"""End-to-end Answer evaluation.

Answers each question twice — with and without retrieved memory — and reports
both accuracies plus the difference. The difference is the only number that
attributes anything to the memory system, because absolute accuracy is dominated
by what the answer model already knew.

Data flow mirrors the platform's: Add all sessions, then for each question run
Search, take the returned candidates **in returned rank order**, and pass a
token-counted prefix of them to the answer model as context. The answer model is
never told which candidate is relevant, and the question carries no gold answer.

Scoring is exact match on the chosen option. No judge is involved for the
multiple-choice questions, which avoids the self-preference bias that arises when
one model both answers and grades.

Usage::

    python eval/run_endtoend.py --qa eval/data/qa.json --data eval/data/benchmark.json
    python eval/run_endtoend.py ... --conditions no_memory,with_memory
    python eval/run_endtoend.py ... --limit 20          # quick signal
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")

# --------------------------------------------------------------- answer -----

_LETTERS = "ABCDEFGH"

# The platform locks its own answer prompt; this is a faithful stand-in and is
# recorded in the output metadata so results are never read as official.
_ANSWER_SYSTEM = (
    "You are a software engineer working in an existing repository. "
    "Answer the question and reply with ONLY the letter of the correct option, "
    "for example: A"
)

_ANSWER_TEMPLATE = """Question about a software repository:

{question}

Options:
{options}
{context}"""

_CONTEXT_TEMPLATE = """
Here are memory records retrieved from this repository's past engineering work.
They may be relevant, partly relevant, or irrelevant; each is verbatim from a
recorded session. Use them only if they help answer the question.

{records}
"""


def format_context(items: list[dict], budget_tokens: int) -> tuple[str, int]:
    """Render returned memories as a token-counted prefix, in returned order.

    Mirrors the platform, which keeps a prefix of candidates in the order the
    system returned them. Order therefore decides what survives.
    """
    from codemem.core.tokens import count_tokens

    if not items:
        return "", 0
    lines: list[str] = []
    used = 0
    for idx, item in enumerate(items, start=1):
        block = f"[{idx}] {item['content']}"
        cost = count_tokens(block)
        if used + cost > budget_tokens and lines:
            break
        lines.append(block)
        used += cost
    if not lines:
        return "", 0
    return _CONTEXT_TEMPLATE.format(records="\n\n".join(lines)), len(lines)


def parse_choice(text: str, n_options: int) -> int | None:
    """Extract the chosen option index from a model reply.

    Tolerant of the usual shapes ("B", "B.", "(B)", "the answer is B") but
    rejects anything ambiguous rather than guessing, so an unparseable reply
    counts as wrong instead of silently matching option 0.
    """
    if not text:
        return None
    text = text.strip()
    patterns = [
        r"^\s*\(?([A-Ha-h])\)?\s*[.):]?\s*$",
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-Ha-h])\)?\b",
        r"^\s*\(?([A-Ha-h])\)?\s*[.):]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            index = _LETTERS.index(match.group(1).upper())
            return index if index < n_options else None
    return None


class Answerer:
    """Answers questions, optionally with retrieved memory as context."""

    def __init__(self, model: str, base_url: str, api_key: str, timeout: float = 120.0):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.calls = 0
        self.failures = 0

    def answer(
        self, question: str, options: list[str], context: str = ""
    ) -> tuple[int | None, str]:
        rendered = "\n".join(
            f"{_LETTERS[i]}. {opt}" for i, opt in enumerate(options)
        )
        user = _ANSWER_TEMPLATE.format(
            question=question, options=rendered, context=context
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _ANSWER_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=16,
            )
            self.calls += 1
            raw = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            self.failures += 1
            return None, f"ERROR: {type(exc).__name__}: {str(exc)[:120]}"
        return parse_choice(raw, len(options)), raw


class Judge:
    """Optional LLM judge, for open-ended variants. Not used by MC scoring."""

    def __init__(self, model: str, base_url: str, api_key: str, timeout: float = 120.0):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def grade(self, question: str, gold: str, proposed: str) -> float | None:
        prompt = (
            "You grade a software-engineering answer.\n"
            f"Question:\n{question}\n\n"
            f"Reference answer (the change that actually resolved this):\n{gold}\n\n"
            f"Proposed answer:\n{proposed}\n\n"
            "Reply with a single number from 0 to 100 for correctness. "
            "Reply with only the number."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=8,
            )
            match = re.search(r"\d+", response.choices[0].message.content or "")
            return float(match.group(0)) / 100.0 if match else None
        except Exception:
            return None


# ------------------------------------------------------------------ run -----


def run_condition(
    label: str,
    questions: list[dict],
    answerer: Answerer,
    *,
    settings,
    memories: list[dict],
    with_memory: bool,
    budget_tokens: int,
    top_k: int,
    progress_every: int = 10,
    repeats: int = 1,
) -> dict:
    """Run every question, optionally repeating each one.

    Repeats exist because a non-deterministic answer model turns a single pass
    into a coin flip on borderline questions. Measured on one question at
    temperature=0: 4 "B" and 4 "C" in eight identical calls, and an explicit
    ``seed`` did not stabilise it. Reporting one pass would therefore present
    noise as a result; the caller gets the per-pass spread *and* a majority vote,
    which is the better estimator of what the model "really" thinks.
    """
    """Run every question under one condition and return per-question outcomes."""
    from fastapi.testclient import TestClient

    from codemem.api.app import create_app

    app = create_app(settings)
    outcomes: list[dict] = []

    with TestClient(app) as client:
        store = app.state.container.store
        if with_memory:
            for memory in memories:
                client.post(
                    "/add",
                    json={
                        "request_id": f"e2e:{memory['id']}",
                        "user_id": memory["user_id"],
                        "session_id": memory["session_id"],
                        "messages": memory["messages"],
                    },
                )
            # Map memory id -> session so we can report whether the retrieved
            # context actually contained a file-overlap-relevant session.
            with store._read() as conn:  # noqa: SLF001
                mapping = {
                    f"mem_{row['id']}": row["session_id"]
                    for row in conn.execute("SELECT id, session_id FROM memory")
                }
        else:
            mapping = {}

        for i, question in enumerate(questions, 1):
            context = ""
            n_shown = 0
            n_relevant_shown = 0
            answer_shown = False  # no memory is supplied in the baseline condition
            if with_memory:
                user_id = _user_for_repo(memories, question["repo"])
                response = client.post(
                    "/search",
                    json={
                        "query": question["question"],
                        # The platform sends options for multiple-choice questions,
                        # so they are sent here too.
                        "options": question["options"],
                        "user_id": user_id,
                        "top_k": top_k,
                    },
                )
                data = response.json().get("data", [])
                context, n_shown = format_context(data, budget_tokens)

                relevant = set(question.get("relevant_sessions") or [])
                shown_sessions = {mapping.get(item["id"]) for item in data[:n_shown]}
                n_relevant_shown = len(shown_sessions & relevant)
                # Whether the session holding the answer was actually retrieved.
                # This is the diagnostic that decides how to read the result: a
                # failure to recall is a retrieval problem, a failure to use a
                # recalled answer is an answer-model problem.
                answer_session = question.get("answer_session")
                answer_shown = bool(
                    answer_session and answer_session in shown_sessions
                )

            votes: list[int | None] = []
            raws: list[str] = []
            for _ in range(max(1, repeats)):
                choice, raw = answerer.answer(
                    question["question"], question["options"], context
                )
                votes.append(choice)
                raws.append(raw)

            # Majority vote over the repeats; ties resolve to the first vote.
            tally: dict[int, int] = {}
            for vote in votes:
                if vote is not None:
                    tally[vote] = tally.get(vote, 0) + 1
            choice = max(tally, key=lambda k: (tally[k], -votes.index(k))) if tally else None
            raw = raws[0]
            outcomes.append(
                {
                    "query_id": question["query_id"],
                    "repo": question["repo"],
                    "gold_index": question["gold_index"],
                    "chosen_index": choice,
                    "correct": choice == question["gold_index"],
                    "unanimous": len(tally) <= 1,
                    "votes": votes,
                    "votes_correct": [v == question["gold_index"] for v in votes],
                    "parsed": choice is not None,
                    "n_shown": n_shown,
                    "n_relevant_shown": n_relevant_shown,
                    "answer_session_shown": answer_shown,
                    "reply": raw[:40],
                }
            )
            if progress_every and i % progress_every == 0:
                acc = sum(o["correct"] for o in outcomes) / len(outcomes)
                print(f"    {label}: {i}/{len(questions)} acc={acc:.3f}", flush=True)

    per_pass = [
        sum(o["votes_correct"][i] for o in outcomes) / len(outcomes)
        for i in range(max(1, repeats))
    ]
    return {
        "label": label,
        "n": len(outcomes),
        "repeats": max(1, repeats),
        "accuracy": sum(o["correct"] for o in outcomes) / len(outcomes),
        "per_pass_accuracy": per_pass,
        "per_pass_min": min(per_pass),
        "per_pass_max": max(per_pass),
        "unanimous_rate": sum(o["unanimous"] for o in outcomes) / len(outcomes),
        "unparsed_rate": sum(not o["parsed"] for o in outcomes) / len(outcomes),
        "outcomes": outcomes,
    }


def redact(config: dict) -> dict:
    """Remove credentials from a config before printing or persisting it.

    An earlier revision printed and stored the whole config, which wrote the
    answer model's API key in cleartext into the console and the results JSON.
    Never persist a secret: redact by key name so a future field cannot leak by
    being forgotten here.
    """
    sensitive = ("api_key", "key", "token", "secret", "password")
    out: dict = {}
    for key, value in config.items():
        if any(marker in key.lower() for marker in sensitive):
            out[key] = "<redacted>" if value else value
        else:
            out[key] = value
    return out


def _user_for_repo(memories: list[dict], repo: str) -> str:
    for memory in memories:
        if memory["repo"] == repo:
            return memory["user_id"]
    return f"bench:{repo}"


def report(results: dict[str, dict], meta: dict) -> None:
    print()
    print("=" * 70)
    print("End-to-end Answer evaluation (file localisation, multiple choice)")
    print("=" * 70)

    print(f"{'condition':<20}{'n':>5}{'majority':>10}{'per-pass range':>18}{'unanimous':>11}")
    print("-" * 64)
    for label, result in results.items():
        spread = f"{result.get('per_pass_min', result['accuracy']):.3f}-{result.get('per_pass_max', result['accuracy']):.3f}"
        print(
            f"{label:<20}{result['n']:>5}{result['accuracy']:>10.3f}"
            f"{spread:>18}{result.get('unanimous_rate', 1.0):>11.3f}"
        )
    print()
    print("  'majority' aggregates repeats per question; 'per-pass range' is the")
    print("  spread across individual passes. A wide range means the answer model is")
    print("  not deterministic and single-pass numbers would be noise.")

    if "no_memory" in results and "with_memory" in results:
        base = results["no_memory"]["accuracy"]
        with_mem = results["with_memory"]["accuracy"]
        delta = with_mem - base
        print()
        print(f"  memory contribution: {delta:+.3f} "
              f"({base:.3f} -> {with_mem:.3f})")
        if delta > 0:
            print("  The retrieved memory improved the answer model's accuracy.")
        elif delta < 0:
            print("  The retrieved memory HURT accuracy: noise displaced the")
            print("  model's own reasoning. This is the failure mode the")
            print("  relevant/noisy conditions in the real track test for.")
        else:
            print("  No measurable difference.")

    print()
    print("definition:")
    for line in _wrap(meta.get("gold_definition", ""), 64):
        print(f"  {line}")
    print("caveats:")
    for caveat in meta.get("caveats", []):
        for line in _wrap(caveat, 64):
            print(f"  {line}")


def _wrap(text: str, width: int) -> list[str]:
    words = (text or "").split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, default=Path("eval/data/qa.json"))
    parser.add_argument("--data", type=Path, default=Path("eval/data/benchmark.json"))
    parser.add_argument("--out", type=Path, default=Path("eval/results/endtoend.json"))
    parser.add_argument("--limit", type=int, default=None, help="question subset")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--conditions", default="no_memory,with_memory",
        help="comma-separated; no_memory isolates the model's own prior",
    )
    parser.add_argument(
        "--budget-tokens", type=int, default=60_000,
        help="cap on the memory context; the platform reserves 117,760 input tokens",
    )
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--answer-base-url", default=None)
    parser.add_argument("--answer-api-key", default=None)
    parser.add_argument("--dense-enabled", default=None)
    parser.add_argument("--rerank-enabled", default=None)
    parser.add_argument("--listwise-enabled", default=None)
    parser.add_argument("--item-tokens", type=int, default=None,
                        help="cap for items rendered in full form")
    parser.add_argument("--ptr-tokens", type=int, default=None,
                        help="cap for the remaining (pointer-form) items")
    parser.add_argument("--full-count", type=int, default=None,
                        help="how many items use the full-form cap")
    parser.add_argument("--budget", type=int, default=None,
                        help="override the total evidence token budget")
    parser.add_argument("--operative-weight", type=float, default=None,
                        help="weight for operative lines when selecting a window")
    parser.add_argument("--repeats", type=int, default=1,
                        help="answer each question N times and majority-vote")
    parser.add_argument("--cards", action="store_true",
                        help="enable L3 experience cards (one overview per "
                             "session, scored but never emitted); spends one "
                             "LLM call per Add")
    parser.add_argument("--session-feature-fusion", action="store_true",
                        help="rank-fuse two session-level signals into the "
                             "session order")
    parser.add_argument("--session-score-topk", type=int, default=None,
                        help="session score = sum of the top-k member scores "
                             "instead of the max (1 = shipped max estimator)")
    parser.add_argument("--card-expansion", action="store_true",
                        help="relax card invariant 5: a gated card may vouch "
                             "its session's verbatim tail chunks into the "
                             "payload when no chunk was admitted on its own")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not args.qa.exists():
        print(f"error: {args.qa} not found; run eval/build_qa.py first", file=sys.stderr)
        return 2
    if not args.data.exists():
        print(f"error: {args.data} not found; run eval/build_benchmark.py first", file=sys.stderr)
        return 2

    # Load .env so credentials come from the documented place.
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

    model = args.answer_model or os.environ.get("CODEMEM_LLM_MODEL", "gpt-4o-mini")
    base_url = args.answer_base_url or os.environ.get("CODEMEM_LLM_BASE_URL")
    api_key = args.answer_api_key or os.environ.get("CODEMEM_LLM_API_KEY")
    if not base_url or not api_key:
        print(
            "error: answer model not configured "
            "(set CODEMEM_LLM_BASE_URL / CODEMEM_LLM_API_KEY in .env)",
            file=sys.stderr,
        )
        return 2

    qa = json.loads(args.qa.read_text(encoding="utf-8"))
    bench = json.loads(args.data.read_text(encoding="utf-8"))

    relevant_by_instance = {
        q["instance_id"]: q.get("relevant", []) for q in bench["queries"]
    }
    questions = qa["questions"]
    for question in questions:
        question["relevant_sessions"] = relevant_by_instance.get(
            question["instance_id"], []
        )
    if args.limit:
        questions = questions[: args.limit]

    memories = bench["memories"]

    from codemem.core.config import Settings

    overrides: dict = {}
    if args.cards:
        overrides["card_enabled"] = True
    if args.card_expansion:
        overrides["card_expansion"] = True
    if getattr(args, "session_score_topk", None) is not None:
        overrides["session_score_topk"] = args.session_score_topk
    if getattr(args, "session_feature_fusion", False):
        overrides["session_feature_fusion"] = True
    if args.dense_enabled is not None:
        overrides["dense_enabled"] = args.dense_enabled.lower() == "true"
    if args.rerank_enabled is not None:
        overrides["rerank_enabled"] = args.rerank_enabled.lower() == "true"
    if args.item_tokens is not None:
        overrides["evidence_item_tokens"] = args.item_tokens
    if args.ptr_tokens is not None:
        overrides["evidence_ptr_tokens"] = args.ptr_tokens
    if args.full_count is not None:
        overrides["evidence_full_count"] = args.full_count
    if args.budget is not None:
        overrides["evidence_budget_tokens"] = args.budget
    if args.operative_weight is not None:
        overrides["evidence_operative_weight"] = args.operative_weight
    if args.listwise_enabled is not None:
        overrides["listwise_enabled"] = args.listwise_enabled.lower() == "true"
    if args.budget_tokens:
        overrides["evidence_budget_tokens"] = args.budget_tokens
    overrides.setdefault("listwise_enabled", os.environ.get("CODEMEM_LISTWISE_ENABLED", "false").lower() == "true")
    overrides.setdefault("llm_base_url", base_url)
    overrides.setdefault("llm_api_key", api_key)
    overrides.setdefault("llm_model", model)

    # Answer-eval runs make hundreds of model calls; keep the service log quiet
    # so the report is readable.
    import logging

    logging.disable(logging.INFO)

    print(f"answer model  : {model} @ {base_url}")
    print(f"questions     : {len(questions)}")
    print(f"memories      : {len(memories)}")
    print(f"conditions    : {args.conditions}")
    print(f"retrieval cfg : {redact(overrides)}")

    answerer = Answerer(model=model, base_url=base_url, api_key=api_key)
    results: dict[str, dict] = {}
    started = time.time()

    for condition in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        with_memory = condition == "with_memory"
        settings = Settings.from_env()
        settings.data_dir = Path(tempfile.mkdtemp())
        for key, value in overrides.items():
            setattr(settings, key, value)
        print(f"\n  running condition: {condition}")
        results[condition] = run_condition(
            condition,
            questions,
            answerer,
            settings=settings,
            memories=memories,
            with_memory=with_memory,
            budget_tokens=args.budget_tokens,
            top_k=args.top_k,
            progress_every=0 if args.quiet else 10,
            repeats=args.repeats,
        )

    report(results, qa.get("meta", {}))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            **qa.get("meta", {}),
            "answer_model": model,
            "answer_base_url": base_url,
            "retrieval_config": redact(overrides),
            "budget_tokens": args.budget_tokens,
            "top_k": args.top_k,
            "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "answer_calls": answerer.calls,
            "answer_failures": answerer.failures,
            "wall_seconds": round(time.time() - started, 1),
        },
        "results": {
            label: {k: v for k, v in result.items() if k != "outcomes"}
            for label, result in results.items()
        },
        "outcomes": {label: result["outcomes"] for label, result in results.items()},
    }
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
