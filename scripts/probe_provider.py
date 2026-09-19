"""Behavioral probe for the LLM endpoint.

The open-source division requires Add/Search to use ``gpt-4o-mini``, and the
platform reproduces the submission — if a proxy silently routes to a different
model, our scores may not reproduce and the entry can be voided. A model's
self-description proves nothing (any proxy can answer "I am gpt-4o-mini"), so
this probe checks *behavioural* fingerprints instead.

Usage::

    python scripts/probe_provider.py                       # uses env config
    python scripts/probe_provider.py --base-url URL --api-key KEY

It reports, for each check, an observed value and whether it matches the
published behaviour of gpt-4o-mini. Nothing here is conclusive on its own;
treat a cluster of mismatches as a warning sign.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field


@dataclass
class Check:
    name: str
    detail: str
    passed: bool | None  # None = inconclusive
    evidence: str = ""


@dataclass
class Probe:
    base_url: str
    api_key: str
    model: str
    checks: list[Check] = field(default_factory=list)

    def _client(self):
        from openai import OpenAI

        return OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=60.0)

    # ------------------------------------------------------------ checks --

    def check_endpoint(self):
        try:
            client = self._client()
            started = time.monotonic()
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                max_tokens=5,
                temperature=0,
            )
            latency = time.monotonic() - started
            text = (resp.choices[0].message.content or "").strip()
            self.checks.append(
                Check(
                    "endpoint_reachable",
                    f"resp={text!r} latency={latency:.2f}s model_field={resp.model!r}",
                    True,
                    text,
                )
            )
        except Exception as exc:
            self.checks.append(
                Check("endpoint_reachable", "request failed", False, str(exc)[:300])
            )

    def check_model_field(self):
        """The response's own model field is weak evidence but cheap to read."""
        try:
            resp = self._client().chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=2,
                temperature=0,
            )
            reported = getattr(resp, "model", "") or ""
            looks_right = "gpt-4o-mini" in reported
            self.checks.append(
                Check(
                    "reported_model_name",
                    "a proxy can rewrite this field, so it is only a hint",
                    looks_right if reported else None,
                    f"model={reported!r}",
                )
            )
        except Exception as exc:
            self.checks.append(
                Check("reported_model_name", "request failed", None, str(exc)[:200])
            )

    def check_tokenizer_fingerprint(self):
        """Fingerprint the backend tokenizer through API usage accounting.

        An earlier version asked the model to *report* token counts, which tests
        nothing: models cannot reliably introspect their own tokenizer, so even
        the genuine model fails (observed on a relay that does appear to be the
        real thing). Prompt tokens in ``usage`` are computed by the backend
        tokenizer, so differencing two prompts through the API reveals the real
        per-string count.

        Reference counts for the probe strings (verified with tiktoken):
          o200k_base  (gpt-4o family): 2, 6, 51, 3
          cl100k_base (gpt-3.5/4):     2, 6, 51, 6
          p50k_base   (older):         2, 5, 51, 9
        The CJK probe is the decisive one: 3 means o200k, 6 means cl100k.
        """
        probes = {
            " hello world": 2,
            " antidisestablishmentarianism": 6,
            " " + "x " * 50: 51,
        }
        cjk_probe = " 你好世界"
        cjk_expected_o200k = 3
        cjk_expected_cl100k = 6

        try:
            client = self._client()

            def prompt_tokens(suffix: str) -> int:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": "Repeat back nothing." + suffix}
                    ],
                    max_tokens=1,
                    temperature=0,
                )
                return int(resp.usage.prompt_tokens)

            baseline = prompt_tokens("")
            counts = {}
            for text, expected in probes.items():
                counts[text] = prompt_tokens(text) - baseline
                self.checks.append(
                    Check(
                        f"tokenizer[{text.strip()[:16] or 'probe'}]",
                        f"backend tokenizer count; o200k_base expects {expected}",
                        counts[text] == expected,
                        f"got {counts[text]}",
                    )
                )

            cjk = prompt_tokens(cjk_probe) - baseline
            if cjk == cjk_expected_o200k:
                verdict, passed = (
                    f"{cjk} -> consistent with o200k_base (gpt-4o family)", True
                )
            elif cjk == cjk_expected_cl100k:
                verdict, passed = (
                    f"{cjk} -> consistent with cl100k_base (gpt-3.5/4 family), "
                    "NOT gpt-4o-mini",
                    False,
                )
            else:
                verdict, passed = f"{cjk} -> matches no known encoder", None
            self.checks.append(
                Check("tokenizer_family[cjk]", "decisive discriminator", passed, verdict)
            )
        except Exception as exc:
            self.checks.append(
                Check("tokenizer_fingerprint", "request failed", None, str(exc)[:200])
            )

    def check_json_mode(self):
        """Structured output is used by enrichment; we must know it works."""
        try:
            resp = self._client().chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": 'Return a JSON object with a single key "ok" set to true.',
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=30,
                temperature=0,
            )
            raw = resp.choices[0].message.content or ""
            parsed = json.loads(raw)
            self.checks.append(
                Check(
                    "json_object_mode",
                    "required for reliable enrichment parsing",
                    isinstance(parsed, dict) and parsed.get("ok") is True,
                    raw[:120],
                )
            )
        except Exception as exc:
            self.checks.append(
                Check("json_object_mode", "not supported or failed", False, str(exc)[:200])
            )

    def check_instruction_following(self):
        """Precise formatting compliance, which enrichment prompts rely on."""
        try:
            resp = self._client().chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Output exactly three lines. Line 1 is the word ALPHA. "
                            "Line 2 is the word BETA. Line 3 is the word GAMMA. "
                            "No other text."
                        ),
                    }
                ],
                max_tokens=30,
                temperature=0,
            )
            text = (resp.choices[0].message.content or "").strip()
            ok = text.split() == ["ALPHA", "BETA", "GAMMA"]
            self.checks.append(
                Check("instruction_following", "exact 3-line output", ok, repr(text[:80]))
            )
        except Exception as exc:
            self.checks.append(
                Check("instruction_following", "request failed", None, str(exc)[:200])
            )

    def check_latency_profile(self):
        """A very fast 'large model' is a smell; so is an extremely slow mini."""
        try:
            client = self._client()
            times = []
            for _ in range(3):
                started = time.monotonic()
                client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Count from 1 to 20."}],
                    max_tokens=60,
                    temperature=0,
                )
                times.append(time.monotonic() - started)
            median = sorted(times)[len(times) // 2]
            plausible = 0.2 <= median <= 20.0
            self.checks.append(
                Check(
                    "latency_profile",
                    f"median of 3 = {median:.2f}s; broad plausibility range only",
                    plausible,
                    ", ".join(f"{t:.2f}s" for t in times),
                )
            )
        except Exception as exc:
            self.checks.append(
                Check("latency_profile", "request failed", None, str(exc)[:200])
            )

    def run(self) -> int:
        self.check_endpoint()
        self.check_model_field()
        self.check_tokenizer_fingerprint()
        self.check_json_mode()
        self.check_instruction_following()
        self.check_latency_profile()

        print(f"\nprovider probe: base_url={self.base_url} model={self.model}\n")
        width = max(len(c.name) for c in self.checks) if self.checks else 10
        failures = 0
        for check in self.checks:
            mark = {True: "PASS", False: "FAIL", None: "?   "}[check.passed]
            if check.passed is False:
                failures += 1
            print(f"  [{mark}] {check.name.ljust(width)}  {check.detail}")
            if check.evidence:
                print(f"         evidence: {check.evidence}")

        print(
            "\nNote: a proxy can pass every check and still not be the real model;\n"
            "and a real model can fail an inconclusive check. Treat multiple FAILs\n"
            "as a warning, and confirm with the platform's smoke run before Full.\n"
        )
        return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CODEMEM_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CODEMEM_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    )
    parser.add_argument(
        "--model", default=os.environ.get("CODEMEM_LLM_MODEL", "gpt-4o-mini")
    )
    args = parser.parse_args(argv)

    if not args.base_url or not args.api_key:
        print(
            "error: base_url and api_key are required "
            "(set CODEMEM_LLM_BASE_URL / CODEMEM_LLM_API_KEY or pass flags)",
            file=sys.stderr,
        )
        return 2

    try:
        import openai  # noqa: F401
    except ImportError:
        print("error: install the llm extra first:  pip install -e '.[llm]'", file=sys.stderr)
        return 2

    return Probe(args.base_url, args.api_key, args.model).run()


if __name__ == "__main__":
    raise SystemExit(main())
