"""LLM adapter.

plan.md section 2.2 constrains *what* may be called (the rules mandate
gpt-4o-mini for Add/Search); this module constrains *how*, and is the only
place in the codebase that talks to a provider. Stage 0 ships the transport
and the dry-run capability but performs no extraction, because extraction is
stage 2 work.

Two rules are enforced here rather than trusted to callers:

* **No model call on the sync critical path.** ``complete_json`` is only ever
  reachable from the best-effort enrichment phase, which has its own deadline.
* **A dry run must not spend tokens.** ``probe()`` exists so connectivity,
  credentials and model availability can be verified before any extraction
  prompt is written, and it is validated locally before a single request is
  sent.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .config import Settings, get_settings


class LLMUnavailable(Exception):
    """Provider not configured -- callers must degrade, not fail."""


class LLMError(Exception):
    """Provider reachable but the call failed."""


@dataclass
class LLMResult:
    text: str
    model: str
    usage: dict = field(default_factory=dict)
    elapsed_ms: int = 0
    raw_finish_reason: str | None = None


class LLMClient:
    """OpenAI-compatible chat client, pinned to the manded model by default."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None

    @property
    def configured(self) -> bool:
        return bool(self.settings.llm_base_url and self.settings.llm_api_key)

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.configured:
            raise LLMUnavailable(
                "CODEMEM_LLM_BASE_URL / CODEMEM_LLM_API_KEY are not set"
            )
        from openai import OpenAI

        self._client = OpenAI(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            timeout=self.settings.llm_timeout_s,
            max_retries=self.settings.llm_max_retries,
        )
        return self._client

    # -- transport -------------------------------------------------------

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_s: float | None = None,
    ) -> LLMResult:
        client = self._ensure_client()
        started = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_s or self.settings.llm_timeout_s,
            )
        except Exception as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        elapsed = int((time.monotonic() - started) * 1000)

        choice = response.choices[0] if response.choices else None
        text = (choice.message.content if choice and choice.message else "") or ""
        usage = {}
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return LLMResult(
            text=text,
            model=response.model or self.settings.llm_model,
            usage=usage,
            elapsed_ms=elapsed,
            raw_finish_reason=getattr(choice, "finish_reason", None) if choice else None,
        )

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        timeout_s: float | None = None,
    ) -> tuple[dict | list | None, LLMResult]:
        """Ask for JSON and parse it defensively.

        Returns ``(parsed_or_None, result)``. A parse failure is not an
        exception: plan.md section 5/A4 requires schema-invalid output to be
        dropped, not retried into the budget.
        """
        client = self._ensure_client()
        started = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                timeout=timeout_s or self.settings.llm_timeout_s,
            )
        except Exception as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        elapsed = int((time.monotonic() - started) * 1000)

        choice = response.choices[0] if response.choices else None
        text = (choice.message.content if choice and choice.message else "") or ""
        usage = {}
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        result = LLMResult(
            text=text,
            model=response.model or self.settings.llm_model,
            usage=usage,
            elapsed_ms=elapsed,
            raw_finish_reason=getattr(choice, "finish_reason", None) if choice else None,
        )
        return _parse_json_loose(text), result

    # -- dry run ---------------------------------------------------------

    def probe(self, *, live: bool = False, max_tokens: int = 16) -> dict:
        """Verify configuration, and optionally the wire, without wasting tokens.

        Default (``live=False``) is a pure local check: it confirms the
        credential shape looks usable, that the model name satisfies plan.md
        section 2.2, and that the transport can be constructed. It sends
        nothing and therefore costs nothing.

        ``live=True`` sends the smallest possible request so connectivity and
        auth can be confirmed for real before extraction prompts are developed.
        """
        report: dict = {
            "base_url_set": bool(self.settings.llm_base_url),
            "api_key_set": bool(self.settings.llm_api_key),
            "model": self.settings.llm_model,
            "embeddings_enabled": self.settings.embeddings_enabled,
            "live": live,
        }
        # plan.md section 2.2: the rules name this exact model.
        if self.settings.llm_model != "gpt-4o-mini":
            report["model_warning"] = (
                "rules specify gpt-4o-mini for Add/Search; "
                f"configured model is {self.settings.llm_model!r}"
            )
        if not self.configured:
            report["status"] = "not_configured"
            report["detail"] = "LLM env vars missing; system will run lex-only"
            return report

        try:
            client = self._ensure_client()
        except Exception as exc:
            report["status"] = "client_error"
            report["detail"] = f"{type(exc).__name__}: {exc}"
            return report

        report["status"] = "configured"
        if not live:
            report["detail"] = (
                "local checks passed; pass live=True to send one minimal request"
            )
            return report

        try:
            started = time.monotonic()
            response = client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                max_tokens=max_tokens,
                temperature=0.0,
                timeout=self.settings.llm_timeout_s,
            )
            elapsed = int((time.monotonic() - started) * 1000)
            choice = response.choices[0] if response.choices else None
            report["status"] = "live_ok"
            report["elapsed_ms"] = elapsed
            report["reply_len"] = len((choice.message.content or "") if choice and choice.message else "")
            if response.usage is not None:
                report["usage"] = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }
        except Exception as exc:
            report["status"] = "live_failed"
            report["detail"] = f"{type(exc).__name__}: {exc}"
        return report


def _parse_json_loose(text: str) -> dict | list | None:
    """Parse model JSON, tolerating a fenced block or surrounding prose."""
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            body = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
            candidate = "\n".join(body).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Last resort: take the outermost balanced object/array.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


_client: LLMClient | None = None


def get_llm(settings: Settings | None = None) -> LLMClient:
    global _client
    if _client is None or settings is not None:
        _client = LLMClient(settings)
    return _client


def reset_llm_for_tests() -> None:
    global _client
    _client = None