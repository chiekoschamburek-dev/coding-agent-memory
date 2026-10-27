"""Cross-encoder rerank layer.

The reranker reorders the selected candidates *before* the final response cut,
so a document ranked just outside the response can still be recovered.  It is
optional: any configuration/transport error returns ``None`` and the retrieval
order stands.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Protocol, Sequence


class Reranker(Protocol):
    @property
    def available(self) -> bool: ...

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        deadline=None,
    ) -> list[float] | None: ...


class NullReranker:
    available = False

    def rerank(self, query: str, documents: Sequence[str], deadline=None):
        return None


class HttpReranker:
    """Client for an external rerank API.

    Two dialects are supported, selected by the base URL:

    * ``https://dashscope.aliyuncs.com/...`` — the Alibaba DashScope native
      rerank endpoint (``POST /api/v1/services/rerank/text-rerank/text-rerank``,
      body ``input.{query,documents}``, response ``output.results`` with
      ``relevance_score``; model ``qwen3-rerank``).
    * anything else — the Cohere-style ``POST {base}/rerank`` (body
      ``{query, documents}``, response ``results`` with ``score``), as served
      by ``scripts/serve_rerank.py``.

    Both score shapes are accepted on either path and normalised to [0, 1].
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 15.0,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.api_key = api_key
        self.model = model
        self.dashscope = "dashscope.aliyuncs.com" in self.base_url

    @property
    def available(self) -> bool:
        return bool(self.base_url)

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        deadline=None,
    ) -> list[float] | None:
        if not self.available or not documents:
            return None
        timeout = self.timeout_s
        if deadline is not None:
            try:
                timeout = max(0.5, min(timeout, deadline.time_left - 0.3))
            except Exception:
                pass
        docs = [str(doc)[:4000] for doc in documents]
        if self.dashscope:
            url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
            body: dict = {
                "model": self.model or "qwen3-rerank",
                "input": {"query": query, "documents": docs},
                "parameters": {"return_documents": False, "top_n": len(docs)},
            }
        else:
            url = self.base_url + "/rerank"
            body = {"query": query, "documents": docs}
            if self.model:
                body["model"] = self.model
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return None
        results = data.get("output", {}).get("results") if isinstance(data, dict) else None
        if results is None and isinstance(data, dict):
            results = data.get("results")
        if not isinstance(results, list):
            return None
        scores = [0.0] * len(documents)
        seen = 0
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item["index"])
                raw_score = item.get("relevance_score", item.get("score"))
                score = float(raw_score)
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= index < len(scores):
                scores[index] = max(0.0, min(1.0, score))
                seen += 1
        return scores if seen == len(documents) else None
