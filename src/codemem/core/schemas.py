"""Wire schemas for the Add / Search contract.

The contract is fixed by the platform (see api_guide.md). These models are the
single source of truth for validation; the response models are strict so that a
2xx can never carry a malformed body.

Track scope: the Coding track sends string ``content``. We nevertheless accept
an ordered ContentPart-like list defensively and flatten its text parts, so an
unexpected payload shape degrades instead of failing contract validation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Role = Literal["user", "assistant"]


def _flatten_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
        return "\n".join(p for p in parts if p)
    raise ValueError("content must be a string for the Coding track")


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Role
    content: str = Field(min_length=1)
    timestamp: int | None = None

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_content(cls, value: Any) -> Any:
        return _flatten_content(value)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _coerce_timestamp(cls, value: Any) -> Any:
        if value is None or isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value))
            except ValueError:
                return None
        return None


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class AddResponse(BaseModel):
    """Echoed verbatim; ``success`` is only true once memory is searchable."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(min_length=1)
    options: list[str] | None = None
    user_id: str = Field(min_length=1)
    top_k: int = Field(ge=1)

    @field_validator("options", mode="before")
    @classmethod
    def _coerce_options(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, list):
            return [str(v) for v in value]
        return None


class SearchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    score: float | None = None
    created_at: str | None = None


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: list[SearchItem]


class HealthResponse(BaseModel):
    status: str
    version: str
    users: int
    memories: int
