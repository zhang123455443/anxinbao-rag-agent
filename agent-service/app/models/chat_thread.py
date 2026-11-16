from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


def _strip_title(title: str) -> str:
    normalized = title.strip()
    if not normalized:
        raise ValueError("title must not be blank")
    return normalized


class ChatThreadCreate(BaseModel):
    title: str = Field(default="新会话", min_length=1, max_length=200)

    _normalize_title = field_validator("title")(_strip_title)


class ChatThreadUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)

    _normalize_title = field_validator("title")(_strip_title)


class ChatThreadResponse(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    additional_info: Any | None = None


class ChatHistoryResponse(BaseModel):
    thread_id: UUID
    messages: list[ChatHistoryMessage]
    interrupt: dict[str, Any] | None = None
    additional_info: Any | None = None
