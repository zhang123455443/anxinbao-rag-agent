from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import asyncpg
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.types import StateSnapshot

from app.models.chat_thread import ChatHistoryMessage, ChatThreadResponse


class ChatThreadNotFoundError(LookupError):
    pass


class ChatThreadRepository:
    """管理会话元数据，不重复保存 LangGraph 消息。"""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def create(self, user_id: int, title: str) -> ChatThreadResponse:
        thread_id = uuid4()
        normalized_title = _normalize_title(title)
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                INSERT INTO chat_threads (id, user_id, title)
                VALUES ($1, $2, $3)
                RETURNING id, title, created_at, updated_at
                """,
                thread_id,
                user_id,
                normalized_title,
            )
        return _thread_from_row(row)

    async def list_by_user(self, user_id: int) -> list[ChatThreadResponse]:
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT id, title, created_at, updated_at
                FROM chat_threads
                WHERE user_id = $1
                ORDER BY updated_at DESC, created_at DESC
                """,
                user_id,
            )
        return [_thread_from_row(row) for row in rows]

    async def require_owned(self, thread_id: UUID, user_id: int) -> ChatThreadResponse:
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT id, title, created_at, updated_at
                FROM chat_threads
                WHERE id = $1 AND user_id = $2
                """,
                thread_id,
                user_id,
            )
        if row is None:
            raise ChatThreadNotFoundError(str(thread_id))
        return _thread_from_row(row)

    async def rename(
        self,
        thread_id: UUID,
        user_id: int,
        title: str,
    ) -> ChatThreadResponse:
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                UPDATE chat_threads
                SET title = $3, updated_at = CURRENT_TIMESTAMP
                WHERE id = $1 AND user_id = $2
                RETURNING id, title, created_at, updated_at
                """,
                thread_id,
                user_id,
                _normalize_title(title),
            )
        if row is None:
            raise ChatThreadNotFoundError(str(thread_id))
        return _thread_from_row(row)

    async def delete(self, thread_id: UUID, user_id: int) -> None:
        async with self.pool.acquire() as connection:
            deleted = await connection.fetchval(
                """
                DELETE FROM chat_threads
                WHERE id = $1 AND user_id = $2
                RETURNING id
                """,
                thread_id,
                user_id,
            )
        if deleted is None:
            raise ChatThreadNotFoundError(str(thread_id))

    async def touch(self, thread_id: UUID, user_id: int) -> None:
        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                UPDATE chat_threads
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = $1 AND user_id = $2
                """,
                thread_id,
                user_id,
            )


def project_history(messages: list[BaseMessage]) -> list[ChatHistoryMessage]:
    """把完整 LangGraph 消息投影为前端可展示的用户和最终助手文本。"""

    projected: list[ChatHistoryMessage] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            if message.tool_calls or message.additional_kwargs.get("tool_calls"):
                continue
            role = "assistant"
        else:
            continue

        content = _message_text(message.content)
        if content:
            projected.append(ChatHistoryMessage(role=role, content=content))
    return projected


def project_history_from_snapshots(snapshots: list[StateSnapshot]) -> list[ChatHistoryMessage]:
    """从 LangGraph 完成态快照恢复可展示历史，并把每轮 additional_info 绑定到当轮 AI 回复。"""

    projected: list[ChatHistoryMessage] = []
    previous_visible_count = 0
    for snapshot in snapshots:
        values = snapshot.values if isinstance(snapshot.values, dict) else {}
        messages = _visible_messages(list(values.get("messages", [])))
        new_messages = messages[previous_visible_count:]
        additional_info = values.get("additional_info")
        last_assistant_index = _last_assistant_index(new_messages)
        for index, message in enumerate(new_messages):
            if (
                additional_info is not None
                and last_assistant_index is not None
                and index == last_assistant_index
                and message.role == "assistant"
            ):
                projected.append(message.model_copy(update={"additional_info": additional_info}))
            else:
                projected.append(message)
        previous_visible_count = len(messages)
    return projected


def _visible_messages(messages: list[BaseMessage]) -> list[ChatHistoryMessage]:
    return project_history(messages)


def _last_assistant_index(messages: list[ChatHistoryMessage]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "assistant":
            return index
    return None


def _thread_from_row(row: asyncpg.Record) -> ChatThreadResponse:
    return ChatThreadResponse(
        id=row["id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _normalize_title(title: str) -> str:
    normalized = title.strip()
    if not normalized:
        raise ValueError("title must not be blank")
    return normalized


def _message_text(content: str | list[str | dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {None, "text"}:
            parts.append(str(block.get("text", "")))
    return "".join(parts).strip()
