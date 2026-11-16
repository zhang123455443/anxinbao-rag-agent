from __future__ import annotations

import json
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.core.logging import get_logger
from app.orchestrator.orchestrator_graph import ChatOrchestrator
from app.security.jwt import UserContext, get_current_user
from app.services.chat_threads import ChatThreadNotFoundError, ChatThreadRepository

router = APIRouter(tags=["chat"])
logger = get_logger(__name__)


class ChatRequest(BaseModel):
    message: str | None = None
    thread_id: UUID | None = Field(default=None, alias="thread_id")
    decision: Optional[dict[str, Any]] = None
    payload: dict[str, Any] = Field(default_factory=dict)


@router.post("/chat")
async def chat(
    http_request: Request,
    request: ChatRequest,
    authorization: tuple[UserContext, str] = Depends(get_current_user)
) -> EventSourceResponse:
    return await _handle_chat(http_request, request, authorization)

async def _handle_chat(
    http_request: Request,
    request: ChatRequest,
    authorization: tuple[UserContext, str]
) -> EventSourceResponse:
    # 获取token
    user_context, token = authorization
    # 必要信息校验
    if not request.thread_id:
        logger.warning("chat_request_missing_thread_id", user_id=user_context.user_id)
        raise HTTPException(status_code=422, detail="thread_id is required")
    if not request.message and not request.decision:
        logger.warning(
            "chat_request_missing_message_or_decision",
            thread_id=request.thread_id,
            user_id=user_context.user_id,
        )
        raise HTTPException(status_code=422, detail="message or decision is required")
    repository = _get_thread_repository(http_request)
    try:
        await repository.require_owned(request.thread_id, user_context.user_id)
    except ChatThreadNotFoundError as exc:
        raise HTTPException(status_code=404, detail="chat thread not found") from exc

    # 获取orchestrator
    orchestrator = _get_orchestrator(http_request)
    # 流式响应
    logger.info(
        "chat_stream_opened",
        thread_id=request.thread_id,
        user_id=user_context.user_id,
        message=request.message,
        decision=request.decision,
        payload_keys=sorted(request.payload.keys()),
    )
    return EventSourceResponse(
        _stream_chat(
            orchestrator=orchestrator,
            repository=repository,
            thread_id=request.thread_id,
            user_id=user_context.user_id,
            message=request.message,
            decision=request.decision,
            payload=request.payload,
            user_token=token,
        )
    )


async def _stream_chat(
    *,
    orchestrator: ChatOrchestrator,
    repository: ChatThreadRepository,
    thread_id: UUID,
    user_id: int,
    message: str | None,
    decision: dict[str, Any] | None,
    payload: dict[str, Any],
    user_token: str,
):
    try:
        async for event in orchestrator.astream(
            thread_id=str(thread_id),
            message=message,
            decision=decision,
            payload=payload,
            user_token=user_token,
        ):
            if "data" in event:
                event = {
                    **event,
                    "data": json.dumps(event["data"], ensure_ascii=False),
                }
            yield event
    finally:
        try:
            await repository.touch(thread_id, user_id)
        except Exception:
            logger.exception(
                "chat_thread_touch_failed",
                thread_id=str(thread_id),
                user_id=user_id,
            )


def _get_orchestrator(request: Request) -> ChatOrchestrator:
    orchestrator = getattr(request.app.state, "chat_orchestrator", None)
    if orchestrator is None:
        logger.error("chat_orchestrator_not_ready")
        raise HTTPException(status_code=500, detail="server is not ready")
    return orchestrator


def _get_thread_repository(request: Request) -> ChatThreadRepository:
    repository = getattr(request.app.state, "chat_thread_repository", None)
    if repository is None:
        logger.error("chat_thread_repository_not_ready")
        raise HTTPException(status_code=500, detail="server is not ready")
    return repository
