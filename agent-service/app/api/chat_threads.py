from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.core.logging import get_logger
from app.models.chat_thread import (
    ChatHistoryResponse,
    ChatThreadCreate,
    ChatThreadResponse,
    ChatThreadUpdate,
)
from app.orchestrator.orchestrator_graph import ChatOrchestrator
from app.security.jwt import UserContext, get_current_user
from app.services.chat_threads import (
    ChatThreadNotFoundError,
    ChatThreadRepository,
    project_history,
    project_history_from_snapshots,
)


router = APIRouter(prefix="/chat-threads", tags=["chat-threads"])
logger = get_logger(__name__)


@router.post("", response_model=ChatThreadResponse, status_code=status.HTTP_201_CREATED)
async def create_chat_thread(
    body: ChatThreadCreate,
    request: Request,
    authorization: tuple[UserContext, str] = Depends(get_current_user),
) -> ChatThreadResponse:
    user, _ = authorization
    thread = await _repository(request).create(user.user_id, body.title)
    logger.info("chat_thread_created", thread_id=str(thread.id), user_id=user.user_id)
    return thread


@router.get("", response_model=list[ChatThreadResponse])
async def list_chat_threads(
    request: Request,
    authorization: tuple[UserContext, str] = Depends(get_current_user),
) -> list[ChatThreadResponse]:
    user, _ = authorization
    return await _repository(request).list_by_user(user.user_id)


@router.patch("/{thread_id}", response_model=ChatThreadResponse)
async def rename_chat_thread(
    thread_id: UUID,
    body: ChatThreadUpdate,
    request: Request,
    authorization: tuple[UserContext, str] = Depends(get_current_user),
) -> ChatThreadResponse:
    user, _ = authorization
    try:
        thread = await _repository(request).rename(thread_id, user.user_id, body.title)
    except ChatThreadNotFoundError as exc:
        raise _not_found() from exc
    logger.info("chat_thread_renamed", thread_id=str(thread_id), user_id=user.user_id)
    return thread


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_thread(
    thread_id: UUID,
    request: Request,
    authorization: tuple[UserContext, str] = Depends(get_current_user),
) -> Response:
    user, _ = authorization
    repository = _repository(request)
    try:
        await repository.require_owned(thread_id, user.user_id)
        await _orchestrator(request).delete_thread_state(str(thread_id))
        await repository.delete(thread_id, user.user_id)
    except ChatThreadNotFoundError as exc:
        raise _not_found() from exc
    logger.info("chat_thread_deleted", thread_id=str(thread_id), user_id=user.user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{thread_id}/messages", response_model=ChatHistoryResponse)
async def get_chat_history(
    thread_id: UUID,
    request: Request,
    authorization: tuple[UserContext, str] = Depends(get_current_user),
) -> ChatHistoryResponse:
    user, _ = authorization
    try:
        await _repository(request).require_owned(thread_id, user.user_id)
    except ChatThreadNotFoundError as exc:
        raise _not_found() from exc
    state = await _orchestrator(request).get_thread_state(str(thread_id))
    snapshots = await _orchestrator(request).get_completed_thread_states(str(thread_id))

    if snapshots and state.interrupts:
        snapshots.append(state)
    messages = (
        project_history_from_snapshots(snapshots)
        if snapshots
        else project_history(list(state.values.get("messages", [])))
    )
    return ChatHistoryResponse(
        thread_id=thread_id,
        messages=messages,
        interrupt=_project_interrupt(state.interrupts),
    )


def _repository(request: Request) -> ChatThreadRepository:
    repository = getattr(request.app.state, "chat_thread_repository", None)
    if repository is None:
        raise HTTPException(status_code=500, detail="server is not ready")
    return repository


def _orchestrator(request: Request) -> ChatOrchestrator:
    orchestrator = getattr(request.app.state, "chat_orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="server is not ready")
    return orchestrator


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="chat thread not found")


def _project_interrupt(interrupts) -> dict | None:
    if not interrupts:
        return None
    value = interrupts[0].value
    if isinstance(value, dict):
        return value
    return {"value": value}
