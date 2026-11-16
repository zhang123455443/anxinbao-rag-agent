from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NotRequired

from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

from app.clients.business_service import BusinessServiceClient
from app.models import LLMProvider

Intent = Literal[
    "chitchat",
    "recommendation_plan",
    "claim",
    "human_handoff",
    "fallback",
]

# 意图识别结果
class RouteResult(BaseModel):
    intent: Intent = Field(description="The best target intent for the current user turn.")
    confidence: float = Field(ge=0, le=1, description="Routing confidence from 0 to 1.")
    reason: str = Field(description="判断用户意图的理由.")

# 父图state
class ParentState(MessagesState):
    active_workflow: NotRequired[str | None]  # 当前激活的工作流
    route_info: NotRequired[RouteResult | None]  # 路由信息
    additional_info: NotRequired[dict | None] # 最终答案中的附加信息，没有就给None

# 上下文信息
@dataclass
class ChatRuntimeContext:
    user_message: str | None
    payload: dict[str, Any] | None
    user_token: str | None
    llms: LLMProvider
    business_client: BusinessServiceClient
    search_tool: Any

# 异常信息
class UnsupportedIntentError(ValueError):
    pass


class InvalidIntentPayloadError(ValueError):
    pass
