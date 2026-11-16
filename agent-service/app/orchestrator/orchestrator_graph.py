from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.stream import CustomTransformer
from langgraph.types import Command, Checkpointer, StateSnapshot
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime

from app.clients.business_service import BusinessServiceClient
from app.core.logging import get_logger
from app.models import LLMProvider
from app.rag import SearchInsuranceKnowledgeTool
from app.orchestrator.intent_router import IntentRouter
from app.orchestrator.orchestrator_runtime import (
    ParentState, ChatRuntimeContext, RouteResult, Intent
)
from app.workflows import Workflow

logger = get_logger(__name__)


def _thread_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}

def _serialize(obj):
    """递归转换对象为可 JSON 序列化的格式"""
    if hasattr(obj, 'value'):
        return _serialize(obj.value)
    elif hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif isinstance(obj, (list, tuple)):
        return [_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj

class ChatOrchestrator:
    def __init__(
        self,
        *,
        intent_router: IntentRouter,
        llms: LLMProvider,
        business_client: BusinessServiceClient,
        search_tool: SearchInsuranceKnowledgeTool,
        workflows: list[Workflow],
        checkpointer: Checkpointer | None = None,
    ) -> None:
        self._checkpointer = checkpointer or InMemorySaver()
        self._intent_router = intent_router
        self._llms = llms
        self._business_client = business_client
        self._search_tool = search_tool
        self._graph = None
        self._workflows = workflows

    def init_graph(self):
        self._graph = self._build_parent_graph()
        logger.info(
            "chat_orchestrator_graph_initialized",
            workflows=[workflow.workflow_name() for workflow in self._workflows],
        )

    async def get_thread_messages(self, thread_id: str) -> list[BaseMessage]:
        """从 LangGraph 最新 State 中读取完整消息，供接口层安全投影。"""

        state = await self.get_thread_state(thread_id)
        return list(state.values.get("messages", []))

    async def get_thread_state(self, thread_id: str) -> StateSnapshot:
        """读取 Thread 的 LangGraph 最新 StateSnapshot。"""

        if self._graph is None:
            raise RuntimeError("chat orchestrator graph is not initialized")
        return await self._graph.aget_state(_thread_config(thread_id))

    async def get_completed_thread_states(self, thread_id: str) -> list[StateSnapshot]:
        """读取 Thread 的完成态 StateSnapshot，按时间从旧到新返回。"""

        if self._graph is None:
            raise RuntimeError("chat orchestrator graph is not initialized")
        snapshots = [
            snapshot
            async for snapshot in self._graph.aget_state_history(_thread_config(thread_id))
            if snapshot.next == ()
        ]
        return list(reversed(snapshots))

    async def delete_thread_state(self, thread_id: str) -> None:
        """删除 Checkpointer 中属于一个 Thread 的全部状态。"""

        await self._checkpointer.adelete_thread(thread_id)

    def _build_parent_graph(self):
        graph = StateGraph(ParentState, context_schema=ChatRuntimeContext)
        graph.add_node("route", self._route_node)
        # 添加WorkFlow节点
        for workflow in self._workflows:
            graph.add_node(workflow.workflow_name(), workflow.build_graph())
        # 添加容错和人工节点
        graph.add_node("fallback", _fallback_node)
        graph.add_node("human_handoff", _human_handoff)

        graph.set_entry_point("route")
        for workflow in self._workflows:
            graph.add_edge(workflow.workflow_name(), END)
        graph.add_edge("fallback", END)
        graph.add_edge("human_handoff", END)
        return graph.compile(checkpointer=self._checkpointer)

    async def _route_node(
        self,
        state: ParentState,
        runtime: Runtime[ChatRuntimeContext]
    ) -> Intent:
        # 需要识别意图。先获取message、payload
        message = runtime.context.user_message
        route_payload = dict(runtime.context.payload or {})
        # 当前 HumanMessage 已进入 State，因此只在它之前查找上一条最终 AI 回复。
        previous_response = _previous_ai_response(state["messages"])
        if previous_response:
            route_payload["conversation_history"] = [
                {"role": "assistant", "content": previous_response}
            ]
        # 根据消息识别意图
        route_result: RouteResult = await self._intent_router.route(
            model=runtime.context.llms.get("chat"),
            message=message,
            payload=route_payload,
        )
        # 更新state并跳转
        intent = route_result.intent
        logger.info(
            "chat_route_decided",
            intent=intent,
            confidence=route_result.confidence,
            reason=route_result.reason,
            payload_keys=sorted(route_payload.keys()),
        )
        return Command(
            goto=intent,
            update={
                "active_workflow": intent,
                "route_info": route_result
            }
        )

    async def astream(
        self,
        *,
        thread_id: str,
        message: str | None,
        decision: dict[str, Any] | None = None,
        payload: dict | None,
        user_token: str | None,
    ) -> AsyncIterator[dict[str, Any]]:
        config = _thread_config(thread_id)
        # Graph input ， 判断是用户消息还是interrupt
        if message:
            user_input = {"messages": [HumanMessage(content=message)], "additional_info": None} # 要重置additional_info
        elif decision:
            user_input = Command(resume=decision, update={"additional_info": None}) # 要重置additional_info
        # 传递Context
        context = ChatRuntimeContext(
            user_message=message,
            payload=payload,
            user_token=user_token,
            llms=self._llms,
            business_client=self._business_client,
            search_tool=self._search_tool,
        )
        # 调用Graph
        logger.info(
            "chat_graph_stream_started",
            thread_id=thread_id,
            has_message=message is not None,
            has_decision=decision is not None,
            payload_keys=sorted((payload or {}).keys()),
        )
        stream = await self._graph.astream_events(
            user_input, config=config, context=context, version="v3", transformers=[CustomTransformer]
        )
        # 流式处理消息
        try:
            additional_info = None
            async for event in stream:
                method = event['method']
                # 判断事件类型，可以是messages、interrupts等等
                if method == 'messages':
                    data = event['params']['data'][0]
                    if isinstance(data, AIMessage):
                        for chunk in data.content:
                            yield {
                                "event": "message",
                                "data": chunk
                            }
                    elif data.get('delta') and data['delta'].get('text'):
                        yield {
                            "event": "message",
                            "data": data['delta'].get('text', ""),
                        }
                if method == 'custom':
                    data = event["params"]["data"]
                    logger.info(custom_data=data)
                    if isinstance(data, dict) and data.get("type") == "additional_info":
                        additional_info = data.get("data")

            if additional_info:
                yield {
                    "event": "additional_info",
                    "data": additional_info,
                }

            if await stream.interrupted():
                interrupts = await stream.interrupts()
                logger.info("chat_graph_interrupted", thread_id=thread_id)
                yield {
                    "event": "interrupt",
                    "data": _serialize(interrupts),
                }
        except Exception:
            logger.exception("chat_graph_stream_failed", thread_id=thread_id)
            raise

        yield {
            "event": "done",
            "data": {
                "thread_id": thread_id,
                "finish_reason": "stop",
            }
        }
        logger.info("chat_graph_stream_completed", thread_id=thread_id)


def _fallback_node(state: ParentState) -> ParentState:
    answer = "我还不能确定你的需求。你可以说明是想咨询产品条款、做保险方案、试算保费，还是需要投保流程帮助。"
    return {
        "active_workflow": "unknown",
        "messages": [AIMessage(content=answer)],
    }


def _human_handoff(state: ParentState) -> ParentState:
    return {
        "active_workflow": "_human_handoff",
        "messages": [AIMessage(content="已记录转人工请求，后续可接入人工客服队列或工单系统。")],
    }


def _previous_ai_response(messages: list[BaseMessage]) -> str | None:
    """返回本轮用户消息之前最近一条可展示的 AI 文本。"""

    previous_messages = messages[:-1] if messages and isinstance(messages[-1], HumanMessage) else messages
    for item in reversed(previous_messages):
        if not isinstance(item, AIMessage):
            continue
        if item.tool_calls or item.additional_kwargs.get("tool_calls"):
            continue
        content = _message_text(item.content)
        if content:
            return content
    return None


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content).strip()

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {None, "text"}:
            parts.append(str(block.get("text", "")))
    return "".join(parts).strip()
