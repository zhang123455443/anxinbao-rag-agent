import json
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from app.orchestrator.orchestrator_runtime import ChatRuntimeContext


def state_command(
    runtime: ToolRuntime[ChatRuntimeContext],
    update: dict[str, Any],
    content: Any,
) -> Command:
    """更新 Workflow State，并返回当前 Tool Call 对应的 ToolMessage。"""

    return Command(
        update={
            **update,
            "messages": [
                ToolMessage(
                    content=json_text(content),
                    tool_call_id=runtime.tool_call_id or "unknown",
                )
            ],
        }
    )


def get_user_token(runtime: ToolRuntime[ChatRuntimeContext]) -> str | None:
    """从 Runtime Context 获取当前用户令牌。"""

    return runtime.context.user_token


def json_text(value: Any) -> str:
    """把 Tool 返回值编码为保留中文的 JSON 文本。"""

    return json.dumps(value, ensure_ascii=False, default=str)
