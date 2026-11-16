import random

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from app.orchestrator.orchestrator_runtime import ParentState, ChatRuntimeContext
from .base import Workflow

chitchat_responses = {
    "greeting": [
        "你好！欢迎来到《保险精品汇》，我是你的私人保险顾问 😊。",
        "嗨！今天过得怎么样？",
    ],
    "weather": [
        "今天天气不错呢！不过我是保险顾问，具体天气你可以看看天气APP哦~",
        "天气这个话题很有趣！不过我更擅长回答保险问题呢~"
    ],
    "feeling": [
        "听起来很有趣呢！能再详细说说吗？",
        "原来如此，我理解了。不过关于保险问题我会更专业哦。"
    ]
}

class ChitchatState(ParentState):
    pass

class ChitchatWorkflow(Workflow):
    model_name = "local"

    def workflow_name(self):
        return "chitchat"

    async def _handle_chitchat(
        self,
        state: ChitchatState,
        runtime: Runtime[ChatRuntimeContext],
    ) -> dict:
        """处理闲聊"""
        user_message = runtime.context.user_message
        # 策略1: 友好回应但引导回业务
        if self._is_simple_chitchat(user_message):
            return self._get_friendly_response(user_message)

        # 策略2: 使用轻量级模型回复（不消耗主模型资源）
        else :
            return await self._lightweight_chitchat(user_message, runtime)

    def _is_simple_chitchat(self, text: str) -> bool:
        """判断是否简单闲聊"""
        simple_patterns = ["你好", "嗨", "早", "晚安", "天气", "吃饭"]
        return any(pattern in text for pattern in simple_patterns)

    def _get_friendly_response(self, text: str) -> str:
        """获取友好回应"""
        message = ""
        if "你好" in text or "嗨" in text:
            message = random.choice(chitchat_responses["greeting"])
        elif "天气" in text:
            message = random.choice(chitchat_responses["weather"])
        else:
            message = "嗯嗯，我在听~"
        message += "\n\n有什么保险问题我可以帮你解答吗？"
        return {"messages": [AIMessage(content=message)]}

    async def _lightweight_chitchat(
        self,
        user_message: str,
        runtime: Runtime[ChatRuntimeContext],
    ) -> dict:
        """使用轻量级模型回复"""
        model = runtime.context.llms.get(self.model_name)
        response = await model.ainvoke([
            SystemMessage(content="你是《保险精品汇》的保险顾问，友好简短回复用户的闲聊，然后引导回业务咨询"),
            HumanMessage(content=user_message)
        ])
        return {"messages": [response]}

    def build_graph(self):
        builder = StateGraph(ParentState)
        builder.add_node("handle_chitchat", self._handle_chitchat)
        builder.set_entry_point("handle_chitchat")

        return builder.compile()
