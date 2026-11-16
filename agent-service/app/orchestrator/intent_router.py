from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import get_logger
from app.orchestrator.orchestrator_runtime import RouteResult


logger = get_logger(__name__)


class IntentRouter:
    async def route(
        self,
        *,
        model: Any,
        message: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> RouteResult:
        if not message:
            logger.info("intent_route_empty_message")
            return RouteResult(intent="fallback", confidence=0.0, reason="用户信息为空")
        # 先尝试llm意图识别
        if model is not None:
            try:
                result = await self._route_with_llm(model, message, payload or {})
                logger.info(
                    "intent_route_llm_succeeded",
                    intent=result.intent,
                    confidence=result.confidence,
                    payload_keys=sorted((payload or {}).keys()),
                )
                return result
            except Exception:
                logger.exception("intent_route_llm_failed")
        # 如果失败，则走规则识别
        result = self._route_with_rules(message)
        logger.info(
            "intent_route_rules_used",
            intent=result.intent,
            confidence=result.confidence,
        )
        return result

    async def _route_with_llm(
        self,
        model: Any,
        message: str,
        payload: dict[str, Any],
    ) -> RouteResult:
        structured_model = model.with_structured_output(RouteResult, method="function_calling")
        response = await structured_model.ainvoke(
            [
                SystemMessage(content=self._system_prompt()),
                HumanMessage(
                    content=(
                        f"用户消息：{message}\n"
                        f"页面/业务上下文 payload：{payload}\n"
                        "请只输出目标业务域节点。"
                    )
                ),
            ]
        )
        return response if isinstance(response, RouteResult) else RouteResult.model_validate(response)

    def _route_with_rules(self, message: str) -> RouteResult:
        text = message.strip()
        lower = text.lower()
        if any(word in text for word in ("转人工", "人工客服", "投诉")):
            intent = "human_handoff"
        elif any(word in text for word in ("理赔", "报销", "赔付", "报案", "材料")):
            intent = "claim"
        elif any(word in text for word in ("投保", "健康告知", "核保", "保单", "下一步")):
            intent = "recommendation_plan"
        elif any(word in text for word in ("推荐", "方案", "组合", "配置", "怎么买", "买什么")):
            intent = "recommendation_plan"
        elif any(word in text for word in ("多少钱", "保费", "试算", "报价", "价格")):
            intent = "recommendation_plan"
        elif any(word in text for word in ("条款", "免责", "等待期", "责任", "保障", "犹豫期")):
            intent = "recommendation_plan"
        elif lower in {"hi", "hello"} or any(word in text for word in ("你好", "您好")):
            intent = "chitchat"
        else:
            intent = "fallback"
        return RouteResult(
            intent=intent,
            confidence=0.7 if intent != "fallback" else 0.2,
            reason="基于关键词判断"
        )

    def _system_prompt(self) -> str:
        return (
            "你是保险商城智能客服的业务域路由节点，只决定把用户消息交给哪个业务 Agent，"
            "不要回答用户问题，不要改写用户问题，不要判断业务域内部动作。"
            "intent 只能从 recommendation_plan、claim、human_handoff、chitchat、fallback 中选择。"
            "保险产品咨询、保险产品推荐、保险产品组合方案管理、投保咨询归 recommendation_plan。"
            "理赔责任、材料、流程、进度、报案归 claim。"
            "高风险、投诉、用户明确要求人工归 human_handoff。"
            "普通寒暄归 chitchat；无法判断归 fallback。"
        )
