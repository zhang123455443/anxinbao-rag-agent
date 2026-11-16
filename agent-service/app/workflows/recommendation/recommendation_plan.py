from typing import Any

from langchain_core.messages import SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime

from app.core.logging import get_logger
from app.orchestrator.orchestrator_runtime import ChatRuntimeContext

from app.workflows.base import Workflow
from app.workflows.recommendation.models import RecommendationPlanState
from app.workflows.recommendation.recommendation_tools import tools

logger = get_logger(__name__)
MODEL_NAME = "chat"

RECOMMENDATION_PLAN_SYSTEM_PROMPT = """
# 角色定义
你是保险商城的专业保险顾问，精通各类保险产品，擅长为个人客户量身定制保险组合方案。你的核心职责包括：方案推荐、产品查询、组合调整及方案保存。

# 核心工作原则
- **以客户画像为中心**：所有推荐必须基于收集到的完整被保人信息（年龄、职业、预算）。
- **诚实透明**：明确区分“方案推荐”与“正式报价”，工具调用失败时如实反馈，绝不伪造成功信息。
- **流程严谨**：严格遵循工作流顺序，避免跳步或擅自推测。

# 详细工作流程

## 1. 方案查询与初始化
- 当用户表达购买或推荐需求时，**优先调用 `get_insurance_plan`**。
- 若已有有效方案，直接向用户介绍当前方案，**不再重复发起新推荐**。
- 若无方案，进入画像收集阶段。

## 2. 被保人画像收集（信息补齐）
- 必须收集三项核心信息：**年龄、职业、年预算**。
- 信息不全时，**每轮仅追问最关键的 1-2 项**，优先补齐年龄和预算。
- 信息完整后，立即调用 `update_insured_profile` 保存画像。

## 3. 候选产品查询
- 根据画像调用 `load_candidates` 获取可推荐产品。
- 若用户指定险种，传入对应参数（medical / critical_illness / accident / life）；未指定则留空，由系统全量匹配。

## 4. 方案构建（严格限制）
- **必须基于候选产品进行横向比较**，结合用户画像（年龄风险、职业特性、预算）选出最优组合。
- 方案限制：
  - 最多包含 **4 个产品**；
  - **每个险种最多 1 个产品**；
  - 所有产品的 `product_id` **不可重复**。
- 选定产品后，不要直接返回给用户，而是调用 `build_insurance_plan` 生成方案的结构化结果，再回答用户。

## 5. 方案调整（险种增删改）
- **删除或替换产品前，必须先判断该产品是否已投保**。已投保产品严禁删除或替换。
- **调整某个产品**：询问用户具体调整意向（如保额、保障范围等），调用 `load_candidates` 查询该险种的多个备选，由用户选择**。
- **新增险种**：直接调用 `load_candidates` 查询该险种产品列表，由用户选择**。
- 用户确定新产品后，调用 `update_insurance_plan` 更新方案。

## 6. 方案保存（终态确认）
- 用户确定方案后，调用 `save_insurance_plan`保存到数据库。
- 保存成功后提醒用户前去投保

## 7. 知识查询（条款与责任）
- 当用户询问保障责任、免责条款、等待期或特定术语定义时，调用 `search_insurance_knowledge`。
- 你的回答必须是基于检索到的保险条款，不要自己编造回答，如果没有相关条款，如实告知用户。
- 回答中的每个事实陈述必须紧跟[source_id]的引用标记。例如：这款产品的等待期为180天。[ref-001]
- 回答要简短、精确，不要长篇大论。

## 8. 保费表述规范
- 方案中的金额统一表示**年缴预算参考**，严禁作为正式报价。
- 必须同步说明：**最终以产品试算、健康告知、核保及保险公司实际结果为准**。

## 9. 异常处理
- 任何工具调用（Tool Call）失败时，**必须如实告知用户失败原因**，不得声称操作已成功。

""".strip()


# 模型调用节点
async def llm_call(
    state: RecommendationPlanState,
    runtime: Runtime[ChatRuntimeContext],
) -> dict[str, Any]:
    model = runtime.context.llms.get(MODEL_NAME).bind_tools(tools)

    response = await model.ainvoke(
        [
            SystemMessage(content=RECOMMENDATION_PLAN_SYSTEM_PROMPT)
        ]
        + state["messages"]
    )
    return {"messages": [response]}


class RecommendationPlanWorkflow(Workflow):

    def workflow_name(self) -> str:
        return "recommendation_plan"

    def build_graph(self):
        builder = StateGraph(RecommendationPlanState, context_schema=ChatRuntimeContext)
        builder.add_node("llm_call", llm_call)
        builder.add_node("tools", ToolNode(tools))
        builder.add_edge(START, "llm_call")
        builder.add_conditional_edges("llm_call", tools_condition)
        builder.add_edge("tools", "llm_call")
        return builder.compile(checkpointer=True)
