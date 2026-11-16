from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from copy import deepcopy
import json
from typing import Any

from langchain.tools import ToolRuntime, tool
from langgraph.config import get_stream_writer
from langgraph.types import Command, interrupt

from app.clients.business_service import BusinessServiceError
from app.core.logging import get_logger
from app.orchestrator.orchestrator_runtime import ChatRuntimeContext
from app.workflows.tool_utils import get_user_token, json_text, state_command
from app.rag import SearchInsuranceKnowledgeRequest, SearchEvidenceResult
from app.rag.models import SectionType

from app.workflows.recommendation.models import RecommendationPlan, InsuredProfile, InsuranceCategory

logger = get_logger(__name__)


@tool
async def update_insured_profile(
    runtime: ToolRuntime[ChatRuntimeContext],
    insured_profile: InsuredProfile
) -> Command:
    """保存或更新被保人的年龄、年预算和职业。"""

    profile = dict(runtime.state.get("insured_profile") or {})
    updates = insured_profile.model_dump()
    profile.update({key: value for key, value in updates.items() if value is not None})
    return state_command(runtime, {"insured_profile": profile}, {"insured_profile": profile})


@tool
async def load_candidates(
    runtime: ToolRuntime[ChatRuntimeContext],
    insurance_category: list[InsuranceCategory] | None = None,
    premium_min: float | None = None,
) -> str | Command:
    """查询候选产品。
    Args:
        insurance_category: 是保险类型，没有可留空，可选值: medical, critical_illness, accident, life。
        premium_min: 表示产品最低保费的严格上限。
    """

    categories = insurance_category or [c.value for c in InsuranceCategory]

    try:
        products = await runtime.context.business_client.list_candidate_products(
            categories,
            premium_min=premium_min,
            limit_per_category=5,
        )
        candidates = normalize_candidate_products(products)

        # 把需要传递给前端的拓展消息写入事件流
        writer = get_stream_writer()
        writer({"type": "additional_info", "data": candidates})

        return state_command(
            runtime,
            {"candidate_products": candidates, "additional_info": candidates},
            {"candidate_products": candidates,}
        )
    except Exception as exc:
        logger.exception("recommendation_candidates_load_failed")
        return json_text({"error": "候选产品查询失败", "detail": str(exc)})


@tool
async def build_insurance_plan(
    recommendation_plan: RecommendationPlan,
    runtime: ToolRuntime[ChatRuntimeContext],
) -> str | Command:
    """把选中的候选产品组装为一套保险方案"""

    # 找到state中的候选产品
    candidates = runtime.state.get("candidate_products") or []
    products_by_id = {
        int(product["id"]): product
        for product in candidates
        if product.get("id") is not None
    }
    # 校验推荐的保险方案，是否符合要求或存在幻觉
    plan_items = recommendation_plan.plan_items
    if not plan_items:
        return json_text({"error": "至少选择一个候选产品"})

    # 校验并组织方案中的产品
    categories: set[str] = set()
    missing_ids: list[int] = []
    final_items = []
    for item in plan_items:
        # 拿到对应的候选产品信息
        product = products_by_id.get(item.product_id)
        # 判断是否是候选产品，缺失的话说明出现幻觉
        if product is None:
            missing_ids.append(item.product_id)
            continue
        # 判断保险类别是否重复，重复说明出现幻觉
        category = str(product.get("category"))
        if category in categories:
            return json_text({"error": "每个险种最多选择一个产品", "category": category})
        categories.add(category)
        # 填充产品信息，组成推荐方案
        final_items.append(build_plan_item(product, item.recommend_reason))

    if missing_ids:
        return json_text(
            {
                "error": "只能选择 load_candidates 返回的候选产品",
                "missing_product_ids": missing_ids,
            }
        )

    # 填充计划信息
    plan = {
        "plan_name": recommendation_plan.name,
        "summary": recommendation_plan.summary,
        "budget_note": recommendation_plan.budget_note,
        "items": final_items,
        "annual_premium_budget": sum_annual_budget(final_items),
        "source": "generated",
        "dirty": True,
    }

    # 把需要传递给前端的拓展消息写入事件流，这里是推荐方案中的产品信息
    writer = get_stream_writer()
    writer({"type": "additional_info", "data": final_items})

    return state_command(runtime, {"plan": plan, "additional_info": final_items}, {"plan": plan})


@tool
async def update_insurance_plan(
    product_id: int,
    runtime: ToolRuntime[ChatRuntimeContext],
) -> str | Command:
    """用候选产品替换方案中同险种的未投保产品。"""

    # 已经有方案的情况下才能更新方案，否则要去生成方案
    plan = deepcopy(runtime.state.get("plan"))
    if not plan:
        return json_text({"error": "State 中没有保险方案，请先生成或查询方案"})

    # 查看用户选择的产品的保险类型
    candidate = next(
        (
            product
            for product in runtime.state.get("candidate_products") or []
            if product.get("id") == product_id
        ),
        None,
    )
    if candidate is None:
        return json_text({"error": "目标产品不在候选产品中，请先调用 load_candidates"})

    # 找到当前方案中的同类型产品
    category = candidate.get("category")
    items = list(plan.get("items") or [])
    index = next(
        (position for position, item in enumerate(items) if item.get("category") == category),
        None,
    )

    # 判断是否存在
    if index is None: # 不存在，则直接添加
        items.append(build_plan_item(candidate, "用户选择"))
    else: # 已存在，判断是否是已投保
        # 已投保，禁止修改，理论上不会走到这一步
        if items[index].get("status") == "insured":
            return json_text({"error": "已投保的方案项禁止修改", "product_id": items[index].get("product_id")})
        # 未投保，直接覆盖
        items[index] = build_plan_item(candidate, "用户选择")

    # 更新state
    plan.update(
        {
            "items": items,
            "annual_premium_budget": sum_annual_budget(items),
            "dirty": True,
        }
    )

    # 把需要传递给前端的拓展消息写入事件流，这里是推荐方案中的产品信息
    writer = get_stream_writer()
    writer({"type": "additional_info", "data": items})
    return state_command(runtime, {"plan": plan, "additional_info": items}, "方案更新成功。")


@tool
async def get_insurance_plan(
    runtime: ToolRuntime[ChatRuntimeContext],
) -> str | Command:
    """优先返回 State 中的方案；没有时查询当前用户的业务方案。"""

    # 先看state是否存在
    state_plan = runtime.state.get("plan")
    if state_plan:
        return json_text(
            {
                "insured_profile": runtime.state.get("insured_profile") or {},
                "plan": state_plan,
            }
        )

    # 获取用户token，查询数据库
    token = get_user_token(runtime)
    if not token:
        return json_text({"error": "用户未登录，无法查询保险方案"})
    try:
        result = await runtime.context.business_client.get_current_insurance_plan(token)
        profile = result.get("insured_profile") or {}
        plan = _database_plan_state(result)

        # 把需要传递给前端的拓展消息写入事件流，这里是推荐方案中的产品信息
        writer = get_stream_writer()
        writer({"type": "additional_info", "data": plan.get('items')})

        return state_command(
            runtime,
            {"insured_profile": profile, "plan": plan, "additional_info": plan.get('items')},
            {"insured_profile": profile, "plan": plan},
        )
    except BusinessServiceError as exc:
        if exc.status_code == 404:
            return json_text({"message": "当前用户还没有保险方案"})
        logger.exception("recommendation_plan_get_failed")
        return json_text({"error": "保险方案查询失败", "detail": str(exc)})


@tool
async def save_insurance_plan(
    runtime: ToolRuntime[ChatRuntimeContext],
) -> str:
    """用户明确确认后，把 State 中的保险方案保存为当前方案。"""

    # 先做数据校验，放到前面，即便是人工确认后还会再次校验
    token = get_user_token(runtime)
    plan = runtime.state.get("plan") or {}
    profile = runtime.state.get("insured_profile") or {}
    if not token:
        return json_text({"error": "用户未登录，无法保存保险方案"})
    if not plan.get("items"):
        return json_text({"error": "State 中没有可保存的保险方案"})

    # 人工确认
    decision = interrupt({
        "plan": plan,
        "confirm_info": "确认保存保险方案吗?",
        "action": ["approve", "reject"]
    })
    if isinstance(decision, str):
        try:
            decision = json.loads(decision)
        except json.JSONDecodeError:
            return f"方案保存确认参数格式错误，期望对象，实际收到字符串：{decision}"

    if not isinstance(decision, dict):
        return "方案保存确认参数格式错误，期望对象。"

    # 用户拒绝，返回给LLM拒绝原因
    if decision.get('action') == 'reject':
        return f"用户拒绝保存此套方案，原因：{decision.get('reject_reason', '未知')}"

    # 用户已批准，开始保存
    try:
        result = await runtime.context.business_client.save_current_insurance_plan(
            token,
            {
                "plan_name": plan.get("plan_name") or "我的保险方案",
                "summary": plan.get("summary"),
                "budget_note": plan.get("budget_note"),
                "insured_profile": profile,
                "items": _plan_items_for_api(plan.get("items") or []),
            },
        )
        # 拿到保存后的数据库plan信息
        plan = _database_plan_state(result)
        writer = get_stream_writer()
        writer({"type": "additional_info", "data": plan.get('items')})

        return state_command(
            runtime,
            {"insured_profile": profile, "plan": plan, "additional_info": plan.get('items')},
            {
                "saved": True,
                "plan_id": result.get("id"),
                "status": result.get("status"),
            },
        )
    except Exception as exc:
        logger.exception("recommendation_plan_save_failed")
        return json_text({"error": "保险方案保存失败", "detail": str(exc)})


@tool
async def search_insurance_knowledge(
    query: str,
    runtime: ToolRuntime[ChatRuntimeContext],
    product_id: int | None = None,
    top_k: int = 5,
) -> str:
    """检索保险产品条款，返回回答证据和可追溯引用来源。"""

    try:
        # 检索
        evidences: list[SearchEvidenceResult] = await runtime.context.search_tool.search(
            SearchInsuranceKnowledgeRequest(
                query=query,
                product_id=product_id,
                top_k=top_k,
            )
        )

        # 组织检索到的条款内容
        clauses = [
            {"content": r.content, "source_id": r.source_id}
            for r in evidences
        ]
        # 组织条款来源
        clause_sources = {e.source_id: e.model_dump() for e in evidences}

        # 把需要传递给前端的拓展消息写入事件流，这里是条款的引用来源
        writer = get_stream_writer()
        writer({"type": "additional_info", "data": clause_sources})

        return state_command(
            runtime,
            {"additional_info": clause_sources},
            {"clauses": clauses},
        )
    except Exception as exc:
        logger.exception("recommendation_plan_knowledge_search_failed", product_id=product_id)
        return json_text({"error": "保险条款检索失败", "detail": str(exc)})



def normalize_candidate_products(products: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """只保留推荐模型需要理解和选择的产品字段。"""

    return [
        {
            "id": product.get("id"),
            "name": product.get("name"),
            "category": product.get("category"),
            "image_url": product.get("image_url"),
            "description": product.get("description"),
            "min_premium": _string_or_none(product.get("min_premium")),
            "target_group": product.get("target_group"),
            "highlights": product.get("highlights") or [],
        }
        for product in products
    ]


def build_plan_item(
    product: dict[str, Any],
    recommend_reason: str,
) -> dict[str, Any]:
    """用候选产品的真实数据组装方案项，避免 LLM 编造产品信息。"""

    return {
        "product_id": product["id"],
        "name": product.get("name"),
        "category": product.get("category"),
        "image_url": product.get("image_url"),
        "highlights": product.get("highlights") or [],
        "annual_premium_budget": _string_or_none(product.get("min_premium")),
        "recommendation_reason": recommend_reason,
    }


def sum_annual_budget(items: Sequence[dict[str, Any]]) -> str | None:
    """汇总方案项的年缴预算参考；任一产品缺少预算时不伪造合计值。"""

    values = [_decimal(item.get("annual_premium_budget")) for item in items]
    if any(value is None for value in values):
        return None
    return str(sum((value for value in values if value is not None), Decimal("0")))


def _decimal(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _string_or_none(value: Any) -> str | None:
    return None if value is None else str(value)




def _database_plan_state(result: dict[str, Any]) -> dict[str, Any]:
    items = [_database_plan_item(item) for item in result.get("items") or []]
    return {
        "id": result.get("id"),
        "status": result.get("status"),
        "plan_name": result.get("plan_name"),
        "summary": result.get("summary"),
        "budget_note": result.get("budget_note"),
        "items": items,
        "annual_premium_budget": result.get("annual_premium_budget"),
        "source": "database",
        "dirty": False,
    }


def _database_plan_item(item: dict[str, Any]) -> dict[str, Any]:
    """把 Biz 返回的嵌套商品信息转换为 Agent 方案中的展示字段。"""

    product = item.get("product") or {}
    return {
        **item,
        "name": product.get("name"),
        "description": product.get("description"),
        "insurer": product.get("insurer"),
        "image_url": product.get("image_url"),
        "highlights": product.get("highlights") or [],
    }


def _plan_items_for_api(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed_fields = {
        "product_id",
        "category",
        "priority",
        "recommendation_reason",
        "annual_premium_budget",
    }
    return [
        {key: value for key, value in item.items() if key in allowed_fields}
        for item in items
    ]


tools = [
    update_insured_profile,
    load_candidates,
    build_insurance_plan,
    update_insurance_plan,
    get_insurance_plan,
    save_insurance_plan,
    search_insurance_knowledge,
]
