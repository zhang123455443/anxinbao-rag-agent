from uuid import UUID

from langchain.tools import ToolRuntime, tool
from langgraph.config import get_stream_writer
from langgraph.types import Command

from app.clients.business_service import BusinessServiceError
from app.core.logging import get_logger
from app.orchestrator.orchestrator_runtime import ChatRuntimeContext
from app.workflows.tool_utils import get_user_token, json_text, state_command
from app.rag import SearchInsuranceKnowledgeRequest, SearchEvidenceResult
from app.rag.models import SectionType
from app.workflows.claim.models import ClaimStatus, PolicyStatus


logger = get_logger(__name__)


@tool
async def list_policies(
    runtime: ToolRuntime[ChatRuntimeContext],
    status: PolicyStatus = PolicyStatus.ACTIVE,
) -> str | Command:
    """查询当前用户保单，默认只查生效保单并按生效时间倒序返回。"""

    token = get_user_token(runtime)
    if not token:
        return json_text({"error": "用户未登录，无法查询保单"})
    try:
        result = await runtime.context.business_client.list_policies(
            token,
            status=status.value,
        )
        policies = result.get("items") or []

        # 把保单写入拓展字段，供前端展示
        writer = get_stream_writer()
        writer({"type": "additional_info", "data": policies})

        return state_command(runtime, {"policies": policies, "additional_info": policies}, {"policies": policies})
    except Exception as exc:
        logger.exception("claim_policies_query_failed")
        return json_text({"error": "保单查询失败", "detail": str(exc)})


@tool
async def search_claim_knowledge(
    query: str,
    product_id: int,
    runtime: ToolRuntime[ChatRuntimeContext],
    top_k: int = 5,
) -> str:
    """查询指定保单产品的理赔责任、免责、材料、流程和时效证据。"""

    try:
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
        logger.exception("claim_knowledge_search_failed", product_id=product_id)
        return json_text({"error": "理赔条款检索失败", "detail": str(exc)})


@tool
async def get_claim_guide(
    policy_id: UUID,
    runtime: ToolRuntime[ChatRuntimeContext],
) -> str:
    """查询指定保单的理赔流程、基础材料、注意事项和申请表下载链接。"""

    token = get_user_token(runtime)
    if not token:
        return json_text({"error": "用户未登录，无法查询理赔指南"})
    try:
        result = await runtime.context.business_client.get_claim_guide(token, policy_id)
        return json_text(result)
    except BusinessServiceError as exc:
        if exc.status_code == 404:
            return json_text({"error": "未找到对应保单"})
        logger.exception("claim_guide_query_failed")
        return json_text({"error": "理赔指南查询失败", "detail": str(exc)})


@tool
async def list_claims(
    runtime: ToolRuntime[ChatRuntimeContext],
    policy_id: UUID | None = None,
    status: ClaimStatus = ClaimStatus.ALL,
) -> str | Command:
    """按保单或状态查询当前用户的理赔记录，默认按更新时间倒序返回。"""

    token = get_user_token(runtime)
    if not token:
        return json_text({"error": "用户未登录，无法查询理赔记录"})
    try:
        result = await runtime.context.business_client.list_claims(
            token,
            policy_id=policy_id,
            status=status.value,
        )
        claims = result.get("items") or []
        return state_command(runtime, {"claims": claims}, {"claims": claims})
    except Exception as exc:
        logger.exception("claims_query_failed")
        return json_text({"error": "理赔记录查询失败", "detail": str(exc)})


@tool
async def get_claim_progress(
    claim_id: UUID,
    runtime: ToolRuntime[ChatRuntimeContext],
) -> str:
    """查询单个理赔案件的当前状态、进度说明和最后更新时间。"""

    token = get_user_token(runtime)
    if not token:
        return json_text({"error": "用户未登录，无法查询理赔进度"})
    try:
        result = await runtime.context.business_client.get_claim(token, claim_id)
        return json_text(result)
    except BusinessServiceError as exc:
        if exc.status_code == 404:
            return json_text({"error": "未找到对应理赔案件"})
        logger.exception("claim_progress_query_failed")
        return json_text({"error": "理赔进度查询失败", "detail": str(exc)})



tools = [
    list_policies,
    search_claim_knowledge,
    get_claim_guide,
    list_claims,
    get_claim_progress,
]
