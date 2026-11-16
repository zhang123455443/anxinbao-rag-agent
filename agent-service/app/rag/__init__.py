"""RAG 对外接口，内部节点由 factory 统一组装。"""

from app.rag.factory import RagComponents, create_rag_components
from app.rag.pipeline import KnowledgeBuildPipeline, KnowledgeBuildRequest, KnowledgeBuildResult
from app.rag.search_tool import (
    SearchInsuranceKnowledgeRequest,
    SearchEvidenceResult,
    SearchInsuranceKnowledgeTool,
)

__all__ = [
    "KnowledgeBuildPipeline",
    "KnowledgeBuildRequest",
    "KnowledgeBuildResult",
    "RagComponents",
    "SearchEvidenceResult",
    "SearchInsuranceKnowledgeRequest",
    "SearchInsuranceKnowledgeTool",
    "create_rag_components",
]
