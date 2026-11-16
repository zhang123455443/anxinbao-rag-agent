from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.rag.models import SectionType
from app.rag.protocols import (
    KnowledgeRepository,
    Reranker,
    RetrievalHit,
    RetrievedChunk,
    Retriever,
)


class SearchInsuranceKnowledgeRequest(BaseModel):
    """RAG Tool 的检索参数，由调用 Tool 的 Agent 生成。"""

    query: str  # 用户问题或 Agent 改写后的检索问题。
    top_k: int = 5  # Rerank 后返回的结果数量。
    product_id: int | None = None  # 指定产品，同时允许召回 product_ids 中含 -1 的通用资料。
    section_types: list[SectionType] = Field(default_factory=list)  # 可选章节过滤条件。


class SearchEvidenceResult(BaseModel):
    """一条检索证据，同时包含回答上下文和可溯源引用信息。"""

    source_id: str  # 本次响应中的引用编号。
    parent_chunk_id: str  # 前端查看父块原文时使用。
    document_id: str  # 引用文档标识。
    document_name: str  # 引用文档名称。
    section_path: list[str] = Field(default_factory=list)  # 引用所在章节。
    clause_no: str | None = None  # 条款编号。
    content: str  # 返回给 Agent 的上下文；重复命中同父块时为父块内容，否则为子块内容。
    score: float  # 子块经过 Rerank 后的相关性分数。


class SearchInsuranceKnowledgeTool:
    """依次编排混合检索、重排和父块回查三个检索节点。"""

    # 只从 Milvus 读取后续重排和溯源需要的字段。
    # product_ids 和 section_type 仅用于过滤，无需随检索结果返回。
    output_fields = [
        "parent_chunk_id",
        "document_id",
        "document_name",
        "section_path",
        "clause_no",
        "content",
    ]

    def __init__(
        self,
        *,
        retriever: Retriever,
        reranker: Reranker,
        repository: KnowledgeRepository,
        candidate_k: int = 50,
    ) -> None:
        # 各节点通过 Protocol 约束，Tool 只负责编排，不依赖具体实现类。
        self.retriever = retriever
        self.reranker = reranker
        self.repository = repository
        self.candidate_k = candidate_k

    async def search(
        self,
        request: SearchInsuranceKnowledgeRequest,
    ) -> list[SearchEvidenceResult]:
        """执行完整检索流程，返回合并上下文和引用信息的证据列表。"""

        # 1. 在 Milvus 中过滤知识范围，并召回较多子块作为候选集。
        retrieval_hits = self.retriever.search(
            request.query,
            self.candidate_k,
            filter_expression=_build_filter(request.product_id, request.section_types),
            output_fields=self.output_fields,
        )

        # 2. 将 Milvus 原始结果转换为节点间统一模型，再计算 Rerank 分数。
        chunks = [_to_retrieved_chunk(hit) for hit in retrieval_hits]
        reranked = self.reranker.rerank(request.query, chunks)[: request.top_k]

        # 3. 只有同一个父块命中多个子块时，才回查父块完整内容。
        parent_counts: dict[str, int] = {}
        for hit in reranked:
            parent_counts[hit.parent_chunk_id] = parent_counts.get(hit.parent_chunk_id, 0) + 1
        duplicated_parent_ids = [
            parent_id for parent_id, count in parent_counts.items() if count > 1
        ]
        parents = await self.repository.get_parent_chunks(duplicated_parent_ids)

        # 4. 按 Rerank 顺序组装证据；重复父块只返回一次父块内容。
        results: list[SearchEvidenceResult] = []
        added_parent_ids: set[str] = set()
        for hit in reranked:
            use_parent = parent_counts.get(hit.parent_chunk_id, 0) > 1
            if use_parent and hit.parent_chunk_id in added_parent_ids:
                continue
            if use_parent:
                added_parent_ids.add(hit.parent_chunk_id)
            parent = parents.get(hit.parent_chunk_id) if use_parent else None
            results.append(
                _result_from_hit(
                    hit,
                    len(results) + 1,
                    content=parent.content if parent is not None else hit.content,
                )
            )

        return results


def _result_from_hit(hit: RetrievedChunk, index: int, *, content: str) -> SearchEvidenceResult:
    """根据命中的子块生成本次响应中的证据。"""

    return SearchEvidenceResult(
        source_id=f"ref-{index:03d}",
        parent_chunk_id=hit.parent_chunk_id,
        document_id=hit.document_id,
        document_name=hit.document_name,
        section_path=hit.section_path,
        clause_no=hit.clause_no,
        content=content,
        score=hit.score,
    )


def _build_filter(product_id: int | None, section_types: list[SectionType]) -> str | None:
    """构造 Milvus 标量过滤表达式，未指定条件时返回 None。"""

    conditions: list[str] = []
    if product_id is not None:
        # -1 表示不属于特定产品的通用保险知识。
        conditions.append(f"ARRAY_CONTAINS_ANY(product_ids, [{product_id}, -1])")
    if section_types:
        values = ", ".join(f'"{section_type.value}"' for section_type in section_types)
        conditions.append(f"section_type in [{values}]")
    return " and ".join(conditions) or None


def _to_retrieved_chunk(hit: RetrievalHit) -> RetrievedChunk:
    """将 Retriever 返回的 (score, fields) 转换为清晰的领域对象。"""

    score, fields = hit
    return RetrievedChunk(
        parent_chunk_id=str(fields["parent_chunk_id"]),
        document_id=str(fields["document_id"]),
        document_name=str(fields["document_name"]),
        section_path=_section_path(fields.get("section_path")),
        clause_no=fields.get("clause_no") or None,
        content=str(fields["content"]),
        score=score,
    )


def _section_path(value: Any) -> list[str]:
    """兼容 Milvus 返回的列表或 JSON 字符串格式。"""

    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(item) for item in json.loads(str(value))]
