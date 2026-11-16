from __future__ import annotations

from typing import Any

from pymilvus import AnnSearchRequest, RRFRanker

from app.rag.protocols import Embedding, RetrievalHit


class MilvusHybridRetriever:
    """在 Milvus 中并行执行稠密检索和稀疏检索，再融合候选结果。"""

    def __init__(
        self,
        *,
        client: Any,
        embedding: Embedding,
        collection_name: str = "insurance_knowledge_chunks",
    ) -> None:
        # Embedding 只通过协议依赖，Retriever 不关心具体向量模型。
        self.client = client
        self.embedding = embedding
        self.collection_name = collection_name

    def search(
        self,
        query: str,
        candidate_k: int,
        *,
        filter_expression: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[RetrievalHit]:
        """返回融合分数和调用方指定的 Milvus 字段。"""

        # BGE-M3 一次编码同时得到 dense 和 sparse 两路查询向量。
        vectors = self.embedding.embed_query(query)
        # 两路检索使用同一过滤条件，确保召回范围保持一致。
        reqs = [
            AnnSearchRequest(
                data=[vectors["dense"][0]],
                anns_field="dense_vector",
                param={"metric_type": "IP"},
                limit=candidate_k,
                filter=filter_expression,
            ),
            AnnSearchRequest(
                data=[vectors["sparse"][0:1]],
                anns_field="sparse_vector",
                param={"metric_type": "IP"},
                limit=candidate_k,
                filter=filter_expression,
            ),
        ]
        # RRF 根据两路结果中的排名进行融合，不要求两种分数处于同一量纲。
        raw_results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=reqs,
            ranker=RRFRanker(),
            limit=candidate_k,
            output_fields=output_fields or [],
        )
        raw_hits = raw_results[0] if raw_results else []
        return [_to_retrieval_hit(hit) for hit in raw_hits]


def _to_retrieval_hit(raw_hit: Any) -> RetrievalHit:
    """屏蔽 pymilvus 命中对象格式，只向下游暴露分数和字段。"""

    fields = raw_hit.get("entity", raw_hit)
    score = float(raw_hit.get("distance", raw_hit.get("score", 0.0)))
    return score, dict(fields)
