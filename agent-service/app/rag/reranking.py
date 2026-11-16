from __future__ import annotations

from dataclasses import replace

from FlagEmbedding import FlagReranker

from app.rag.protocols import RetrievedChunk


class BGEReranker:
    """使用 BGE Cross-Encoder 对 Milvus 候选子块进行精排。"""

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        # 模型在对象初始化时加载，避免每次检索重复加载权重。
        self.model = FlagReranker(
            model_name,
            use_fp16=False,
            devices=device,
            max_length=max_length,
        )

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """计算 query 与每个子块的相关性，并按分数从高到低排序。"""

        if not chunks:
            return []

        # Reranker 接收 [query, passage] 文本对，不使用原始向量。
        pairs = [[query, chunk.content] for chunk in chunks]
        scores = self.model.compute_score(pairs)
        if not isinstance(scores, list):
            scores = [float(scores)]

        # 使用 replace 生成新对象，避免原地覆盖 Retriever 返回的数据。
        reranked = [
            replace(chunk, score=float(score))
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        return sorted(reranked, key=lambda chunk: chunk.score, reverse=True)
