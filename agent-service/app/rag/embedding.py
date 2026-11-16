from __future__ import annotations

from typing import Any


class BGEM3Embedding:
    """使用 BGE-M3 同时生成 Milvus 所需的稠密向量和稀疏向量。"""

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
    ) -> None:
        # 使用 pymilvus 提供的封装，返回值可以直接交给 Milvus 存储和检索。
        from pymilvus.model.hybrid import BGEM3EmbeddingFunction

        # device=None 时由 FlagEmbedding 自动选择可用加速设备，否则回退到 CPU。
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            normalize_embeddings=True,
            use_fp16=False,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    def embed_query(self, text: str) -> dict[str, Any]:
        """向量化单条查询，结果中每种向量仍保留 batch 维度。"""

        if not text.strip():
            raise ValueError("query must not be empty")
        return self.embedding_function.encode_queries([text])

    def embed_documents(self, texts: list[str]) -> dict[str, Any]:
        """批量向量化待入库的子块文本。"""

        if not texts:
            raise ValueError("texts must not be empty")
        return self.embedding_function.encode_documents(texts)
