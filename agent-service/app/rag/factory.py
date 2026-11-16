from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings
from app.rag.chunking import MarkdownChunker
from app.rag.embedding import BGEM3Embedding
from app.rag.milvus_retriever import MilvusHybridRetriever
from app.rag.markdown_optimization import DeepSeekMarkdownOptimizer
from app.rag.parsing import MinerUParser
from app.rag.pipeline import KnowledgeBuildPipeline
from app.rag.postgres_repository import PostgresKnowledgeRepository
from app.rag.reranking import BGEReranker
from app.rag.search_tool import SearchInsuranceKnowledgeTool
from app.rag.vector_store import MilvusVectorStore


@dataclass
class RagComponents:
    """应用启动时创建并共享的 RAG 核心组件。"""

    pipeline: KnowledgeBuildPipeline
    search_tool: SearchInsuranceKnowledgeTool
    _milvus_client: Any = field(repr=False)

    def close(self) -> None:
        """关闭 RAG 组件持有的 Milvus 连接。"""

        self._milvus_client.close()


async def create_rag_components(
    settings: Settings,
    pool: Any,
) -> RagComponents:
    """创建知识库构建 Pipeline 和检索 Tool，并共享重量级基础组件。"""

    if not settings.milvus_uri:
        raise RuntimeError("MILVUS_URI or MILVUS_HOST is not configured")

    from pymilvus import MilvusClient

    repository = PostgresKnowledgeRepository(pool)

    milvus_client = MilvusClient(uri=settings.milvus_uri)
    try:
        # 构建和检索共享同一个 BGE-M3 模型，避免重复占用内存。
        embedding = BGEM3Embedding(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )
        pipeline = KnowledgeBuildPipeline(
            repository=repository,
            parser=MinerUParser(
                token=settings.mineru_token,
                base_url=settings.mineru_api_base_url,
                flash_base_url=settings.mineru_flash_base_url,
            ),
            markdown_optimizer=DeepSeekMarkdownOptimizer(
                deepseek_api_key=settings.deepseek_api_key or "",
                deepseek_base_url=settings.deepseek_base_url,
                model=settings.markdown_optimizer_model,
                timeout=settings.markdown_optimizer_timeout_seconds,
                max_tokens=settings.markdown_optimizer_max_tokens,
            ),
            chunker=MarkdownChunker(),
            embedding=embedding,
            vector_store=MilvusVectorStore(
                client=milvus_client,
                collection_name=settings.milvus_collection,
                dense_dim=1024,
            ),
        )
        search_tool = SearchInsuranceKnowledgeTool(
            retriever=MilvusHybridRetriever(
                client=milvus_client,
                embedding=embedding,
                collection_name=settings.milvus_collection,
            ),
            reranker=BGEReranker(
                model_name=settings.rerank_model,
                device=settings.rerank_device,
                max_length=settings.rerank_max_length,
            ),
            repository=repository,
            candidate_k=settings.rag_candidate_k,
        )
    except Exception:
        milvus_client.close()
        raise

    return RagComponents(
        pipeline=pipeline,
        search_tool=search_tool,
        _milvus_client=milvus_client,
    )
