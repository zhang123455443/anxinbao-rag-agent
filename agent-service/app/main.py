import asyncio
import sys
from contextlib import asynccontextmanager

# Psycopg 的异步连接在 Windows 上需要 SelectorEventLoop。
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.api.chat import router as chat_router
from app.api.chat_threads import router as chat_threads_router
from app.api.health import router as health_router
from app.clients.business_service import BusinessServiceClient
from app.core.config import settings
from app.core.database import Database, asyncpg_database_url
from app.core.logging import configure_logging, get_logger
from app.models import LLMProvider
from app.orchestrator.intent_router import IntentRouter
from app.orchestrator.orchestrator_graph import ChatOrchestrator
from app.rag import RagComponents, create_rag_components
from app.services.chat_threads import ChatThreadRepository
from app.workflows import create_workflows

configure_logging(level=settings.log_level, log_format=settings.log_format)
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    database = Database(settings)
    rag: RagComponents | None = None
    business_client: BusinessServiceClient | None = None
    checkpointer_context = None
    try:
        logger.info("agent_service_starting", app_version=settings.app_version)
        # 初始化数据库连接池
        await database.initialize()
        app.state.database = database
        if not settings.database_url:
            raise RuntimeError("DATABASE_URL is not configured")
        checkpointer_context = AsyncPostgresSaver.from_conn_string(
            asyncpg_database_url(settings.database_url)
        )
        checkpointer = await checkpointer_context.__aenter__()
        await checkpointer.setup()
        app.state.chat_thread_repository = ChatThreadRepository(database.pool)
        # 初始化模型
        llms = LLMProvider.create(settings)
        # 初始化RAG工具，Markdown 优化节点复用应用启动时创建的 DeepSeek 模型。
        rag = await create_rag_components(
            settings,
            database.pool,
        )
        app.state.rag = rag
        # 初始化业务端工具
        business_client = BusinessServiceClient(settings.business_service_url)
        # 初始化工作流
        intent_router = IntentRouter()
        workflows = create_workflows()
        # 初始化主Graph
        chat_orchestrator = ChatOrchestrator(
            intent_router=intent_router,
            llms=llms,
            business_client=business_client,
            search_tool=rag.search_tool,
            workflows=workflows,
            checkpointer=checkpointer,
        )
        chat_orchestrator.init_graph()
        app.state.chat_orchestrator = chat_orchestrator
        logger.info(
            "agent_service_started",
            workflow_count=len(workflows),
            workflows=[workflow.workflow_name() for workflow in workflows],
        )
        yield
    finally:
        logger.info("agent_service_stopping")
        if rag is not None:
            rag.close()
        if business_client is not None:
            await business_client.aclose()
        if checkpointer_context is not None:
            await checkpointer_context.__aexit__(None, None, None)
        await database.close()
        logger.info("agent_service_stopped")

def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(chat_threads_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")

    @app.middleware("http")
    async def log_http_requests(request, call_next):
        import time

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("http_request_failed", method=request.method, path=request.url.path)
            raise
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "http_request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    # 启动命令：python -m app.main
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=settings.app_port,
        reload=False,
        loop="app.core.event_loop:selector_event_loop_factory",
    )
