# 安心保 · AI 保险智能导购助手

基于 RAG + LangGraph 智能体的保险咨询对话系统，用户通过自然语言描述需求，Agent 自动完成意图识别、知识库检索、产品推荐、方案生成与人机确认。

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | React 18 + Vite + Ant Design + TypeScript |
| 业务服务 | Python 3.11 + FastAPI + PostgreSQL |
| 智能体服务 | Python 3.11 + FastAPI + LangGraph + LangChain |
| RAG | Milvus 向量库 + BAAI/bge-m3 Embedding + bge-reranker 重排 |
| 大模型 | DeepSeek API（云端）/ Ollama（本地 Qwen） |
| 流式输出 | SSE（Server-Sent Events） |
| 部署 | Docker / 本地 PyCharm + 虚拟机 |

## 项目架构

```
┌─────────────┐     /agent-api      ┌──────────────────┐
│  Frontend   │ ──────────────────→ │  agent-service   │
│  React:5173 │     /business-api   │  FastAPI:8002    │
└─────────────┘ ──────────────────→ │  RAG + LangGraph │
                                     └───────┬──────────┘
                                             │
                    ┌────────────────────────┼────────────────┐
                    │                        │                │
             ┌──────▼──────┐        ┌───────▼──────┐   ┌─────▼─────┐
             │ biz-service │        │  PostgreSQL  │   │  Milvus   │
             │ FastAPI:8001│        │   :5432      │   │  :19530   │
             └─────────────┘        └──────────────┘   └───────────┘
```

## 核心功能

- **意图路由**：LangGraph 编排，自动区分闲聊、产品推荐、理赔咨询三类意图
- **RAG 知识库**：保险条款文档解析 → 分块 → Embedding → Milvus 向量检索 → Rerank 重排
- **多轮对话**：基于 LangGraph Checkpoint 的会话状态管理，支持上下文记忆
- **人机协同（HITL）**：推荐方案生成后中断，等待用户确认再继续
- **SSE 流式输出**：打字机效果实时返回回复

## 目录结构

```
hm-insurance/
├── agent-service/          # 智能体服务（RAG + LangGraph）
│   ├── app/
│   │   ├── api/            # FastAPI 路由（chat、chat-threads、health）
│   │   ├── core/           # 配置、数据库连接、日志
│   │   ├── rag/            # RAG 全流程（解析、分块、Embedding、检索、重排）
│   │   ├── orchestrator/   # LangGraph 编排器（意图路由）
│   │   ├── workflows/      # 专项工作流（推荐、理赔、闲聊）
│   │   ├── clients/        # 调用 biz-service 的 HTTP 客户端
│   │   └── models/         # 数据模型
│   └── scripts/            # RAG 数据导入脚本
├── biz-service/            # 业务服务（产品、订单、理赔、认证）
│   └── app/
│       ├── routers/        # 路由（products、orders、claims、auth...）
│       └── config.py       # 配置
└── frontend/               # React 前端
    └── src/
        ├── pages/          # 页面（首页、顾问页、申请表）
        └── api.ts          # 接口封装
```

## 本地运行

### 前置条件

- Python 3.11+（使用 [uv](https://docs.astral.sh/uv/) 管理依赖）
- Node.js 18+
- 虚拟机或本地已启动 PostgreSQL 和 Milvus
- DeepSeek API Key（或本地 Ollama）

### 1. 配置环境变量

```bash
# agent-service
cp agent-service/.env.example agent-service/.env
# 编辑 .env，填入 DEEPSEEK_API_KEY 和数据库地址

# biz-service
cp biz-service/.env.example biz-service/.env
```

### 2. 启动业务服务

```bash
cd biz-service
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8001
```

### 3. 启动智能体服务

```bash
cd agent-service
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8002
```

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

浏览器访问 `http://localhost:5173`

## 开发说明

- 智能体服务默认连接虚拟机 `192.168.150.101` 上的 PostgreSQL 和 Milvus
- 大模型默认使用 DeepSeek API，可在 `.env` 中切换为本地 Ollama
- RAG 知识库导入：`python scripts/rag_ingest.py`
