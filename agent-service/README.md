# agent-service

Python FastAPI AI 服务，负责 RAG、Conversation Orchestrator、专项 Agent/Workflow 和 HITL。

当前状态：M0 最小启动骨架。

## 本地运行

```powershell
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8002
```

健康检查：

```text
GET http://localhost:8002/api/v1/health
```

## Chat Orchestrator 联调

先启动 VM PostgreSQL 背后的 biz-service，再启动 agent-service：

```powershell
# biz-service 示例端口
cd ..\biz-service
uv run uvicorn app.main:app --host 127.0.0.1 --port 18001

# agent-service
cd ..\agent-service
$env:BUSINESS_SERVICE_URL = "http://127.0.0.1:18001"
uv run uvicorn app.main:app --host 127.0.0.1 --port 8002
```

运行最小烟测：

```powershell
$env:BUSINESS_SERVICE_URL = "http://127.0.0.1:18001"
$env:AGENT_SERVICE_URL = "http://127.0.0.1:8002"
uv run python scripts\chat_orchestrator_smoke.py
```

该脚本会注册临时用户，调用 `POST /api/v1/chat` 生成保险组合方案，并验证当前方案保存结果。

Docker 运行：

```powershell
docker compose up agent-service --build
```
