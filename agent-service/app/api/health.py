from datetime import datetime, timezone

from fastapi import APIRouter, Response

from app.core.config import settings
from app.services.health_checks import run_dependency_checks

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {
        "status": "healthy",
        "service": "agent-service",
        "version": settings.app_version,
        "time": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "postgresql": "not_checked",
            "milvus": "not_checked",
            "redis": "not_checked",
            "business_service": "not_checked",
            "ollama": "not_checked",
        },
    }


@router.get("/health/ready")
async def readiness(response: Response) -> dict:
    checks = await run_dependency_checks(settings)
    degraded = any(check.get("status") == "unhealthy" for check in checks.values())
    if degraded:
        response.status_code = 503
    return {
        "status": "degraded" if degraded else "healthy",
        "service": "agent-service",
        "version": settings.app_version,
        "time": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }
