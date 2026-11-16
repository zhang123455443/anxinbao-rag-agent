from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from app.core.config import Settings

CheckResult = dict[str, Any]


async def run_dependency_checks(settings: Settings) -> dict[str, CheckResult]:
    checks = await asyncio.gather(
        check_postgresql(settings),
        check_redis(settings),
        check_milvus(settings),
        check_http_service("business_service", f"{settings.business_service_url}/api/v1/health", settings),
        check_http_service("ollama", f"{settings.ollama_base_url}/api/tags", settings)
        if settings.ollama_base_url
        else skipped("ollama", "OLLAMA_BASE_URL is not configured"),
    )
    return {name: result for name, result in checks}


async def check_postgresql(settings: Settings) -> tuple[str, CheckResult]:
    if not settings.database_url:
        return "postgresql", status("skipped", "DATABASE_URL is not configured")

    async def _check() -> None:
        import asyncpg

        url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
        connection = await asyncpg.connect(url, timeout=settings.health_check_timeout_seconds)
        try:
            await connection.execute("SELECT 1")
        finally:
            await connection.close()

    return "postgresql", await timed(_check, settings.health_check_timeout_seconds)


async def check_redis(settings: Settings) -> tuple[str, CheckResult]:
    if not settings.redis_url:
        return "redis", status("skipped", "REDIS_URL is not configured")

    async def _check() -> None:
        from redis import asyncio as redis

        client = redis.from_url(
            settings.redis_url,
            socket_connect_timeout=settings.health_check_timeout_seconds,
            protocol=2,
        )
        try:
            await client.ping()
        finally:
            await client.aclose()

    return "redis", await timed(_check, settings.health_check_timeout_seconds)


async def check_milvus(settings: Settings) -> tuple[str, CheckResult]:
    if not settings.milvus_host:
        return "milvus", status("skipped", "MILVUS_HOST is not configured")

    async def _check() -> None:
        def connect_and_disconnect() -> None:
            from pymilvus import connections, utility

            alias = f"health_{uuid4().hex}"
            connections.connect(
                alias=alias,
                host=settings.milvus_host,
                port=str(settings.milvus_port),
                timeout=settings.health_check_timeout_seconds,
            )
            try:
                utility.get_server_version(using=alias)
            finally:
                connections.disconnect(alias)

        await asyncio.to_thread(connect_and_disconnect)

    return "milvus", await timed(_check, settings.health_check_timeout_seconds)


async def check_http_service(name: str, url: str, settings: Settings) -> tuple[str, CheckResult]:
    async def _check() -> None:
        import httpx

        async with httpx.AsyncClient(timeout=settings.health_check_timeout_seconds) as client:
            response = await client.get(url)
            response.raise_for_status()

    return name, await timed(_check, settings.health_check_timeout_seconds)


async def timed(
    check: Callable[[], Awaitable[None]],
    timeout_seconds: float,
) -> CheckResult:
    started = time.perf_counter()
    try:
        await asyncio.wait_for(check(), timeout=timeout_seconds + 0.5)
    except Exception as exc:  # noqa: BLE001 - health endpoint should report dependency errors.
        return status(
            "unhealthy",
            f"{exc.__class__.__name__}: {exc}",
            started,
        )
    return status("healthy", None, started)


async def skipped(name: str, message: str) -> tuple[str, CheckResult]:
    return name, status("skipped", message)


def status(value: str, message: str | None = None, started: float | None = None) -> CheckResult:
    result: CheckResult = {"status": value}
    if message:
        result["message"] = message
    if started is not None:
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result
