from __future__ import annotations

from pathlib import Path

import asyncpg

from app.core.config import Settings
from app.core.logging import get_logger


logger = get_logger(__name__)


def asyncpg_database_url(database_url: str) -> str:
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


class DatabaseNotInitializedError(RuntimeError):
    pass


class Database:
    def __init__(self, settings: Settings) -> None:
        database_url = settings.database_url
        self._database_url = asyncpg_database_url(database_url) if database_url else None
        self._settings = settings
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise DatabaseNotInitializedError("database pool is not initialized")
        return self._pool

    async def initialize(self) -> None:
        if self._pool is not None:
            logger.debug("database_pool_already_initialized")
            return
        if not self._database_url:
            logger.warning("database_url_not_configured")
            return
        logger.info(
            "database_pool_initializing",
            min_size=self._settings.database_pool_min_size,
            max_size=self._settings.database_pool_max_size,
        )
        self._pool = await asyncpg.create_pool(
            self._database_url,
            min_size=self._settings.database_pool_min_size,
            max_size=self._settings.database_pool_max_size,
            max_queries=self._settings.database_pool_max_queries,
            max_inactive_connection_lifetime=(
                self._settings.database_pool_max_inactive_connection_lifetime
            ),
            timeout=self._settings.database_connect_timeout_seconds,
            command_timeout=self._settings.database_command_timeout_seconds,
        )
        await initialize_schema(self._pool)
        logger.info("database_pool_initialized")

    async def close(self) -> None:
        if self._pool is None:
            return
        logger.info("database_pool_closing")
        await self._pool.close()
        self._pool = None
        logger.info("database_pool_closed")

def create_database(settings: Settings) -> Database:
    return Database(settings)


async def initialize_schema(pool: asyncpg.Pool) -> None:
    """执行 agent-service 根目录下的统一建表脚本。"""

    schema_path = Path(__file__).resolve().parents[2] / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")
    async with pool.acquire() as connection:
        await connection.execute(schema_sql)
