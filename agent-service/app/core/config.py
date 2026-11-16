from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


APP_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(APP_ROOT / ".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "agent-service"
    app_version: str = "0.1.0"
    app_port: int = Field(default=8002, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="console", alias="LOG_FORMAT")

    infra_host: str | None = Field(default=None, alias="INFRA_HOST")
    database_url: str | None = Field(default=None, alias="DATABASE_URL")
    database_pool_min_size: int = Field(default=1, alias="DATABASE_POOL_MIN_SIZE")
    database_pool_max_size: int = Field(default=10, alias="DATABASE_POOL_MAX_SIZE")
    database_pool_max_queries: int = Field(default=50000, alias="DATABASE_POOL_MAX_QUERIES")
    database_pool_max_inactive_connection_lifetime: float = Field(
        default=300.0,
        alias="DATABASE_POOL_MAX_INACTIVE_CONNECTION_LIFETIME",
    )
    database_connect_timeout_seconds: float = Field(default=10.0, alias="DATABASE_CONNECT_TIMEOUT_SECONDS")
    database_command_timeout_seconds: float | None = Field(
        default=None,
        alias="DATABASE_COMMAND_TIMEOUT_SECONDS",
    )
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_agent_db: str | None = Field(default=None, alias="POSTGRES_AGENT_DB")
    postgres_agent_user: str | None = Field(default=None, alias="POSTGRES_AGENT_USER")
    postgres_agent_password: str | None = Field(default=None, alias="POSTGRES_AGENT_PASSWORD")
    postgres_agent_port: str | None = Field(default=None, alias="POSTGRES_AGENT_PORT")
    business_database_url: str | None = Field(default=None, alias="BUSINESS_DATABASE_URL")
    postgres_biz_db: str = Field(default="insurance_biz", alias="POSTGRES_BIZ_DB")
    postgres_biz_user: str | None = Field(default=None, alias="POSTGRES_BIZ_USER")
    postgres_biz_password: str | None = Field(default=None, alias="POSTGRES_BIZ_PASSWORD")
    postgres_biz_port: int | None = Field(default=None, alias="POSTGRES_BIZ_PORT")
    redis_url: str | None = Field(default=None, alias="REDIS_URL")
    milvus_uri: str | None = Field(default=None, alias="MILVUS_URI")
    milvus_host: str | None = Field(default=None, alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    ollama_base_url: str | None = Field(default="http://127.0.0.1:11434", alias="OLLAMA_BASE_URL")
    ollama_chat_model: str | None = Field(default="qwen3.5:0.8b", alias="OLLAMA_CHAT_MODEL")
    business_service_url: str = Field(
        default="http://localhost:8001",
        alias="BUSINESS_SERVICE_URL",
    )
    health_check_timeout_seconds: float = Field(default=2.0, alias="HEALTH_CHECK_TIMEOUT_SECONDS")
    jwt_issuer: str = Field(default="hm-insurance", alias="JWT_ISSUER")
    jwt_secret: str = Field(default="change-me-in-dev", alias="JWT_SECRET")
    service_api_key: str = Field(default="change-me-service-api-key", alias="SERVICE_API_KEY")
    mineru_token: str | None = Field(default=None, alias="MINERU_TOKEN")
    mineru_api_base_url: str | None = Field(default=None, alias="MINERU_API_BASE_URL")
    mineru_flash_base_url: str | None = Field(default=None, alias="MINERU_FLASH_BASE_URL")
    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    chat_model: str = Field(default="deepseek-chat", alias="CHAT_MODEL")
    markdown_optimizer_model: str = Field(default="deepseek-v4-flash", alias="MARKDOWN_OPTIMIZER_MODEL")
    chat_temperature: float = Field(default=0.2, alias="CHAT_TEMPERATURE")
    markdown_optimizer_timeout_seconds: float = Field(
        default=180.0,
        alias="MARKDOWN_OPTIMIZER_TIMEOUT_SECONDS",
    )
    markdown_optimizer_max_tokens: int = Field(
        default=32000,
        alias="MARKDOWN_OPTIMIZER_MAX_TOKENS",
    )
    milvus_collection: str = Field(
        default="insurance_knowledge_chunks",
        alias="MILVUS_COLLECTION",
    )
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    embedding_device: str | None = Field(
        default=None,
        alias="EMBEDDING_DEVICE",
        description="由 FlagEmbedding 自动选择 GPU，GPU 不可用时回退到 CPU",
    )
    rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANK_MODEL")
    rerank_device: str | None = Field(
        default=None,
        alias="RERANK_DEVICE",
        description="由 FlagEmbedding 自动选择 GPU，GPU 不可用时回退到 CPU",
    )
    rerank_max_length: int = Field(default=512, alias="RERANK_MAX_LENGTH")
    rag_candidate_k: int = Field(default=50, alias="RAG_CANDIDATE_K")

    cors_origins: list[str] = ["http://localhost:5173"]

    @model_validator(mode="after")
    def derive_infra_urls(self) -> "Settings":
        if self.database_url is None and self.infra_host:
            if self.postgres_agent_db and self.postgres_agent_user and self.postgres_agent_password:
                self.database_url = (
                    "postgresql://"
                    f"{self.postgres_agent_user}:{self.postgres_agent_password}"
                    f"@{self.infra_host}:{self.postgres_agent_port or self.postgres_port}/{self.postgres_agent_db}"
                )
        if self.milvus_host is None and self.infra_host:
            self.milvus_host = self.infra_host
        if self.milvus_uri is None and self.milvus_host:
            self.milvus_uri = f"http://{self.milvus_host}:{self.milvus_port}"
        if self.business_database_url is None and self.infra_host:
            biz_user = self.postgres_biz_user or self.postgres_agent_user
            biz_password = self.postgres_biz_password or self.postgres_agent_password
            if biz_user and biz_password:
                self.business_database_url = (
                    f"postgresql://{biz_user}:{biz_password}"
                    f"@{self.infra_host}:{self.postgres_biz_port or self.postgres_port}"
                    f"/{self.postgres_biz_db}"
                )
        return self


settings = Settings()
