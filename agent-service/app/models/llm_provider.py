from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.core.config import Settings, settings
from app.core.logging import get_logger


ModelName = Literal["chat", "local"]
logger = get_logger(__name__)


class LLMNotConfiguredError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMProvider:
    """保存应用启动时创建的模型，供 LangGraph Runtime Context 使用。"""

    chat_model: BaseChatModel
    local_model: BaseChatModel

    @classmethod
    def create(cls, app_settings: Settings = settings) -> LLMProvider:
        """根据应用配置创建并保存所有模型实例。"""

        return cls(
            chat_model=_create_chat_model(app_settings),
            local_model=_create_local_model(app_settings),
        )

    def get(self, name: ModelName) -> BaseChatModel:
        if name == "chat":
            return self.chat_model
        if name == "local":
            return self.local_model
        raise ValueError(f"Unknown model name: {name}")


def _create_chat_model(app_settings: Settings) -> BaseChatModel:
    if not app_settings.deepseek_api_key:
        logger.error("chat_llm_not_configured", model=app_settings.chat_model)
        raise LLMNotConfiguredError("DEEPSEEK_API_KEY is not configured")
    logger.info(
        "chat_llm_initializing",
        model=app_settings.chat_model,
        base_url=app_settings.deepseek_base_url,
    )
    return ChatOpenAI(
        model=app_settings.chat_model,
        api_key=app_settings.deepseek_api_key,
        base_url=app_settings.deepseek_base_url,
        temperature=app_settings.chat_temperature,
        timeout=30,
    )


def _create_local_model(app_settings: Settings) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    logger.info(
        "local_llm_initializing",
        model=app_settings.ollama_chat_model,
        base_url=app_settings.ollama_base_url,
    )
    return ChatOllama(
        model=app_settings.ollama_chat_model,
        temperature=app_settings.chat_temperature,
        reasoning=False,
    )
