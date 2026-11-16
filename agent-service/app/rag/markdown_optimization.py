from typing import Any

from openai import AsyncOpenAI

MARKDOWN_OPTIMIZATION_PROMPT = """
下面这份Markdown文档是从保险条款PDF解析得来，由于PDF中的各个小节是以表格形式存在，所以解析时出现错乱。你分析内容，帮我转为格式正确的Markdown，特别是标题编号要正确。
- 标题等级要从1级标题开始，逐层增加。
- 处理时正文开始之前面的目录等部分删除。
- 输出结果只包含Markdown正文，不要解释或代码围栏。
- 不要修改保险条款。
""".strip()


class DeepSeekMarkdownOptimizer:
    """使用 DeepSeek 修复 MinerU 解析后的 Markdown 结构。"""

    def __init__(
        self,
        *,
        deepseek_api_key: str,
        deepseek_base_url: str,
        model: str,
        timeout: float,
        max_tokens: int,
    ) -> None:
        if not deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        self.model = model
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(
            api_key=deepseek_api_key,
            base_url=_beta_base_url(deepseek_base_url),
            timeout=timeout,
            max_retries=2,
        )

    async def call_openai(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str, str | None]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=self.max_tokens,
            extra_body={"thinking": {"type": "disabled"}}
        )
        if not response.choices:
            raise RuntimeError("DeepSeek returned no optimized Markdown choices")
        choice = response.choices[0]
        return _message_text(choice.message.content), choice.finish_reason

    async def optimize(self, markdown: str) -> str:
        if not markdown.strip():
            return ""

        # 先调用一次
        optimized, finish_reason = await self.call_openai(self._messages(markdown))

        # 单次输出达到上限时，把当前完整输出作为 assistant 前缀继续补写。
        while finish_reason == 'length':
            continuation, finish_reason = await self.call_openai(self._continue_messages(markdown, optimized))
            if not continuation and finish_reason == "length":
                raise RuntimeError("DeepSeek returned empty Markdown continuation")
            # 拼接后续内容，继续生成
            optimized += continuation

        # 格式化结果并返回
        optimized = optimized.strip()
        optimized = _strip_markdown_fence(optimized)
        if not optimized:
            raise RuntimeError("DeepSeek returned empty optimized Markdown")
        return optimized

    @staticmethod
    def _messages(markdown: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": MARKDOWN_OPTIMIZATION_PROMPT},
            {"role": "user", "content": markdown},
        ]

    @staticmethod
    def _continue_messages(markdown: str, prefix: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": MARKDOWN_OPTIMIZATION_PROMPT},
            {"role": "user", "content": markdown},
            {"role": "assistant", "content": prefix, "prefix": True},
        ]


def _beta_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return normalized if normalized.endswith("/beta") else f"{normalized}/beta"


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def _strip_markdown_fence(markdown: str) -> str:
    lines = markdown.strip().splitlines()
    if lines and lines[0].strip().lower() in {"```", "```markdown", "```md"}:
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
