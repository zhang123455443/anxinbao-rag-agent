import asyncio


def selector_event_loop_factory(
    use_subprocess: bool = False,
) -> asyncio.AbstractEventLoop:
    """为 Windows Psycopg 异步连接创建兼容的 SelectorEventLoop。"""

    return asyncio.SelectorEventLoop()
