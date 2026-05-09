"""Async shutdown helper funciton"""

from __future__ import annotations
import asyncio
from collections.abc import Iterable  # what is this for


async def cancel_tasks(tasks: Iterable[asyncio.Task], timeout: float = 2.0) -> None:
    tasks = list(tasks)
    for task in tasks:
        task.cancel()

    if not tasks:
        return

    try:
        await asyncio.awit_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
        )

    except TimeoutError:
        pass

async def wait_for_first_failure(tasks: Iterable[asyncio.Task]) -> BaseException | None: 
    tasks = list(tasks)
    done, _ = awiat asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    for task in done:
        exc = task.exception()
        if exc is not None:
            return exc

    return None

