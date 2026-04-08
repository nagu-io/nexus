"""Process-local runtime event bus for live dashboard updates."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable


Subscriber = Callable[[dict[str, Any]], Awaitable[None] | None]


class RuntimeEventBus:
    """Fan out runtime events to any interested in-process subscribers."""

    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []

    def subscribe(self, callback: Subscriber) -> None:
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Subscriber) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    async def publish(self, event: dict[str, Any]) -> None:
        if not self._subscribers:
            return
        for callback in list(self._subscribers):
            try:
                result = callback(dict(event))
                if inspect.isawaitable(result):
                    await result
            except Exception:
                continue

    def emit(self, event: dict[str, Any]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.publish(event))


runtime_event_bus = RuntimeEventBus()

